"""
ECG waveform profile: average heartbeat morphology by DL-ECG risk tertile.
Uses preprocessed_mat_tiled data source.
"""

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import find_peaks, resample
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────────
ECG_DIR = Path("data/ecg_matrices/preprocessed_tiled")
M5_PATH = Path("model5_cvd_mace4/E2E_MultiTask_32_2_phase_cvd_mace4_1yr_oof.csv")
OUT_DIR = Path("results/figures/ecg_waveform_profiles_tiled")
FINAL_OUT_DIR = Path("results/manuscript_final/ecg_waveform_cvd4")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 500  # Hz
RESAMPLE_LEN = 400  # resample each beat to this many points

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_II_IDX = 1

# ── Load OOF + risk groups ──────────────────────────────────────────────────
print("Loading OOF predictions...")
oof = pd.read_csv(M5_PATH)
risk_norm = oof["oof_log_risk"].copy()
for fold in oof["test_fold"].unique():
    m = oof["test_fold"] == fold
    v = oof.loc[m, "oof_log_risk"]
    risk_norm.loc[m] = (v - v.mean()) / v.std()
oof["risk_norm"] = risk_norm

rmin, rmax = oof["risk_norm"].min(), oof["risk_norm"].max()
t1, t2 = rmin + (rmax - rmin) / 3, rmin + 2 * (rmax - rmin) / 3
oof["risk_group"] = pd.cut(oof["risk_norm"],
    bins=[rmin - 0.001, t1, t2, rmax + 0.001],
    labels=["Low", "Intermediate", "High"])


def detect_r_peaks(lead_ii, sr=500):
    """R-peak detection on Lead II."""
    min_dist = int(0.3 * sr)
    height_thr = np.percentile(lead_ii, 70)
    peaks, _ = find_peaks(lead_ii, distance=min_dist, height=height_thr)
    if len(peaks) < 3:
        peaks, _ = find_peaks(lead_ii, distance=min_dist, height=np.median(lead_ii))
    return peaks


def extract_rr_beats(ecg_12lead, r_peaks, sr=500):
    """Extract full beats using each beat's own RR interval."""
    if len(r_peaks) < 3:
        return None

    beats = []
    for i in range(1, len(r_peaks) - 1):
        start = (r_peaks[i - 1] + r_peaks[i]) // 2
        end = (r_peaks[i] + r_peaks[i + 1]) // 2

        if start < 0 or end > ecg_12lead.shape[1]:
            continue

        beat_len = end - start
        if beat_len < int(0.3 * sr) or beat_len > int(2.0 * sr):
            continue

        beat = ecg_12lead[:, start:end]
        beat_resampled = resample(beat, RESAMPLE_LEN, axis=1)
        # Baseline correction: subtract mean of first 5% samples (TP segment)
        baseline = beat_resampled[:, :int(RESAMPLE_LEN * 0.05)].mean(axis=1, keepdims=True)
        beat_resampled = beat_resampled - baseline
        beats.append(beat_resampled)

    if len(beats) < 2:
        return None
    return np.stack(beats)


# ── Collect beats per risk group ────────────────────────────────────────────
group_beats = {"Low": [], "Intermediate": [], "High": []}
n_loaded = 0
n_failed = 0

print("\nExtracting beats (RR-based) from preprocessed_mat_tiled...")
for _, row in oof.iterrows():
    fname = row["source_file"]
    grp = row["risk_group"]
    if pd.isna(grp):
        continue

    fpath = ECG_DIR / fname
    if not fpath.exists():
        n_failed += 1
        continue

    try:
        mat = sio.loadmat(str(fpath))
        ecg = mat["feats"]  # (12, 5000)
        if ecg.shape != (12, 5000):
            n_failed += 1
            continue

        r_peaks = detect_r_peaks(ecg[LEAD_II_IDX], SAMPLE_RATE)
        beats = extract_rr_beats(ecg, r_peaks, SAMPLE_RATE)
        if beats is None:
            n_failed += 1
            continue

        patient_median = np.median(beats, axis=0)
        group_beats[grp].append(patient_median)
        n_loaded += 1

    except Exception:
        n_failed += 1
        continue

    if n_loaded % 500 == 0:
        print(f"  Loaded {n_loaded} patients...")

print(f"\nLoaded: {n_loaded}, Failed: {n_failed}")
for grp, bl in group_beats.items():
    print(f"  {grp}: {len(bl)} patients")

# ── Compute group statistics ────────────────────────────────────────────────
group_mean = {}
group_sem = {}
for grp in ["Low", "Intermediate", "High"]:
    arr = np.stack(group_beats[grp])
    group_mean[grp] = arr.mean(axis=0)
    group_sem[grp] = arr.std(axis=0) / np.sqrt(len(arr))

time_pct = np.linspace(0, 100, RESAMPLE_LEN)

COLORS = {"Low": "#2ecc71", "Intermediate": "#3498db", "High": "#e74c3c"}
LEAD_LAYOUT = [
    [0, 3, 6, 9],   # I, aVR, V1, V4
    [1, 4, 7, 10],  # II, aVL, V2, V5
    [2, 5, 8, 11],  # III, aVF, V3, V6
]


def style_compact_lead_ax(ax, lead_idx):
    ax.set_title(LEAD_NAMES[lead_idx], fontsize=11.5, fontweight="bold", pad=4)
    ax.axhline(0, color="gray", linewidth=0.35, alpha=0.35)
    ax.set_xlim(0, 100)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.85)


def finish_and_save(fig, axes_flat, title, ncol, fname):
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.985)
    fig.legend(handles, labels, loc="upper center", ncol=ncol, fontsize=9.5,
               frameon=False, bbox_to_anchor=(0.5, 0.955),
               handlelength=1.8, columnspacing=1.3, handletextpad=0.5)
    fig.subplots_adjust(left=0.03, right=0.995, bottom=0.04, top=0.90,
                        wspace=0.08, hspace=0.14)
    for out_dir in [OUT_DIR, FINAL_OUT_DIR]:
        fig.savefig(out_dir / f"{fname}.png", dpi=220, bbox_inches="tight")
        fig.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}.png")


def plot_three_groups_separate(group_order, colors, title, fname):
    """Show each risk group in its own 3x4 block, with compact layout."""
    stats = {grp: (group_mean[grp], group_sem[grp], len(group_beats[grp])) for grp in group_order}

    # Per-lead limits shared across all risk-group blocks
    lead_lims = {}
    for lead_idx in range(12):
        vals = []
        for grp in group_order:
            avg, sem, _ = stats[grp]
            vals.extend((avg[lead_idx] - 1.96 * sem[lead_idx]).tolist())
            vals.extend((avg[lead_idx] + 1.96 * sem[lead_idx]).tolist())
        vmin, vmax = min(vals), max(vals)
        margin = max((vmax - vmin) * 0.10, 0.01)
        lead_lims[lead_idx] = (vmin - margin, vmax + margin)

    fig = plt.figure(figsize=(14.2, 5.0))
    outer = fig.add_gridspec(1, len(group_order), wspace=0.08)

    for block_i, grp in enumerate(group_order):
        avg, sem, n = stats[grp]
        inner = outer[block_i].subgridspec(3, 4, hspace=0.10, wspace=0.08)

        # Group title
        if grp == "Low":
            gtitle = f"Low predicted risk (n={n})"
        elif grp == "Intermediate":
            gtitle = f"Intermediate predicted risk (n={n})"
        else:
            gtitle = f"High predicted risk (n={n})"
        fig.text((block_i + 0.5) / len(group_order), 0.955, gtitle,
                 ha="center", va="top", fontsize=11.2, fontweight="normal", color=colors[grp])

        for r in range(3):
            for c in range(4):
                lead_idx = LEAD_LAYOUT[r][c]
                ax = fig.add_subplot(inner[r, c])
                ax.plot(time_pct, avg[lead_idx], color=colors[grp], linewidth=1.8)
                ax.fill_between(
                    time_pct,
                    avg[lead_idx] - 1.96 * sem[lead_idx],
                    avg[lead_idx] + 1.96 * sem[lead_idx],
                    color=colors[grp], alpha=0.18
                )
                ax.axhline(0, color="gray", linewidth=0.30, alpha=0.30)
                ax.set_xlim(0, 100)
                ax.set_ylim(*lead_lims[lead_idx])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.tick_params(length=0)
                ax.set_title(LEAD_NAMES[lead_idx], fontsize=9.8, pad=1.5)
                for spine in ax.spines.values():
                    spine.set_linewidth(0.75)

    fig.subplots_adjust(left=0.015, right=0.992, bottom=0.055, top=0.875)
    for out_dir in [OUT_DIR, FINAL_OUT_DIR]:
        fig.savefig(out_dir / f"{fname}.png", dpi=240, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {fname}.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: 12-lead comparison (3x4 grid)
# ═══════════════════════════════════════════════════════════════════════════
plot_three_groups_separate(
    ["Low", "Intermediate", "High"],
    COLORS,
    "CVD Composite 4: Average ECG by DL-ECG risk group",
    "ecg_waveform_12lead_risk_groups"
)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Key leads (II, V1, V5) — larger
# ═══════════════════════════════════════════════════════════════════════════
key_leads = [(1, "Lead II"), (6, "Lead V1"), (10, "Lead V5")]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (lead_idx, lead_name) in zip(axes, key_leads):
    for grp in ["Low", "Intermediate", "High"]:
        avg = group_mean[grp][lead_idx]
        sem = group_sem[grp][lead_idx]
        n = len(group_beats[grp])
        ax.plot(time_pct, avg, color=COLORS[grp], linewidth=2,
                label=f"{grp} (n={n})")
        ax.fill_between(time_pct, avg - 1.96 * sem, avg + 1.96 * sem,
                        color=COLORS[grp], alpha=0.15)

    ax.set_title(lead_name, fontsize=13, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.3, alpha=0.4)
    ax.axvline(35, color="gray", linewidth=0.4, linestyle=":", alpha=0.4)
    ax.set_xlabel("Beat cycle (%)", fontsize=11)
    ax.set_ylabel("Amplitude (normalized)", fontsize=11)
    ax.set_xlim(0, 100)
    ax.legend(fontsize=9)

fig.suptitle("Average single-beat morphology by DL-ECG risk group (tiled)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_DIR / "ecg_waveform_key_leads_risk_groups.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT_DIR / "ecg_waveform_key_leads_risk_groups.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved: ecg_waveform_key_leads_risk_groups.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Low vs High — overlay per lead
# ═══════════════════════════════════════════════════════════════════════════
def plot_two_group_comparison(grp_a, grp_b, color_a, color_b, title, fname):
    fig, axes = plt.subplots(3, 4, figsize=(12.2, 8.6), sharex=True)
    axes_flat = axes.flatten()

    for lead_idx in range(12):
        ax = axes_flat[lead_idx]
        for grp, color in [(grp_a, color_a), (grp_b, color_b)]:
            avg = group_mean[grp][lead_idx]
            sem = group_sem[grp][lead_idx]
            n = len(group_beats[grp])
            ax.plot(time_pct, avg, color=color, linewidth=1.8,
                    label=f"{grp} (n={n})")
            ax.fill_between(time_pct, avg - 1.96 * sem, avg + 1.96 * sem,
                            color=color, alpha=0.15)

        style_compact_lead_ax(ax, lead_idx)

    finish_and_save(fig, axes_flat, title, 2, fname)


plot_two_group_comparison(
    "Low", "High", "#2ecc71", "#e74c3c",
    "Average single-beat ECG: Low-risk vs High-risk (tiled)",
    "ecg_waveform_low_vs_high"
)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Intermediate vs High — overlay per lead
# ═══════════════════════════════════════════════════════════════════════════
plot_two_group_comparison(
    "Intermediate", "High", "#3498db", "#e74c3c",
    "Average single-beat ECG: Intermediate-risk vs High-risk (tiled)",
    "ecg_waveform_intermediate_vs_high"
)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Intermediate+High merged vs Low
# ═══════════════════════════════════════════════════════════════════════════
merged_beats = group_beats["Intermediate"] + group_beats["High"]
arr_merged = np.stack(merged_beats)
merged_mean = arr_merged.mean(axis=0)
merged_sem = arr_merged.std(axis=0) / np.sqrt(len(arr_merged))
n_merged = len(merged_beats)

fig, axes = plt.subplots(3, 4, figsize=(12.2, 8.6), sharex=True)
axes_flat = axes.flatten()

for lead_idx in range(12):
    ax = axes_flat[lead_idx]
    # Low
    avg_low = group_mean["Low"][lead_idx]
    sem_low = group_sem["Low"][lead_idx]
    n_low = len(group_beats["Low"])
    ax.plot(time_pct, avg_low, color="#2ecc71", linewidth=1.8,
            label=f"Low (n={n_low})")
    ax.fill_between(time_pct, avg_low - 1.96 * sem_low, avg_low + 1.96 * sem_low,
                    color="#2ecc71", alpha=0.15)
    # Intermediate + High
    ax.plot(time_pct, merged_mean[lead_idx], color="#e74c3c", linewidth=1.8,
            label=f"Intermediate+High (n={n_merged})")
    ax.fill_between(time_pct,
                    merged_mean[lead_idx] - 1.96 * merged_sem[lead_idx],
                    merged_mean[lead_idx] + 1.96 * merged_sem[lead_idx],
                    color="#e74c3c", alpha=0.15)

    style_compact_lead_ax(ax, lead_idx)

finish_and_save(
    fig, axes_flat,
    "CVD Composite 4: Low-risk vs Intermediate+High-risk",
    2, "ecg_waveform_low_vs_intermediate_high"
)

print("\nDone.")

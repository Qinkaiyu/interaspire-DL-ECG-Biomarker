"""
ECG waveform profile: average heartbeat morphology by DL-ECG risk tertile.
Uses mat_zero_filled (raw, unnormalized) data source.
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
ECG_DIR = Path("data/ecg_matrices/zero_filled")
M5_PATH = Path("model5_cvd_mace4/E2E_MultiTask_32_2_phase_cvd_mace4_1yr_oof.csv")
OUT_DIR = Path("results/figures/ecg_waveform_profiles_raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
        beats.append(beat_resampled)

    if len(beats) < 2:
        return None
    return np.stack(beats)


# ── Collect beats per risk group ────────────────────────────────────────────
group_beats = {"Low": [], "Intermediate": [], "High": []}
n_loaded = 0
n_failed = 0

print("\nExtracting beats (RR-based) from mat_zero_filled (raw)...")
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
        ecg = mat["feats"]  # (5000, 12) in mat_zero_filled — need transpose
        if ecg.shape == (5000, 12):
            ecg = ecg.T  # -> (12, 5000)
        elif ecg.shape != (12, 5000):
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

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: 12-lead comparison (3x4 grid)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 4, figsize=(18, 12), sharex=True)
axes_flat = axes.flatten()

for lead_idx in range(12):
    ax = axes_flat[lead_idx]
    for grp in ["Low", "Intermediate", "High"]:
        avg = group_mean[grp][lead_idx]
        sem = group_sem[grp][lead_idx]
        n = len(group_beats[grp])
        ax.plot(time_pct, avg, color=COLORS[grp], linewidth=1.5,
                label=f"{grp} (n={n})")
        ax.fill_between(time_pct, avg - 1.96 * sem, avg + 1.96 * sem,
                        color=COLORS[grp], alpha=0.15)

    ax.set_title(LEAD_NAMES[lead_idx], fontsize=12, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.3, alpha=0.4)
    if lead_idx % 4 == 0:
        ax.set_ylabel("Amplitude (uV)", fontsize=10)
    if lead_idx >= 8:
        ax.set_xlabel("Beat cycle (%)", fontsize=10)
    ax.set_xlim(0, 100)

handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=11,
           bbox_to_anchor=(0.5, 1.01))
fig.suptitle("Average single-beat ECG morphology by DL-ECG risk tertile (raw signal)\n"
             "(median beat per patient, mean +/- 95% CI across patients)",
             fontsize=14, fontweight="bold", y=1.04)
plt.tight_layout()
fig.savefig(OUT_DIR / "ecg_waveform_12lead_risk_groups.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT_DIR / "ecg_waveform_12lead_risk_groups.pdf", bbox_inches="tight")
plt.close(fig)
print("\nSaved: ecg_waveform_12lead_risk_groups.png")

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
    ax.set_xlabel("Beat cycle (%)", fontsize=11)
    ax.set_ylabel("Amplitude (uV)", fontsize=11)
    ax.set_xlim(0, 100)
    ax.legend(fontsize=9)

fig.suptitle("Average single-beat morphology by DL-ECG risk group (raw signal)",
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
    fig, axes = plt.subplots(3, 4, figsize=(18, 12), sharex=True)
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

        ax.set_title(LEAD_NAMES[lead_idx], fontsize=12, fontweight="bold")
        ax.axhline(0, color="gray", linewidth=0.3, alpha=0.4)
        ax.set_xlim(0, 100)
        if lead_idx % 4 == 0:
            ax.set_ylabel("Amplitude (uV)", fontsize=10)
        if lead_idx >= 8:
            ax.set_xlabel("Beat cycle (%)", fontsize=10)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=12,
               bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.04)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"{fname}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}.png")


plot_two_group_comparison(
    "Low", "High", "#2ecc71", "#e74c3c",
    "Average single-beat ECG: Low-risk vs High-risk (raw signal)",
    "ecg_waveform_low_vs_high"
)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Intermediate vs High — overlay per lead
# ═══════════════════════════════════════════════════════════════════════════
plot_two_group_comparison(
    "Intermediate", "High", "#3498db", "#e74c3c",
    "Average single-beat ECG: Intermediate-risk vs High-risk (raw signal)",
    "ecg_waveform_intermediate_vs_high"
)

print("\nDone.")

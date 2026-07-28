"""
ECG waveform profile for manuscript: clean, compact layout.

Generates:
  1. Side-by-side panel (Low-risk blue | High-risk red) — reference style
  2. Overlaid version (Low vs High on same subplots)
  3. Overlaid version (Low vs Intermediate+High on same subplots)

Uses clinical standard ECG lead layout:
  Row 1: I,   aVR, V1, V4
  Row 2: II,  aVL, V2, V5
  Row 3: III, aVF, V3, V6
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
OUT_DIR = Path("results/manuscript_final/ecg_waveform_cvd4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 500
RESAMPLE_LEN = 400
MAX_TRACES = 80  # max individual patient traces to show per group

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_II_IDX = 1

# Clinical standard ECG layout: (row, col) -> lead index
CLINICAL_LAYOUT = [
    [0, 3, 6, 9],    # I,   aVR, V1, V4
    [1, 4, 7, 10],   # II,  aVL, V2, V5
    [2, 5, 8, 11],   # III, aVF, V3, V6
]


def detect_r_peaks(lead_ii, sr=500):
    min_dist = int(0.3 * sr)
    height_thr = np.percentile(lead_ii, 70)
    peaks, _ = find_peaks(lead_ii, distance=min_dist, height=height_thr)
    if len(peaks) < 3:
        peaks, _ = find_peaks(lead_ii, distance=min_dist, height=np.median(lead_ii))
    return peaks


def extract_rr_beats(ecg_12lead, r_peaks, sr=500):
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
        baseline = beat_resampled[:, :int(RESAMPLE_LEN * 0.05)].mean(axis=1, keepdims=True)
        beat_resampled = beat_resampled - baseline
        beats.append(beat_resampled)
    if len(beats) < 2:
        return None
    return np.stack(beats)


# ── Load data ─────────────────────────────────────────────────────────────
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

# ── Extract beats ─────────────────────────────────────────────────────────
group_beats = {"Low": [], "Intermediate": [], "High": []}
n_loaded = 0
n_failed = 0

print("\nExtracting beats...")
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
        ecg = mat["feats"]
        if ecg.shape != (12, 5000):
            n_failed += 1
            continue
        r_peaks = detect_r_peaks(ecg[LEAD_II_IDX], SAMPLE_RATE)
        beats = extract_rr_beats(ecg, r_peaks, SAMPLE_RATE)
        if beats is None:
            n_failed += 1
            continue
        patient_median = np.median(beats, axis=0)  # (12, RESAMPLE_LEN)
        group_beats[grp].append(patient_median)
        n_loaded += 1
    except Exception:
        n_failed += 1
        continue
    if n_loaded % 500 == 0:
        print(f"  Loaded {n_loaded}...")

print(f"\nLoaded: {n_loaded}, Failed: {n_failed}")
for grp, bl in group_beats.items():
    print(f"  {grp}: {len(bl)} patients")

# ── Compute stats ─────────────────────────────────────────────────────────
group_mean = {}
group_arr = {}
for grp in ["Low", "Intermediate", "High"]:
    arr = np.stack(group_beats[grp])  # (N, 12, RESAMPLE_LEN)
    group_arr[grp] = arr
    group_mean[grp] = arr.mean(axis=0)

time_x = np.linspace(0, 100, RESAMPLE_LEN)


def subsample_traces(arr, max_n=MAX_TRACES):
    """Randomly subsample patient traces for display."""
    if len(arr) <= max_n:
        return arr
    rng = np.random.default_rng(42)
    idx = rng.choice(len(arr), size=max_n, replace=False)
    return arr[idx]


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Side-by-side (Low = blue left | High = red right)
# ═══════════════════════════════════════════════════════════════════════════
print("\nGenerating side-by-side figure...")

COLOR_LOW = "#1f4e79"      # dark blue
COLOR_HIGH = "#c0392b"     # dark red
COLOR_LOW_BG = "#a3c4e0"   # light blue for individual traces
COLOR_HIGH_BG = "#e8a6a0"  # light red for individual traces

fig, axes = plt.subplots(3, 8, figsize=(16, 6.5),
                         gridspec_kw={"wspace": 0.08, "hspace": 0.35})

# Subsample traces
traces_low = subsample_traces(group_arr["Low"])
traces_high = subsample_traces(group_arr["High"])

n_low = len(group_arr["Low"])
n_high = len(group_arr["High"])

for row_idx in range(3):
    for col_idx in range(4):
        lead_idx = CLINICAL_LAYOUT[row_idx][col_idx]
        lead_name = LEAD_NAMES[lead_idx]

        # ── Left panel: Low risk (blue) ──
        ax_l = axes[row_idx, col_idx]

        # Individual traces
        for trace in traces_low:
            ax_l.plot(time_x, trace[lead_idx], color=COLOR_LOW_BG,
                      linewidth=0.3, alpha=0.3)
        # Mean
        ax_l.plot(time_x, group_mean["Low"][lead_idx],
                  color=COLOR_LOW, linewidth=1.2)

        ax_l.set_title(lead_name, fontsize=9, fontweight="bold", pad=2)
        ax_l.set_xlim(0, 100)
        ax_l.set_xticks([])
        ax_l.set_yticks([])
        ax_l.spines["top"].set_visible(False)
        ax_l.spines["right"].set_visible(False)
        ax_l.spines["bottom"].set_visible(False)
        ax_l.spines["left"].set_visible(False)

        # ── Right panel: High risk (red) ──
        ax_r = axes[row_idx, col_idx + 4]

        for trace in traces_high:
            ax_r.plot(time_x, trace[lead_idx], color=COLOR_HIGH_BG,
                      linewidth=0.3, alpha=0.3)
        ax_r.plot(time_x, group_mean["High"][lead_idx],
                  color=COLOR_HIGH, linewidth=1.2)

        ax_r.set_title(lead_name, fontsize=9, fontweight="bold", pad=2)
        ax_r.set_xlim(0, 100)
        ax_r.set_xticks([])
        ax_r.set_yticks([])
        ax_r.spines["top"].set_visible(False)
        ax_r.spines["right"].set_visible(False)
        ax_r.spines["bottom"].set_visible(False)
        ax_r.spines["left"].set_visible(False)

        # Match y-axis range between left/right for same lead
        y_all = np.concatenate([traces_low[:, lead_idx].ravel(),
                                traces_high[:, lead_idx].ravel()])
        ymin, ymax = np.percentile(y_all, 1), np.percentile(y_all, 99)
        margin = (ymax - ymin) * 0.1
        ax_l.set_ylim(ymin - margin, ymax + margin)
        ax_r.set_ylim(ymin - margin, ymax + margin)

# Panel titles
fig.text(0.27, 0.97, f"Low risk (n={n_low})", fontsize=12, fontweight="bold",
         color=COLOR_LOW, ha="center", va="top")
fig.text(0.75, 0.97, f"High risk (n={n_high})", fontsize=12, fontweight="bold",
         color=COLOR_HIGH, ha="center", va="top")

# Vertical separator
fig.patches.append(plt.Rectangle((0.498, 0.03), 0.004, 0.9,
                                  transform=fig.transFigure,
                                  facecolor="#cccccc", edgecolor="none"))

plt.subplots_adjust(top=0.92, bottom=0.02, left=0.02, right=0.98)
fig.savefig(OUT_DIR / "ecg_waveform_sidebyside.png", dpi=300, bbox_inches="tight",
            facecolor="white")
fig.savefig(OUT_DIR / "ecg_waveform_sidebyside.pdf", bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print("  Saved: ecg_waveform_sidebyside.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Overlaid — Low vs High on same subplots
# ═══════════════════════════════════════════════════════════════════════════
print("\nGenerating overlaid Low vs High figure...")

fig, axes = plt.subplots(3, 4, figsize=(10, 6.5),
                         gridspec_kw={"wspace": 0.08, "hspace": 0.35})

for row_idx in range(3):
    for col_idx in range(4):
        lead_idx = CLINICAL_LAYOUT[row_idx][col_idx]
        lead_name = LEAD_NAMES[lead_idx]
        ax = axes[row_idx, col_idx]

        # Individual traces — Low
        for trace in traces_low:
            ax.plot(time_x, trace[lead_idx], color=COLOR_LOW_BG,
                    linewidth=0.2, alpha=0.2)
        # Individual traces — High
        for trace in traces_high:
            ax.plot(time_x, trace[lead_idx], color=COLOR_HIGH_BG,
                    linewidth=0.2, alpha=0.2)
        # Mean — Low
        ax.plot(time_x, group_mean["Low"][lead_idx],
                color=COLOR_LOW, linewidth=1.2)
        # Mean — High
        ax.plot(time_x, group_mean["High"][lead_idx],
                color=COLOR_HIGH, linewidth=1.2)

        ax.set_title(lead_name, fontsize=9, fontweight="bold", pad=2)
        ax.set_xlim(0, 100)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        y_all = np.concatenate([traces_low[:, lead_idx].ravel(),
                                traces_high[:, lead_idx].ravel()])
        ymin, ymax = np.percentile(y_all, 1), np.percentile(y_all, 99)
        margin = (ymax - ymin) * 0.1
        ax.set_ylim(ymin - margin, ymax + margin)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=COLOR_LOW, linewidth=2, label=f"Low risk (n={n_low})"),
    Line2D([0], [0], color=COLOR_HIGH, linewidth=2, label=f"High risk (n={n_high})"),
]
fig.legend(handles=legend_elements, loc="upper center", ncol=2, fontsize=10,
           frameon=False, bbox_to_anchor=(0.5, 1.0))

plt.subplots_adjust(top=0.90, bottom=0.02, left=0.02, right=0.98)
fig.savefig(OUT_DIR / "ecg_waveform_overlaid_low_vs_high.png", dpi=300,
            bbox_inches="tight", facecolor="white")
fig.savefig(OUT_DIR / "ecg_waveform_overlaid_low_vs_high.pdf",
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print("  Saved: ecg_waveform_overlaid_low_vs_high.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Overlaid — Low vs Intermediate+High on same subplots
# ═══════════════════════════════════════════════════════════════════════════
print("\nGenerating overlaid Low vs Intermediate+High figure...")

merged_arr = np.concatenate([group_arr["Intermediate"], group_arr["High"]], axis=0)
merged_mean = merged_arr.mean(axis=0)
n_merged = len(merged_arr)
traces_merged = subsample_traces(merged_arr)

fig, axes = plt.subplots(3, 4, figsize=(10, 6.5),
                         gridspec_kw={"wspace": 0.08, "hspace": 0.35})

for row_idx in range(3):
    for col_idx in range(4):
        lead_idx = CLINICAL_LAYOUT[row_idx][col_idx]
        lead_name = LEAD_NAMES[lead_idx]
        ax = axes[row_idx, col_idx]

        # Individual traces — Low
        for trace in traces_low:
            ax.plot(time_x, trace[lead_idx], color=COLOR_LOW_BG,
                    linewidth=0.2, alpha=0.2)
        # Individual traces — Intermediate+High
        for trace in traces_merged:
            ax.plot(time_x, trace[lead_idx], color=COLOR_HIGH_BG,
                    linewidth=0.2, alpha=0.2)
        # Mean — Low
        ax.plot(time_x, group_mean["Low"][lead_idx],
                color=COLOR_LOW, linewidth=1.2)
        # Mean — Intermediate+High
        ax.plot(time_x, merged_mean[lead_idx],
                color=COLOR_HIGH, linewidth=1.2)

        ax.set_title(lead_name, fontsize=9, fontweight="bold", pad=2)
        ax.set_xlim(0, 100)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        y_all = np.concatenate([traces_low[:, lead_idx].ravel(),
                                traces_merged[:, lead_idx].ravel()])
        ymin, ymax = np.percentile(y_all, 1), np.percentile(y_all, 99)
        margin = (ymax - ymin) * 0.1
        ax.set_ylim(ymin - margin, ymax + margin)

legend_elements = [
    Line2D([0], [0], color=COLOR_LOW, linewidth=2, label=f"Low risk (n={n_low})"),
    Line2D([0], [0], color=COLOR_HIGH, linewidth=2,
           label=f"Intermediate+High risk (n={n_merged})"),
]
fig.legend(handles=legend_elements, loc="upper center", ncol=2, fontsize=10,
           frameon=False, bbox_to_anchor=(0.5, 1.0))

plt.subplots_adjust(top=0.90, bottom=0.02, left=0.02, right=0.98)
fig.savefig(OUT_DIR / "ecg_waveform_overlaid_low_vs_inthigh.png", dpi=300,
            bbox_inches="tight", facecolor="white")
fig.savefig(OUT_DIR / "ecg_waveform_overlaid_low_vs_inthigh.pdf",
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print("  Saved: ecg_waveform_overlaid_low_vs_inthigh.png")

print("\nDone.")

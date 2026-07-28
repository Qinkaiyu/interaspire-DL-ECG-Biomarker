"""
ECG waveform profiles for all endpoints:
  - CVD Composite 4: DL-ECG risk tertile (Low vs Intermediate+High)
  - CVD Composite 6: DL-ECG risk tertile (Low vs Intermediate+High)
  - Component endpoints: Event vs No-event (CV death, MI, HF, Stroke)

Uses preprocessed_mat_tiled with per-beat baseline correction.
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
DATA_PATH = Path("data/INTERASPIRE_analysis_dataset.xlsx")
M5_CVD4 = Path("model5_cvd_mace4/E2E_MultiTask_32_2_phase_cvd_mace4_1yr_oof.csv")
M5_CVD6 = Path("model5_cvd_mace6/E2E_MultiTask_32_2_phase_cvd_mace6_1yr_oof.csv")
FIG_BASE = Path("results/figures")

SAMPLE_RATE = 500
RESAMPLE_LEN = 400

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_II_IDX = 1


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


# ── Load all ECG median beats keyed by source_file ─────────────────────────
# We compute median beat once, then reuse for all analyses.

# Collect all unique source_files across CVD4 and CVD6
oof4 = pd.read_csv(M5_CVD4)
oof6 = pd.read_csv(M5_CVD6)
all_files = set(oof4["source_file"].tolist()) | set(oof6["source_file"].tolist())

print(f"Loading ECG beats for {len(all_files)} unique files...")
median_beats = {}  # source_file -> (12, RESAMPLE_LEN)
n_loaded = 0
n_failed = 0

for fname in all_files:
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
        median_beats[fname] = np.median(beats, axis=0)
        n_loaded += 1
    except Exception:
        n_failed += 1
        continue
    if n_loaded % 500 == 0:
        print(f"  Loaded {n_loaded}...")

print(f"Loaded: {n_loaded}, Failed: {n_failed}")

time_pct = np.linspace(0, 100, RESAMPLE_LEN)


# ── Plotting helpers ───────────────────────────────────────────────────────
def compute_stats(beat_list):
    arr = np.stack(beat_list)
    return arr.mean(axis=0), arr.std(axis=0) / np.sqrt(len(arr)), len(arr)


def plot_two_groups(beats_a, beats_b, label_a, label_b, color_a, color_b,
                    title, out_dir, fname):
    mean_a, sem_a, n_a = compute_stats(beats_a)
    mean_b, sem_b, n_b = compute_stats(beats_b)

    fig, axes = plt.subplots(3, 4, figsize=(18, 12), sharex=True)
    axes_flat = axes.flatten()

    for lead_idx in range(12):
        ax = axes_flat[lead_idx]
        for avg, sem, n, label, color in [
            (mean_a, sem_a, n_a, f"{label_a} (n={n_a})", color_a),
            (mean_b, sem_b, n_b, f"{label_b} (n={n_b})", color_b),
        ]:
            ax.plot(time_pct, avg[lead_idx], color=color, linewidth=1.8, label=label)
            ax.fill_between(time_pct,
                            avg[lead_idx] - 1.96 * sem[lead_idx],
                            avg[lead_idx] + 1.96 * sem[lead_idx],
                            color=color, alpha=0.15)
        ax.set_title(LEAD_NAMES[lead_idx], fontsize=12, fontweight="bold")
        ax.axhline(0, color="gray", linewidth=0.3, alpha=0.4)
        ax.set_xlim(0, 100)
        if lead_idx % 4 == 0:
            ax.set_ylabel("Amplitude", fontsize=10)
        if lead_idx >= 8:
            ax.set_xlabel("Beat cycle (%)", fontsize=10)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=12,
               bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.04)
    plt.tight_layout()
    fig.savefig(out_dir / f"{fname}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}.png")


def plot_three_groups(group_dict, colors, title, out_dir, fname):
    """group_dict: {label: beat_list}"""
    stats = {k: compute_stats(v) for k, v in group_dict.items()}

    fig, axes = plt.subplots(3, 4, figsize=(18, 12), sharex=True)
    axes_flat = axes.flatten()

    for lead_idx in range(12):
        ax = axes_flat[lead_idx]
        for grp, (avg, sem, n) in stats.items():
            ax.plot(time_pct, avg[lead_idx], color=colors[grp], linewidth=1.5,
                    label=f"{grp} (n={n})")
            ax.fill_between(time_pct,
                            avg[lead_idx] - 1.96 * sem[lead_idx],
                            avg[lead_idx] + 1.96 * sem[lead_idx],
                            color=colors[grp], alpha=0.15)
        ax.set_title(LEAD_NAMES[lead_idx], fontsize=12, fontweight="bold")
        ax.axhline(0, color="gray", linewidth=0.3, alpha=0.4)
        ax.set_xlim(0, 100)
        if lead_idx % 4 == 0:
            ax.set_ylabel("Amplitude", fontsize=10)
        if lead_idx >= 8:
            ax.set_xlabel("Beat cycle (%)", fontsize=10)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.04)
    plt.tight_layout()
    fig.savefig(out_dir / f"{fname}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}.png")


# ── Helper: assign risk tertile ────────────────────────────────────────────
def assign_risk_groups(oof_df):
    risk_norm = oof_df["oof_log_risk"].copy()
    for fold in oof_df["test_fold"].unique():
        m = oof_df["test_fold"] == fold
        v = oof_df.loc[m, "oof_log_risk"]
        risk_norm.loc[m] = (v - v.mean()) / v.std()
    oof_df = oof_df.copy()
    oof_df["risk_norm"] = risk_norm
    rmin, rmax = risk_norm.min(), risk_norm.max()
    t1 = rmin + (rmax - rmin) / 3
    t2 = rmin + 2 * (rmax - rmin) / 3
    oof_df["risk_group"] = pd.cut(risk_norm,
        bins=[rmin - 0.001, t1, t2, rmax + 0.001],
        labels=["Low", "Intermediate", "High"])
    return oof_df


def collect_beats_by_group(oof_df, group_col="risk_group"):
    groups = {}
    for _, row in oof_df.iterrows():
        grp = row[group_col]
        fname = row["source_file"]
        if pd.isna(grp) or fname not in median_beats:
            continue
        grp_str = str(grp)
        if grp_str not in groups:
            groups[grp_str] = []
        groups[grp_str].append(median_beats[fname])
    return groups


# ═══════════════════════════════════════════════════════════════════════════
# 1. CVD Composite 4 — risk tertile
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== CVD Composite 4 ===")
out_cvd4 = FIG_BASE / "ecg_waveform_cvd4_tiled"
out_cvd4.mkdir(parents=True, exist_ok=True)

oof4 = assign_risk_groups(oof4)
g4 = collect_beats_by_group(oof4)
for k, v in g4.items():
    print(f"  {k}: {len(v)}")

colors3 = {"Low": "#2ecc71", "Intermediate": "#3498db", "High": "#e74c3c"}

# 3-group
plot_three_groups(g4, colors3,
    "CVD Composite 4: Average ECG by DL-ECG risk tertile",
    out_cvd4, "ecg_waveform_12lead_risk_groups")

# Low vs High
plot_two_groups(g4["Low"], g4["High"], "Low", "High", "#2ecc71", "#e74c3c",
    "CVD Composite 4: Low-risk vs High-risk",
    out_cvd4, "ecg_waveform_low_vs_high")

# Intermediate vs High
plot_two_groups(g4["Intermediate"], g4["High"], "Intermediate", "High",
    "#3498db", "#e74c3c",
    "CVD Composite 4: Intermediate-risk vs High-risk",
    out_cvd4, "ecg_waveform_intermediate_vs_high")

# Low vs Intermediate+High merged
plot_two_groups(g4["Low"], g4["Intermediate"] + g4["High"],
    "Low", "Intermediate+High", "#2ecc71", "#e74c3c",
    "CVD Composite 4: Low-risk vs Intermediate+High-risk",
    out_cvd4, "ecg_waveform_low_vs_intermediate_high")


# ═══════════════════════════════════════════════════════════════════════════
# 2. CVD Composite 6 — risk tertile
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== CVD Composite 6 ===")
out_cvd6 = FIG_BASE / "ecg_waveform_cvd6_tiled"
out_cvd6.mkdir(parents=True, exist_ok=True)

oof6 = assign_risk_groups(oof6)
g6 = collect_beats_by_group(oof6)
for k, v in g6.items():
    print(f"  {k}: {len(v)}")

plot_three_groups(g6, colors3,
    "CVD Composite 6: Average ECG by DL-ECG risk tertile",
    out_cvd6, "ecg_waveform_12lead_risk_groups")

plot_two_groups(g6["Low"], g6["High"], "Low", "High", "#2ecc71", "#e74c3c",
    "CVD Composite 6: Low-risk vs High-risk",
    out_cvd6, "ecg_waveform_low_vs_high")

plot_two_groups(g6["Intermediate"], g6["High"], "Intermediate", "High",
    "#3498db", "#e74c3c",
    "CVD Composite 6: Intermediate-risk vs High-risk",
    out_cvd6, "ecg_waveform_intermediate_vs_high")

plot_two_groups(g6["Low"], g6["Intermediate"] + g6["High"],
    "Low", "Intermediate+High", "#2ecc71", "#e74c3c",
    "CVD Composite 6: Low-risk vs Intermediate+High-risk",
    out_cvd6, "ecg_waveform_low_vs_intermediate_high")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Component endpoints — Event vs No-event
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Component Endpoints ===")

# Load clinical data for component derivation
df = pd.read_excel(DATA_PATH, engine="openpyxl")
# Merge with CVD4 OOF to get source_file
oof4_slim = pd.read_csv(M5_CVD4)[["STUDYID", "source_file"]]
if "source_file" in df.columns:
    df = df.drop(columns=["source_file"])
df = df.merge(oof4_slim, on="STUDYID", how="inner")

first_comp = df["first_component_Composite_4"].fillna("")
df["comp_cv_death"] = ((first_comp == "SURVIVAL") & (df["cvd_death_flag"] == 1)).astype(int)
df["comp_mi"] = (first_comp == "HOSPAMI").astype(int)
df["comp_hf"] = (first_comp == "HOSPHF").astype(int)
df["comp_stroke"] = (first_comp == "HOSPSTROKE").astype(int)

components = [
    ("CV Death", "comp_cv_death"),
    ("MI", "comp_mi"),
    ("HF", "comp_hf"),
    ("Stroke", "comp_stroke"),
]

for comp_name, comp_col in components:
    out_comp = FIG_BASE / f"ecg_waveform_{comp_col}_tiled"
    out_comp.mkdir(parents=True, exist_ok=True)

    event_beats = []
    noevent_beats = []
    for _, row in df.iterrows():
        fname = row["source_file"]
        if fname not in median_beats:
            continue
        if row[comp_col] == 1:
            event_beats.append(median_beats[fname])
        else:
            noevent_beats.append(median_beats[fname])

    print(f"\n  {comp_name}: Event={len(event_beats)}, No-event={len(noevent_beats)}")

    if len(event_beats) < 5:
        print(f"  Skipping {comp_name}: too few events")
        continue

    plot_two_groups(noevent_beats, event_beats,
        "No event", comp_name, "#2ecc71", "#e74c3c",
        f"{comp_name}: Average ECG morphology (Event vs No-event)",
        out_comp, "ecg_waveform_event_vs_noevent")

    # Low vs Intermediate+High — restricted to EVENT patients only
    low_beats_event = []
    inthigh_beats_event = []
    for _, row in df.iterrows():
        fname = row["source_file"]
        if fname not in median_beats:
            continue
        if row[comp_col] != 1:
            continue  # only event patients
        sid = row["STUDYID"]
        risk_row = oof4.loc[oof4["STUDYID"] == sid]
        if risk_row.empty:
            continue
        grp = risk_row.iloc[0]["risk_group"]
        if pd.isna(grp):
            continue
        if grp == "Low":
            low_beats_event.append(median_beats[fname])
        else:
            inthigh_beats_event.append(median_beats[fname])

    print(f"  {comp_name} event risk split: Low={len(low_beats_event)}, Int+High={len(inthigh_beats_event)}")

    if len(low_beats_event) >= 3 and len(inthigh_beats_event) >= 3:
        plot_two_groups(low_beats_event, inthigh_beats_event,
            "Low", "Intermediate+High", "#2ecc71", "#e74c3c",
            f"{comp_name} patients: Low-risk vs Intermediate+High-risk ECG",
            out_comp, "ecg_waveform_low_vs_intermediate_high")
    else:
        print(f"  Skipping risk split for {comp_name}: too few in one group")

print("\nAll done.")

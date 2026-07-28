"""
Fig. 6 v2: Subgroup analysis — AIRE Fig. 4A style
Multiple panels, each subgroup category as a panel.
Dots for each subgroup level × model. Horizontal C-index axis.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from lifelines.utils import concordance_index

BASE = Path(".")
DATA_PATH = BASE / "data" / "INTERASPIRE_analysis_dataset.xlsx"
OOF_M04 = BASE / "results" / "oof_predictions" / "oof_CVD4.csv"
OOF_M5 = BASE / "model5_cvd_mace4" / "E2E_MultiTask_32_2_phase_cvd_mace4_1yr_oof.csv"
OUT_DIR = BASE / "results" / "figures"

evt_col = "event_CVD_Composite_4"
time_col = "time_Composite_4"

# ── Load & merge ─────────────────────────────────────────────────────────────
raw = pd.read_excel(DATA_PATH, engine="openpyxl")
df04 = pd.read_csv(OOF_M04)
df5 = pd.read_csv(OOF_M5)
df5 = df5.rename(columns={"oof_log_risk": "Model_5_risk", "test_fold": "fold5"})

merged = df04.merge(df5[["STUDYID", "Model_5_risk", "fold5"]], on="STUDYID", how="inner")

raw_cols = ["STUDYID", "RQSEX", "AGE", "COUNTRY", "RQINDEX",
            "Sinus Rhythm", "Atrial Fibrillation", "Atrial Flutter",
            "Supraventricular Tachycardia", "RBBB", "LBBB",
            "1 AV Block", "2 AV Block", "3 AV Block",
            "Left Axis Deviation", "Right Axis Deviation", "Q Wave",
            "ST Elevation", "ST Depression", "T Wave Inversion",
            "Ischaemic", "LVH", "RVH", "QT Prolongation", "MI (Old)", "MI(Acute)"]
raw_cols = [c for c in raw_cols if c in raw.columns]
merged = merged.merge(raw[raw_cols], on="STUDYID", how="left")

# ── Derived columns ──────────────────────────────────────────────────────────
merged["age_group"] = np.where(merged["AGE"] < 65, "<65 years", "\u226565 years")

merged["idx_type"] = merged["RQINDEX"].map({
    "Acute myocardial infarction STEMI": "STEMI",
    "Acute myocardial infarction Non-STEMI": "NSTEMI",
    "Unstable angina / Acute myocardial ischaemia": "UA",
    "Elective percutaneous transluminal coronary angioplasty": "Elective PCI",
    "Elective coronary artery by-pass surgery": "Elective CABG",
})

abnormal_cols = ["Atrial Fibrillation", "Atrial Flutter", "Supraventricular Tachycardia",
                 "RBBB", "LBBB", "1 AV Block", "2 AV Block", "3 AV Block",
                 "Left Axis Deviation", "Right Axis Deviation", "Q Wave",
                 "ST Elevation", "ST Depression", "T Wave Inversion",
                 "Ischaemic", "LVH", "RVH", "QT Prolongation", "MI (Old)", "MI(Acute)"]
has_ecg = merged["Sinus Rhythm"].notna()
is_normal = (merged["Sinus Rhythm"] == "Yes")
for col in abnormal_cols:
    if col in merged.columns:
        is_normal = is_normal & ((merged[col] == "No") | merged[col].isna())
merged["ecg_type"] = "No ECG"
merged.loc[has_ecg & is_normal, "ecg_type"] = "Normal ECG"
merged.loc[has_ecg & ~is_normal, "ecg_type"] = "Abnormal ECG"

# ── Models ───────────────────────────────────────────────────────────────────
MODEL_DEFS = [
    ("Model 3", "Model_3_cox_risk", "fold"),
    ("Model 4", "Model_4_cox_risk", "fold"),
    ("Model 5", "Model_5_risk", "fold5"),
]

COLORS = {"Model 3": "#FF9500", "Model 4": "#AF52DE", "Model 5": "#FF3B30"}
MARKERS = {"Model 3": "s", "Model 4": "D", "Model 5": "o"}
SIZES = {"Model 3": 40, "Model 4": 40, "Model 5": 55}

# ── Subgroup panels ──────────────────────────────────────────────────────────
PANELS = [
    ("A  Sex", "RQSEX", None),
    ("B  Age", "age_group", None),
    ("C  Country", "COUNTRY", None),
    ("D  Index event", "idx_type", None),
    ("E  ECG status", "ecg_type", ["Normal ECG", "Abnormal ECG"]),
]

# ── Compute C-index per subgroup ─────────────────────────────────────────────
def compute_mean_fold_c(data, cox_col, fold_col):
    fold_cs = []
    for fid in sorted(data[fold_col].unique()):
        sf = data[data[fold_col] == fid]
        y = sf[evt_col].values.astype(float)
        t = sf[time_col].values.astype(float)
        r = sf[cox_col].values
        v = ~np.isnan(r)
        if v.sum() < 10 or len(np.unique(y[v])) < 2:
            continue
        try:
            fold_cs.append(concordance_index(np.clip(t[v], 0.5, None), -r[v], y[v]))
        except Exception:
            continue
    if not fold_cs:
        return np.nan, np.nan, np.nan
    mean_c = np.mean(fold_cs)
    sd_c = np.std(fold_cs, ddof=1) if len(fold_cs) > 1 else 0
    n_folds = len(fold_cs)
    ci_lo = mean_c - 1.96 * sd_c / np.sqrt(n_folds)
    ci_hi = mean_c + 1.96 * sd_c / np.sqrt(n_folds)
    return mean_c, ci_lo, ci_hi


# Compute all
all_results = {}
for panel_title, col, subset_vals in PANELS:
    if subset_vals:
        values = subset_vals
    else:
        values = sorted(merged[col].dropna().unique())

    panel_data = []
    for val in values:
        sub = merged[merged[col] == val]
        n = len(sub)
        ev = int(sub[evt_col].sum())
        if ev < 5:
            continue

        for mname, cox_col, fold_col in MODEL_DEFS:
            c, lo, hi = compute_mean_fold_c(sub, cox_col, fold_col)
            panel_data.append({
                "value": val, "n": n, "events": ev,
                "model": mname, "c": c, "c_lo": lo, "c_hi": hi,
            })

    all_results[panel_title] = pd.DataFrame(panel_data)

# ── Figure ───────────────────────────────────────────────────────────────────
n_panels = len(all_results)
fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 6), sharey=False)

for idx, (panel_title, panel_df) in enumerate(all_results.items()):
    ax = axes[idx]

    if panel_df.empty:
        ax.set_title(panel_title, fontsize=10, fontweight="bold")
        continue

    # Get unique subgroup values (sorted by Model 5 C desc)
    m5_order = panel_df[panel_df["model"] == "Model 5"].sort_values("c", ascending=True)
    values_ordered = m5_order["value"].tolist()

    y_positions = {v: i for i, v in enumerate(values_ordered)}
    offsets = {"Model 3": -0.2, "Model 4": 0.0, "Model 5": 0.2}

    for _, row in panel_df.iterrows():
        if row["value"] not in y_positions:
            continue
        if np.isnan(row["c"]):
            continue

        yp = y_positions[row["value"]] + offsets[row["model"]]
        xerr_lo = row["c"] - row["c_lo"]
        xerr_hi = row["c_hi"] - row["c"]

        # Scale marker by sample size
        base_size = SIZES[row["model"]]

        ax.errorbar(
            row["c"], yp,
            xerr=[[max(xerr_lo, 0)], [max(xerr_hi, 0)]],
            fmt=MARKERS[row["model"]], color=COLORS[row["model"]],
            markersize=6, capsize=3, capthick=1,
            elinewidth=1.2, markeredgewidth=0, zorder=5,
            alpha=0.85,
        )

    # Y labels with n and events
    ytick_labels = []
    for v in values_ordered:
        sub_info = panel_df[panel_df["value"] == v].iloc[0]
        ytick_labels.append(f"{v}\n(n={sub_info['n']}, {sub_info['events']}ev)")

    ax.set_yticks(range(len(values_ordered)))
    ax.set_yticklabels(ytick_labels, fontsize=7.5)
    ax.axvline(0.5, color="grey", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("C-index", fontsize=9)
    ax.set_title(panel_title, fontsize=10, fontweight="bold")
    ax.set_xlim(0.35, 0.95)
    ax.grid(True, axis="x", alpha=0.15)
    ax.tick_params(axis="x", labelsize=8)

# Legend on last panel
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["Model 3"],
           markersize=8, label="Model 3 (Clinical)"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=COLORS["Model 4"],
           markersize=8, label="Model 4 (+ ECG biomarkers)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["Model 5"],
           markersize=8, label="Model 5 (DL-ECG)"),
]
axes[-1].legend(handles=legend_elements, loc="lower right", fontsize=7.5,
                frameon=True, fancybox=False, edgecolor="#ccc")

plt.tight_layout(w_pad=1.5)
fig.savefig(OUT_DIR / "fig6_subgroup_CVD4.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "fig6_subgroup_CVD4.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSaved -> {OUT_DIR / 'fig6_subgroup_CVD4.png'}")

# ── Print summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 100)
for panel_title, panel_df in all_results.items():
    print(f"\n{panel_title}")
    values = panel_df["value"].unique()
    for v in values:
        sub = panel_df[panel_df["value"] == v]
        n = sub.iloc[0]["n"]
        ev = sub.iloc[0]["events"]
        m3 = sub[sub["model"] == "Model 3"]["c"].values
        m4 = sub[sub["model"] == "Model 4"]["c"].values
        m5 = sub[sub["model"] == "Model 5"]["c"].values
        m3s = f"{m3[0]:.3f}" if len(m3) and not np.isnan(m3[0]) else "—"
        m4s = f"{m4[0]:.3f}" if len(m4) and not np.isnan(m4[0]) else "—"
        m5s = f"{m5[0]:.3f}" if len(m5) and not np.isnan(m5[0]) else "—"
        print(f"  {v:<20s} n={n:5d} ev={ev:3d}  M3={m3s}  M4={m4s}  M5={m5s}")

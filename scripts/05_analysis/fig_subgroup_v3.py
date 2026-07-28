"""
Fig. 6 v3: Subgroup analysis — AIRE Fig. 4A style
Y-axis: C-index, X-axis: subgroup levels (dots)
Multiple panels, one per subgroup category.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
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
merged["age_group"] = np.where(merged["AGE"] < 65, "<65", "\u226565")

merged["idx_type"] = merged["RQINDEX"].map({
    "Acute myocardial infarction STEMI": "STEMI",
    "Acute myocardial infarction Non-STEMI": "NSTEMI",
    "Unstable angina / Acute myocardial ischaemia": "UA",
    "Elective percutaneous transluminal coronary angioplasty": "ePCI",
    "Elective coronary artery by-pass surgery": "eCABG",
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
merged.loc[has_ecg & is_normal, "ecg_type"] = "Normal"
merged.loc[has_ecg & ~is_normal, "ecg_type"] = "Abnormal"

# ── Models ───────────────────────────────────────────────────────────────────
MODEL_DEFS = [
    ("Model 3", "Model_3_cox_risk", "fold"),
    ("Model 4", "Model_4_cox_risk", "fold"),
    ("Model 5", "Model_5_risk", "fold5"),
]

COLORS = {"Model 3": "#FF9500", "Model 4": "#AF52DE", "Model 5": "#FF3B30"}
MARKERS = {"Model 3": "s", "Model 4": "D", "Model 5": "o"}

# ── Panels ───────────────────────────────────────────────────────────────────
PANELS = [
    ("A  Sex", "RQSEX", ["Male", "Female"]),
    ("B  Age", "age_group", ["<65", "\u226565"]),
    ("C  Country", "COUNTRY", None),  # auto from data, filter >=5 events
    ("D  Index event", "idx_type", ["STEMI", "NSTEMI", "UA", "ePCI", "eCABG"]),
    ("E  ECG status", "ecg_type", ["Normal", "Abnormal"]),
]

# ── Compute C-index ──────────────────────────────────────────────────────────
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
    ci_lo = mean_c - 1.96 * sd_c / np.sqrt(len(fold_cs))
    ci_hi = mean_c + 1.96 * sd_c / np.sqrt(len(fold_cs))
    return mean_c, ci_lo, ci_hi


all_panel_data = {}
for panel_title, col, subset_vals in PANELS:
    if subset_vals is None:
        vals = sorted(merged[col].dropna().unique())
    else:
        vals = subset_vals

    rows = []
    for val in vals:
        sub = merged[merged[col] == val]
        n = len(sub)
        ev = int(sub[evt_col].sum())
        if ev < 5:
            continue
        for mname, cox_col_m, fold_col in MODEL_DEFS:
            c, lo, hi = compute_mean_fold_c(sub, cox_col_m, fold_col)
            rows.append({"value": val, "n": n, "events": ev,
                         "model": mname, "c": c, "c_lo": lo, "c_hi": hi})
    all_panel_data[panel_title] = pd.DataFrame(rows)

# ── Figure ───────────────────────────────────────────────────────────────────
n_panels = len(all_panel_data)
# Variable width based on number of subgroup levels
widths = []
for pt, pdf in all_panel_data.items():
    n_vals = pdf["value"].nunique() if not pdf.empty else 1
    widths.append(max(n_vals * 0.8, 1.5))

fig, axes = plt.subplots(1, n_panels, figsize=(sum(widths) + 2, 5.5),
                          gridspec_kw={"width_ratios": widths})

model_offsets = {"Model 3": -0.2, "Model 4": 0.0, "Model 5": 0.2}

for idx, (panel_title, panel_df) in enumerate(all_panel_data.items()):
    ax = axes[idx]

    if panel_df.empty:
        ax.set_title(panel_title, fontsize=10, fontweight="bold")
        continue

    values = panel_df["value"].unique()
    x_map = {v: i for i, v in enumerate(values)}

    for _, row in panel_df.iterrows():
        if np.isnan(row["c"]):
            continue
        x = x_map[row["value"]] + model_offsets[row["model"]]

        ax.plot(
            x, row["c"],
            marker=MARKERS[row["model"]], color=COLORS[row["model"]],
            markersize=8, markeredgewidth=0, zorder=5,
            alpha=0.85, linestyle="none",
        )

    # X-axis labels
    tick_labels = []
    for v in values:
        info = panel_df[panel_df["value"] == v].iloc[0]
        tick_labels.append(f"{v}\n({info['n']},{info['events']}ev)")

    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(tick_labels, fontsize=6.5, rotation=0, ha="center")
    ax.axhline(0.5, color="grey", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.set_ylim(0.35, 0.95)
    ax.grid(True, axis="y", alpha=0.15)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_title(panel_title, fontsize=10, fontweight="bold")

    if idx == 0:
        ax.set_ylabel("C-index", fontsize=11)
    else:
        ax.set_yticklabels([])

# Legend with overall C-index
from matplotlib.lines import Line2D

# Get overall C-index for each model from first panel data (Sex panel has "Overall" equivalent)
# Compute overall C for legend
overall_cs = {}
for mname, cox_col_m, fold_col in MODEL_DEFS:
    c, lo, hi = compute_mean_fold_c(merged, cox_col_m, fold_col)
    overall_cs[mname] = (c, lo, hi)

legend_elements = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["Model 3"],
           markersize=8, label=f'Model 3 (Clinical) C={overall_cs["Model 3"][0]:.3f}'),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=COLORS["Model 4"],
           markersize=8, label=f'Model 4 (+ ECG biomarkers) C={overall_cs["Model 4"][0]:.3f}'),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["Model 5"],
           markersize=8, label=f'Model 5 (DL-ECG) C={overall_cs["Model 5"][0]:.3f}'),
]
axes[0].legend(handles=legend_elements, loc="lower left", fontsize=7.5,
               frameon=True, fancybox=False, edgecolor="#ccc")

plt.tight_layout(w_pad=0.5)
fig.savefig(OUT_DIR / "fig6_subgroup_CVD4.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "fig6_subgroup_CVD4.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved -> {OUT_DIR / 'fig6_subgroup_CVD4.png'}")

"""
Fig. 6: Subgroup analysis — Mean-fold C-index by subgroup
AIRE Fig. 4A + Raghunath Fig. 3C style
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
OUT_DIR.mkdir(parents=True, exist_ok=True)

evt_col = "event_CVD_Composite_4"
time_col = "time_Composite_4"

# ── Load & merge ─────────────────────────────────────────────────────────────
raw = pd.read_excel(DATA_PATH, engine="openpyxl")
df04 = pd.read_csv(OOF_M04)
df5 = pd.read_csv(OOF_M5)
df5 = df5.rename(columns={"oof_log_risk": "Model_5_risk", "test_fold": "fold5"})

merged = df04.merge(df5[["STUDYID", "Model_5_risk", "fold5"]], on="STUDYID", how="inner")

# Merge raw columns for subgroup definitions
raw_cols = ["STUDYID", "RQSEX", "AGE", "COUNTRY", "RQINDEX",
            "Sinus Rhythm", "Atrial Fibrillation", "Atrial Flutter",
            "Supraventricular Tachycardia", "RBBB", "LBBB",
            "1 AV Block", "2 AV Block", "3 AV Block",
            "Left Axis Deviation", "Right Axis Deviation", "Q Wave",
            "ST Elevation", "ST Depression", "T Wave Inversion",
            "Ischaemic", "LVH", "RVH", "QT Prolongation", "MI (Old)", "MI(Acute)"]
raw_cols = [c for c in raw_cols if c in raw.columns]
merged = merged.merge(raw[raw_cols], on="STUDYID", how="left")

# ── Define subgroups ─────────────────────────────────────────────────────────
# Age
merged["age_group"] = np.where(merged["AGE"] < 65, "<65 years", "\u226565 years")

# Region
asia = ["Singapore", "China", "Malaysia", "Philippines", "Indonesia", "UAE"]
europe = ["Poland", "Portugal"]
latam = ["Colombia", "Argentina"]
africa = ["Egypt", "Nigeria", "Tanzania", "Kenia"]
merged["region"] = merged["COUNTRY"].apply(
    lambda x: "Asia" if x in asia else "Europe" if x in europe
    else "Latin America" if x in latam else "Africa" if x in africa else "Other"
)

# Index event simplified
merged["idx_type"] = merged["RQINDEX"].map({
    "Acute myocardial infarction STEMI": "STEMI",
    "Acute myocardial infarction Non-STEMI": "NSTEMI",
    "Unstable angina / Acute myocardial ischaemia": "UA",
    "Elective percutaneous transluminal coronary angioplasty": "Elective PCI",
    "Elective coronary artery by-pass surgery": "Elective CABG",
})

# Normal ECG
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

# ── Subgroup definitions ─────────────────────────────────────────────────────
SUBGROUPS = [
    ("Overall", "overall", ["All"]),
    ("Sex", "RQSEX", ["Male", "Female"]),
    ("Age", "age_group", ["<65 years", "\u226565 years"]),
    ("Region", "region", ["Asia", "Africa", "Europe", "Latin America"]),
    ("Index event", "idx_type", ["STEMI", "NSTEMI", "UA", "Elective PCI"]),
    ("ECG status", "ecg_type", ["Normal ECG", "Abnormal ECG"]),
]

# Models to compare
MODEL_DEFS = [
    ("Model 3", "Model_3_cox_risk", "fold"),
    ("Model 4", "Model_4_cox_risk", "fold"),
    ("Model 5", "Model_5_risk", "fold5"),
]

COLORS_MODEL = {"Model 3": "#FF9500", "Model 4": "#AF52DE", "Model 5": "#FF3B30"}
MARKERS = {"Model 3": "s", "Model 4": "D", "Model 5": "o"}

# ── Compute mean-fold C-index per subgroup per model ─────────────────────────
rows = []

for sg_label, sg_col, sg_values in SUBGROUPS:
    for sg_val in sg_values:
        if sg_val == "All":
            sub = merged
        else:
            sub = merged[merged[sg_col] == sg_val]

        n = len(sub)
        ev = int(sub[evt_col].sum())

        for mname, cox_col, fold_col in MODEL_DEFS:
            fold_cs = []
            for fid in sorted(sub[fold_col].unique()):
                sf = sub[sub[fold_col] == fid]
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

            mean_c = np.mean(fold_cs) if fold_cs else np.nan
            sd_c = np.std(fold_cs, ddof=1) if len(fold_cs) > 1 else 0
            ci_lo = mean_c - 1.96 * sd_c / np.sqrt(len(fold_cs)) if fold_cs else np.nan
            ci_hi = mean_c + 1.96 * sd_c / np.sqrt(len(fold_cs)) if fold_cs else np.nan

            rows.append({
                "category": sg_label,
                "subgroup": sg_val,
                "n": n, "events": ev,
                "model": mname,
                "c": mean_c, "c_lo": ci_lo, "c_hi": ci_hi,
            })

res = pd.DataFrame(rows)

# Print summary
print(f"{'Category':<14s} {'Subgroup':<16s} {'N':>5s} {'Ev':>4s} | ", end="")
for mname, _, _ in MODEL_DEFS:
    print(f"{mname:>22s}", end="")
print()
print("-" * 90)

prev_cat = ""
for sg_label, sg_col, sg_values in SUBGROUPS:
    for sg_val in sg_values:
        sub_res = res[(res["category"] == sg_label) & (res["subgroup"] == sg_val)]
        n = sub_res.iloc[0]["n"]
        ev = sub_res.iloc[0]["events"]
        cat_str = sg_label if sg_label != prev_cat else ""
        prev_cat = sg_label
        print(f"{cat_str:<14s} {sg_val:<16s} {n:5d} {ev:4d} | ", end="")
        for mname, _, _ in MODEL_DEFS:
            r = sub_res[sub_res["model"] == mname].iloc[0]
            if np.isnan(r["c"]):
                print(f"{'  —':>22s}", end="")
            else:
                print(f"  {r['c']:.3f} ({r['c_lo']:.3f}-{r['c_hi']:.3f})", end="")
        print()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE: Forest-plot style
# ═══════════════════════════════════════════════════════════════════════════════

# Build row list for plotting
plot_rows = []
for sg_label, sg_col, sg_values in SUBGROUPS:
    plot_rows.append({"type": "header", "label": sg_label})
    for sg_val in sg_values:
        sub_res = res[(res["category"] == sg_label) & (res["subgroup"] == sg_val)]
        n = sub_res.iloc[0]["n"]
        ev = sub_res.iloc[0]["events"]
        plot_rows.append({
            "type": "data",
            "label": f"  {sg_val}  (n={n}, {ev} ev)",
            "data": {m: sub_res[sub_res["model"] == m].iloc[0] for m in ["Model 3", "Model 4", "Model 5"]},
        })

n_rows = len(plot_rows)
fig, ax = plt.subplots(figsize=(10, 0.45 * n_rows + 1.5))

y_pos = list(range(n_rows))[::-1]
offsets = {"Model 3": -0.15, "Model 4": 0.0, "Model 5": 0.15}

for i, row in enumerate(plot_rows):
    if row["type"] == "header":
        ax.text(-0.02, y_pos[i], row["label"], fontsize=10, fontweight="bold",
                transform=ax.get_yaxis_transform(), va="center", ha="right")
    else:
        ax.text(-0.02, y_pos[i], row["label"], fontsize=8.5,
                transform=ax.get_yaxis_transform(), va="center", ha="right")

        for mname in ["Model 3", "Model 4", "Model 5"]:
            r = row["data"][mname]
            if np.isnan(r["c"]):
                continue
            yerr_lo = r["c"] - r["c_lo"]
            yerr_hi = r["c_hi"] - r["c"]
            ax.errorbar(
                r["c"], y_pos[i] + offsets[mname],
                xerr=[[yerr_lo], [yerr_hi]],
                fmt=MARKERS[mname], color=COLORS_MODEL[mname],
                markersize=6, capsize=3, capthick=1.2,
                elinewidth=1.5, markeredgewidth=0, zorder=5,
            )

# Reference line
ax.axvline(0.5, color="grey", linestyle="--", alpha=0.3, linewidth=0.8)

ax.set_yticks(y_pos)
ax.set_yticklabels([""] * n_rows)
ax.set_xlabel("C-index (mean fold ± 95% CI)", fontsize=11)
ax.set_xlim(0.35, 0.90)
ax.grid(True, axis="x", alpha=0.15)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", labelsize=10)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS_MODEL["Model 3"],
           markersize=8, label="Model 3 (Clinical)"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=COLORS_MODEL["Model 4"],
           markersize=8, label="Model 4 (+ ECG biomarkers)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS_MODEL["Model 5"],
           markersize=8, label="Model 5 (DL-ECG)"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
          frameon=True, fancybox=False, edgecolor="#ccc")

ax.set_title("Subgroup analysis: C-index by patient characteristics",
             fontsize=12, fontweight="bold", loc="left")

plt.tight_layout()
fig.savefig(OUT_DIR / "fig6_subgroup_CVD4.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "fig6_subgroup_CVD4.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSaved -> {OUT_DIR / 'fig6_subgroup_CVD4.png'}")

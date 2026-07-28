"""
Fig. 7 v2: ECG biomarker association with DL-ECG predicted risk
AIRE Fig. 6C / PheWAS style — correlation plot colored by ECG category.
Shows which ECG features drive the DL model's predictions.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, pointbiserialr

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

# Normalize per fold
for fid in sorted(merged["fold5"].unique()):
    mask = merged["fold5"] == fid
    vals = merged.loc[mask, "Model_5_risk"]
    merged.loc[mask, "Model_5_risk"] = (vals - vals.mean()) / vals.std()

merged = merged.merge(raw[["STUDYID"] + [c for c in raw.columns if c in [
    "Sinus Rhythm", "Atrial Fibrillation", "Atrial Flutter",
    "Supraventricular Tachycardia",
    "RBBB", "LBBB", "1 AV Block", "2 AV Block", "3 AV Block",
    "Left Axis Deviation", "Right Axis Deviation", "Q Wave",
    "ST Elevation", "ST Depression", "T Wave Inversion",
    "Ischaemic", "LVH", "RVH", "QT Prolongation", "QT Shortening",
    "MI (Old)", "MI(Acute)",
    "PR Interval (ms)", "QRS Duration (ms)", "QT Interval (ms)",
]]], on="STUDYID", how="left")

# ── ECG biomarkers with categories ───────────────────────────────────────────
ECG_FEATURES = [
    # (column, display_name, category)
    # Rhythm
    ("Sinus Rhythm", "Sinus rhythm", "Rhythm"),
    ("Atrial Fibrillation", "Atrial fibrillation", "Rhythm"),
    ("Atrial Flutter", "Atrial flutter", "Rhythm"),

    # Conduction
    ("LBBB", "Left bundle branch block", "Conduction"),
    ("RBBB", "Right bundle branch block", "Conduction"),
    ("1 AV Block", "First-degree AV block", "Conduction"),
    ("Left Axis Deviation", "Left axis deviation", "Conduction"),
    ("Right Axis Deviation", "Right axis deviation", "Conduction"),

    # Ischemia / Infarction
    ("ST Elevation", "ST elevation", "Ischemia"),
    ("ST Depression", "ST depression", "Ischemia"),
    ("T Wave Inversion", "T wave inversion", "Ischemia"),
    ("Ischaemic", "Ischaemic changes", "Ischemia"),
    ("Q Wave", "Q wave", "Ischemia"),
    ("MI (Old)", "Old MI", "Ischemia"),
    ("MI(Acute)", "Acute MI", "Ischemia"),

    # Structural
    ("LVH", "LV hypertrophy", "Structural"),
    ("RVH", "RV hypertrophy", "Structural"),
    ("QT Prolongation", "QT prolongation", "Repolarization"),

    # Intervals (continuous)
    ("PR Interval (ms)", "PR interval", "Intervals"),
    ("QRS Duration (ms)", "QRS duration", "Intervals"),
    ("QT Interval (ms)", "QT interval", "Intervals"),
]

CATEGORY_COLORS = {
    "Rhythm": "#3498db",
    "Conduction": "#e67e22",
    "Ischemia": "#e74c3c",
    "Structural": "#2ecc71",
    "Repolarization": "#9b59b6",
    "Intervals": "#1abc9c",
}

# ── Compute correlations ─────────────────────────────────────────────────────
results = []

for col, display, category in ECG_FEATURES:
    if col not in merged.columns:
        continue

    # Binary → encode
    if merged[col].dtype == object:
        vals = merged[col].map({"Yes": 1, "No": 0})
    else:
        vals = pd.to_numeric(merged[col], errors="coerce")

    valid = vals.notna() & merged["Model_5_risk"].notna()
    if valid.sum() < 50:
        continue

    x = vals[valid].values.astype(float)
    y = merged.loc[valid, "Model_5_risk"].values

    # Use point-biserial for binary, Spearman for continuous
    is_binary = len(np.unique(x[~np.isnan(x)])) <= 2
    if is_binary:
        corr, p = pointbiserialr(x, y)
    else:
        corr, p = spearmanr(x, y)

    n_pos = int((x > 0).sum()) if is_binary else int(valid.sum())
    prevalence = n_pos / valid.sum() * 100 if is_binary else np.nan

    results.append({
        "col": col, "display": display, "category": category,
        "corr": corr, "p": p,
        "n": int(valid.sum()), "prevalence": prevalence,
        "is_binary": is_binary,
    })

res = pd.DataFrame(results)

# Print
print(f"\n{'Biomarker':<28s} {'Category':<14s} {'Corr':>7s} {'P':>10s} {'Prev':>7s}")
print("-" * 70)
for _, r in res.sort_values("corr", ascending=False).iterrows():
    p_str = f"<0.001" if r["p"] < 0.001 else f"{r['p']:.3f}"
    prev = f"{r['prevalence']:.1f}%" if r["is_binary"] else "cont."
    print(f"  {r['display']:<26s} {r['category']:<14s} {r['corr']:+7.3f} {p_str:>10s} {prev:>7s}")

# ── Sort by correlation ──────────────────────────────────────────────────────
res = res.sort_values("corr", ascending=True).reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE: AIRE Fig. 6C style — horizontal dot plot with category colors
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 7))

y_pos = np.arange(len(res))

for i, (_, row) in enumerate(res.iterrows()):
    color = CATEGORY_COLORS.get(row["category"], "#999")
    # Marker size proportional to -log10(p)
    sig = min(-np.log10(max(row["p"], 1e-20)), 15)
    size = max(sig * 4, 20)

    ax.scatter(
        row["corr"], y_pos[i],
        s=size, c=color, edgecolors="white", linewidths=0.5,
        zorder=5, alpha=0.85,
    )

# Reference line at 0
ax.axvline(0, color="grey", linestyle="-", linewidth=0.8, alpha=0.4)

# Labels
ax.set_yticks(y_pos)
ax.set_yticklabels(res["display"].values, fontsize=9)
ax.set_xlabel("Correlation with DL-ECG predicted risk", fontsize=11)
ax.set_title("Association of ECG biomarkers with\nDL-ECG risk score (Model 5)",
             fontsize=12, fontweight="bold")
ax.grid(True, axis="x", alpha=0.15)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", labelsize=10)

# Annotations for significant ones
for i, (_, row) in enumerate(res.iterrows()):
    if row["p"] < 0.05:
        p_str = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*"
        side = 1 if row["corr"] > 0 else -1
        ax.text(row["corr"] + side * 0.01, y_pos[i],
                p_str, fontsize=8, va="center",
                ha="left" if row["corr"] > 0 else "right",
                color=CATEGORY_COLORS.get(row["category"], "#999"),
                fontweight="bold")

# Category legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
           markersize=8, label=cat)
    for cat, c in CATEGORY_COLORS.items()
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=8,
          frameon=True, fancybox=False, edgecolor="#ccc", framealpha=0.95,
          title="ECG category", title_fontsize=9)

# Axis labels
ax.text(-0.02, -0.06, "Negative correlation\n(lower predicted risk)",
        transform=ax.transAxes, fontsize=7.5, ha="left", color="#666", fontstyle="italic")
ax.text(0.98, -0.06, "Positive correlation\n(higher predicted risk)",
        transform=ax.transAxes, fontsize=7.5, ha="right", color="#666", fontstyle="italic")

plt.tight_layout()
fig.savefig(OUT_DIR / "fig7_ecg_association_CVD4.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "fig7_ecg_association_CVD4.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSaved -> {OUT_DIR / 'fig7_ecg_association_CVD4.png'}")
res.to_csv(OUT_DIR / "fig7_ecg_association_CVD4_data.csv", index=False)

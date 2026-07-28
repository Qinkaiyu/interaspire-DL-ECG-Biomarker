"""
Fig. 7 v3: PheWAS-style scatter — AIRE Fig. 6B
X-axis: phenotypes (spread by category)
Y-axis: correlation coefficient with Model 5 risk
Color: category
Label key phenotypes.
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

merged = merged.merge(raw, on="STUDYID", how="left", suffixes=("", "_raw"))

# ── Define ALL phenotypes by category ────────────────────────────────────────
# (column, display_name, category, encoding)
# encoding: "binary_yes" = Yes/No, "binary_01" = 0/1, "continuous" = numeric

PHENOTYPES = [
    # Demographics
    ("AGE", "Age", "Demographics", "continuous"),
    ("RQSEX", "Male sex", "Demographics", "binary_male"),
    ("RQDCBMI", "BMI", "Demographics", "continuous"),
    ("RQDCSYSBP", "Systolic BP", "Demographics", "continuous"),
    ("RQDCDIASBP", "Diastolic BP", "Demographics", "continuous"),

    # Lab values
    ("CVARLABHBA1CPERCENT", "HbA1c", "Lab values", "continuous"),
    ("CVAREGFR", "eGFR", "Lab values", "continuous"),
    ("CVARLABHDL", "HDL cholesterol", "Lab values", "continuous"),
    ("CVARLABLDLFRIEDEWALD", "LDL cholesterol", "Lab values", "continuous"),
    ("CVARLABTC", "Total cholesterol", "Lab values", "continuous"),

    # Medical history
    ("RQHISTOFHYPER", "Hypertension hx", "Medical history", "binary_yes"),
    ("RQHISTOFDIAB", "Diabetes hx", "Medical history", "binary_yes"),
    ("RQHISTOFHEARTFAIL", "Heart failure hx", "Medical history", "binary_yes"),
    ("RQHISTOFCORARTDIS", "CAD hx", "Medical history", "binary_yes"),
    ("RQHISTOFCOPD", "COPD hx", "Medical history", "binary_yes"),
    ("RQHISTOFPRECVD", "Prior CVD hx", "Medical history", "binary_yes"),
    ("any_af", "Atrial fibrillation hx", "Medical history", "binary_01"),

    # Medications
    ("RQASPIOROTHER", "Antiplatelet", "Medications", "binary_yes"),
    ("RQANTICOAGULANTS", "Anticoagulant", "Medications", "binary_yes"),
    ("RQANTIHYPDRUGS", "Antihypertensive", "Medications", "binary_yes"),
    ("CVARMEDBPLOWERING", "BP-lowering drug", "Medications", "binary_yes"),
    ("CVARMEDLIPIDLOWERING", "Lipid-lowering drug", "Medications", "binary_yes"),
    ("CVARMEDGLUCOSELOWERING", "Glucose-lowering drug", "Medications", "binary_yes"),

    # ECG — Rhythm
    ("Sinus Rhythm", "Sinus rhythm", "ECG parameters", "binary_yes"),
    ("Atrial Fibrillation", "ECG: AF", "ECG parameters", "binary_yes"),
    ("Atrial Flutter", "ECG: Atrial flutter", "ECG parameters", "binary_yes"),

    # ECG — Conduction
    ("LBBB", "ECG: LBBB", "ECG parameters", "binary_yes"),
    ("RBBB", "ECG: RBBB", "ECG parameters", "binary_yes"),
    ("1 AV Block", "ECG: 1st AV block", "ECG parameters", "binary_yes"),
    ("Left Axis Deviation", "ECG: Left axis dev", "ECG parameters", "binary_yes"),
    ("Right Axis Deviation", "ECG: Right axis dev", "ECG parameters", "binary_yes"),

    # ECG — Ischemia
    ("ST Elevation", "ECG: ST elevation", "ECG parameters", "binary_yes"),
    ("ST Depression", "ECG: ST depression", "ECG parameters", "binary_yes"),
    ("T Wave Inversion", "ECG: T wave inversion", "ECG parameters", "binary_yes"),
    ("Ischaemic", "ECG: Ischaemic", "ECG parameters", "binary_yes"),
    ("Q Wave", "ECG: Q wave", "ECG parameters", "binary_yes"),
    ("MI (Old)", "ECG: Old MI", "ECG parameters", "binary_yes"),
    ("MI(Acute)", "ECG: Acute MI", "ECG parameters", "binary_yes"),

    # ECG — Structural / Repolarization
    ("LVH", "ECG: LVH", "ECG parameters", "binary_yes"),
    ("RVH", "ECG: RVH", "ECG parameters", "binary_yes"),
    ("QT Prolongation", "ECG: QT prolongation", "ECG parameters", "binary_yes"),

    # ECG — Intervals
    ("PR Interval (ms)", "ECG: PR interval", "ECG parameters", "continuous"),
    ("QRS Duration (ms)", "ECG: QRS duration", "ECG parameters", "continuous"),
    ("QT Interval (ms)", "ECG: QT interval", "ECG parameters", "continuous"),

    # Index event
    ("idx_STEMI", "Index: STEMI", "Index event", "binary_01_derived"),
    ("idx_NSTEMI", "Index: NSTEMI", "Index event", "binary_01_derived"),
    ("idx_UA", "Index: UA", "Index event", "binary_01_derived"),
    ("idx_ePCI", "Index: Elective PCI", "Index event", "binary_01_derived"),
]

CATEGORY_COLORS = {
    "Demographics": "#636EFA",
    "Lab values": "#EF553B",
    "Medical history": "#00CC96",
    "Medications": "#AB63FA",
    "ECG parameters": "#FFA15A",
    "Index event": "#19D3F3",
}

CATEGORY_ORDER = ["Demographics", "Lab values", "Medical history",
                  "Medications", "ECG parameters", "Index event"]

# ── Derive index event columns ───────────────────────────────────────────────
merged["idx_STEMI"] = (merged["RQINDEX"] == "Acute myocardial infarction STEMI").astype(float)
merged["idx_NSTEMI"] = (merged["RQINDEX"] == "Acute myocardial infarction Non-STEMI").astype(float)
merged["idx_UA"] = (merged["RQINDEX"] == "Unstable angina / Acute myocardial ischaemia").astype(float)
merged["idx_ePCI"] = (merged["RQINDEX"] == "Elective percutaneous transluminal coronary angioplasty").astype(float)

# ── Compute correlations ─────────────────────────────────────────────────────
results = []

for col, display, category, encoding in PHENOTYPES:
    if col not in merged.columns:
        continue

    if encoding == "binary_yes":
        vals = merged[col].map({"Yes": 1, "No": 0})
    elif encoding == "binary_male":
        vals = merged[col].map({"Male": 1, "Female": 0}) if merged[col].dtype == object else merged[col]
    elif encoding in ("binary_01", "binary_01_derived"):
        vals = merged[col].astype(float)
    else:
        vals = pd.to_numeric(merged[col], errors="coerce")

    valid = vals.notna() & merged["Model_5_risk"].notna()
    if valid.sum() < 50:
        continue

    x = vals[valid].values.astype(float)
    y = merged.loc[valid, "Model_5_risk"].values

    is_binary = len(np.unique(x[~np.isnan(x)])) <= 2
    if is_binary:
        if (x == 1).sum() < 5 or (x == 0).sum() < 5:
            continue
        corr, p = pointbiserialr(x, y)
    else:
        corr, p = spearmanr(x, y)

    results.append({
        "col": col, "display": display, "category": category,
        "corr": corr, "p": p, "n": int(valid.sum()),
    })

res = pd.DataFrame(results)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE: PheWAS scatter
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 6))

x_pos = 0
x_positions = []
x_ticks = []
x_tick_positions = []

for cat in CATEGORY_ORDER:
    cat_data = res[res["category"] == cat].sort_values("corr", ascending=False)
    if cat_data.empty:
        continue

    cat_start = x_pos
    for _, row in cat_data.iterrows():
        color = CATEGORY_COLORS[cat]
        # Size by significance
        sig = min(-np.log10(max(row["p"], 1e-20)), 15)
        size = max(sig * 8, 15)

        alpha = 0.9 if row["p"] < 0.05 else 0.3

        ax.scatter(x_pos, row["corr"], s=size, c=color,
                   alpha=alpha, edgecolors="white", linewidths=0.3, zorder=5)

        x_positions.append((x_pos, row))
        x_pos += 1

    # Category label position
    cat_mid = (cat_start + x_pos - 1) / 2
    x_ticks.append(cat)
    x_tick_positions.append(cat_mid)

    # Separator
    if cat != CATEGORY_ORDER[-1]:
        ax.axvline(x_pos - 0.5, color="#ddd", linewidth=0.5, zorder=1)

    x_pos += 1  # gap between categories

# Reference line
ax.axhline(0, color="grey", linewidth=0.8, alpha=0.4)

# Label top/bottom phenotypes with adjustable text to avoid overlap
LABEL_THRESHOLD = 0.15  # only label if |corr| > threshold
labeled = set()
label_data = []
for xp, row in x_positions:
    if abs(row["corr"]) > LABEL_THRESHOLD and row["display"] not in labeled:
        label_data.append((xp, row["corr"], row["display"], row["category"]))
        labeled.add(row["display"])

# Use adjustText if available, otherwise manual offset
try:
    from adjustText import adjust_text
    texts = []
    for xp, yp, display, cat in label_data:
        t = ax.text(xp, yp, display, fontsize=6.5,
                    color=CATEGORY_COLORS[cat], fontweight="bold")
        texts.append(t)
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.5),
                force_text=(0.3, 0.5), force_points=(0.3, 0.5),
                expand_text=(1.2, 1.4), expand_points=(1.2, 1.4))
except ImportError:
    # Manual: stagger labels with alternating offsets
    for i, (xp, yp, display, cat) in enumerate(label_data):
        if yp > 0:
            offset_y = 0.015 + (i % 3) * 0.012
            va = "bottom"
        else:
            offset_y = -0.015 - (i % 3) * 0.012
            va = "top"
        ax.annotate(
            display, (xp, yp),
            xytext=(xp, yp + offset_y),
            fontsize=6, ha="center", va=va,
            color=CATEGORY_COLORS[cat], fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="#ccc", lw=0.4),
        )

# X-axis category labels
ax.set_xticks(x_tick_positions)
ax.set_xticklabels(x_ticks, fontsize=9, fontweight="bold")

ax.set_ylabel("Correlation coefficient", fontsize=11)
ax.set_title("Association of clinical phenotypes with DL-ECG predicted risk (Model 5)",
             fontsize=13, fontweight="bold")
ax.grid(True, axis="y", alpha=0.15)
ax.tick_params(axis="x", length=0)
ax.tick_params(axis="y", labelsize=10)

# Category legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=CATEGORY_COLORS[cat],
           markersize=8, label=cat)
    for cat in CATEGORY_ORDER
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
          frameon=True, fancybox=False, edgecolor="#ccc", framealpha=0.95,
          title="Category", title_fontsize=9)

# Axis annotations
ax.text(0.01, 0.02, "Negative correlation\n(lower predicted risk)",
        transform=ax.transAxes, fontsize=7.5, color="#666", fontstyle="italic", va="bottom")
ax.text(0.01, 0.98, "Positive correlation\n(higher predicted risk)",
        transform=ax.transAxes, fontsize=7.5, color="#666", fontstyle="italic", va="top")

plt.tight_layout()
fig.savefig(OUT_DIR / "fig7_phewas_CVD4.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "fig7_phewas_CVD4.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved -> {OUT_DIR / 'fig7_phewas_CVD4.png'}")

# Print summary
print(f"\nTop positive correlations:")
for _, r in res.sort_values("corr", ascending=False).head(10).iterrows():
    p_str = "<0.001" if r["p"] < 0.001 else f"{r['p']:.3f}"
    print(f"  {r['display']:<25s} {r['category']:<18s} r={r['corr']:+.3f}  p={p_str}")

print(f"\nTop negative correlations:")
for _, r in res.sort_values("corr", ascending=True).head(5).iterrows():
    p_str = "<0.001" if r["p"] < 0.001 else f"{r['p']:.3f}"
    print(f"  {r['display']:<25s} {r['category']:<18s} r={r['corr']:+.3f}  p={p_str}")

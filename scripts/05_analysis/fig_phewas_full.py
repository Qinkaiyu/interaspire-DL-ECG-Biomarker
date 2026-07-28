"""
Fig. 7 Full PheWAS: Scan ALL ~700 variables in the dataset,
auto-classify by column prefix, compute correlation with Model 5 risk.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pointbiserialr

BASE = Path(".")
DATA_PATH = BASE / "data" / "INTERASPIRE_analysis_dataset.xlsx"
OOF_M04 = BASE / "results" / "oof_predictions" / "oof_CVD4.csv"
OOF_M5 = BASE / "model5_cvd_mace4" / "E2E_MultiTask_32_2_phase_cvd_mace4_1yr_oof.csv"
OUT_DIR = BASE / "results" / "figures"

evt_col = "event_CVD_Composite_4"

# ── Load & merge ─────────────────────────────────────────────────────────────
raw = pd.read_excel(DATA_PATH, engine="openpyxl")
df04 = pd.read_csv(OOF_M04)
df5 = pd.read_csv(OOF_M5)
df5 = df5.rename(columns={"oof_log_risk": "Model_5_risk", "test_fold": "fold5"})
merged = df04.merge(df5[["STUDYID", "Model_5_risk", "fold5"]], on="STUDYID", how="inner")

for fid in sorted(merged["fold5"].unique()):
    mask = merged["fold5"] == fid
    vals = merged.loc[mask, "Model_5_risk"]
    merged.loc[mask, "Model_5_risk"] = (vals - vals.mean()) / vals.std()

merged = merged.merge(raw, on="STUDYID", how="left", suffixes=("", "_raw"))

# ── Category mapping by prefix ───────────────────────────────────────────────
# INTERASPIRE variable naming convention:
# RQ* = Recruitment/Index event data
# RR* = Risk factors at recruitment (medications, conditions)
# IV* = Interview data
# RS* = Risk factor survey (smoking, diet, activity, BP, cholesterol, glucose)
# LC* = Lifestyle counselling
# MD* = Medications at follow-up
# FH* = Follow-up health (ECG, Echo, lab at follow-up visit)
# SB* = Subjective well-being (HADS, EuroQol, HSQ, MAQ)
# CVAR* = Computed/derived variables (binned risk factors)

CATEGORY_MAP = {
    # Demographics & Anthropometrics
    "AGE": "Demographics",
    "RQSEX": "Demographics",
    "RQETHNIC": "Demographics",
    "COUNTRY": "Demographics",
    "RQDCHEIGHT": "Demographics",
    "RQDCWEIGHT": "Demographics",
    "RQDCBMI": "Demographics",
    "RQDCWAIST": "Demographics",
    "RQDCSYSBP": "Demographics",
    "RQDCDIASBP": "Demographics",

    # Index event
    "RQINDEX": "Index event",
    "RQCABG": "Index event",
    "RQPTCA": "Index event",
    "idx_": "Index event",
}

PREFIX_CATEGORY = [
    # Order matters — first match wins
    ("Sinus Rhythm", "ECG findings"),
    ("Atrial F", "ECG findings"),
    ("Supraventricular", "ECG findings"),
    ("Ventricular T", "ECG findings"),
    ("Ventricular F", "ECG findings"),
    ("RBBB", "ECG findings"),
    ("LBBB", "ECG findings"),
    ("Q Wave", "ECG findings"),
    ("ST ", "ECG findings"),
    ("T Wave", "ECG findings"),
    ("Ischaemic", "ECG findings"),
    ("QT ", "ECG findings"),
    ("LVH", "ECG findings"),
    ("RVH", "ECG findings"),
    ("1 AV", "ECG findings"),
    ("2 AV", "ECG findings"),
    ("3 AV", "ECG findings"),
    ("Left Axis", "ECG findings"),
    ("Right Axis", "ECG findings"),
    ("MI (", "ECG findings"),
    ("MI(", "ECG findings"),
    ("PR Interval", "ECG findings"),
    ("QRS Duration", "ECG findings"),
    ("QT Interval", "ECG findings"),

    ("FH", "Follow-up exam (Echo/ECG/Lab)"),

    ("SB", "Mental health & QoL"),

    ("CVAR", "Derived risk factors"),

    ("RQHIST", "Medical history"),
    ("RQPREMAT", "Medical history"),
    ("RQPRE", "Medical history"),
    ("RQRISK", "Medical history"),
    ("RQSMOK", "Medical history"),

    ("RQCOL", "Lab values (index)"),
    ("RQHDL", "Lab values (index)"),
    ("RQTRIG", "Lab values (index)"),
    ("RQLDL", "Lab values (index)"),
    ("RQSERUM", "Lab values (index)"),
    ("RQHBA1C", "Lab values (index)"),
    ("RQBLG", "Lab values (index)"),
    ("RQOGTT", "Lab values (index)"),
    ("RQTROP", "Lab values (index)"),
    ("RQNTPRO", "Lab values (index)"),
    ("RQBNP", "Lab values (index)"),
    ("RQCRP", "Lab values (index)"),
    ("RQDDIM", "Lab values (index)"),
    ("RQINTLEU", "Lab values (index)"),
    ("RQLEU", "Lab values (index)"),

    ("RQASP", "Medications (index)"),
    ("RQANTI", "Medications (index)"),
    ("RQGLUC", "Medications (index)"),
    ("RQLIPID", "Medications (index)"),
    ("RQFIB", "Medications (index)"),
    ("RQCHOL", "Medications (index)"),
    ("RQDAILY", "Medications (index)"),

    ("RR", "Risk factors & Meds (recruitment)"),

    ("IV", "Interview data"),

    ("RS", "Risk factor survey"),

    ("LC", "Lifestyle counselling"),

    ("MD", "Medications (follow-up)"),

    ("RQ", "Other recruitment data"),

    ("any_af", "Derived variables"),
    ("ecg_", "Derived variables"),
    ("echo_", "Derived variables"),
    ("af_from", "Derived variables"),
    ("clinical cluster", "Derived variables"),
    ("newav", "Derived variables"),
    ("event_", "Outcomes"),
    ("time_", "Outcomes"),
    ("first_comp", "Outcomes"),
    ("SURVIVAL", "Outcomes"),
    ("VITAL", "Outcomes"),
    ("HOSP", "Outcomes"),
    ("cvd_death", "Outcomes"),
]

EXCLUDE_PATTERNS = [
    "STUDYID", "PATID", "CENTREID", "Date", "date", "DATE", "DT",
    "path", "PATH", "csv", "source", "match", "shape", "transpose",
    "prediction", "Unnamed", "excel2", "fold", "COMPDT", "Region",
    "Comments", "sample_id", "n_segments", "total_timepoints",
    "MODE", "MODEOTHER",
    # Exclude outcomes
    "event_", "time_", "first_comp", "SURVIVAL", "VITAL", "HOSP",
    "cvd_death", "days_to",
    # Exclude model predictions (not phenotypes)
    "Model_0_", "Model_1_", "Model_2_", "Model_3_", "Model_4_",
    "Model_5_", "m5_cal",
    # Exclude derived AF variables (redundant / computed)
    "clinical cluster", "af_from_", "ecg_confirmed_af", "echo_confirmed_af",
    "any_af_clean", "any_af_no_biomarker", "ecg_af", "echo_af", "any_af",
    "rqecg_af",
    # Exclude raw identifiers
    "AGE", "COUNTRY", "RQSEX", "RQETHNIC", "RQINDEX",
    "idx_STEMI", "idx_NSTEMI", "idx_UA", "idx_ePCI", "idx_eCABG",
]


def get_category(col):
    # Exact match first
    if col in CATEGORY_MAP:
        return CATEGORY_MAP[col]
    # Prefix match
    for prefix, cat in PREFIX_CATEGORY:
        if col.startswith(prefix):
            return cat
    return "Other"


def should_exclude(col):
    for pat in EXCLUDE_PATTERNS:
        if pat in col:
            return True
    return False


# ── Compute correlations for ALL variables ───────────────────────────────────
risk = merged["Model_5_risk"].values
results = []
category_counts = {}

for col in merged.columns:
    if should_exclude(col) or col == "Model_5_risk":
        continue

    category = get_category(col)
    if category == "Outcomes":
        continue

    s = merged[col]

    # Try to make numeric
    if s.dtype == object:
        unique_vals = set(s.dropna().unique())
        if unique_vals <= {"Yes", "No"} or unique_vals <= {"Yes", "No", "Not Recorded", "Unsure"}:
            vals = s.map({"Yes": 1, "No": 0}).astype(float)
        elif unique_vals <= {"Male", "Female"}:
            vals = s.map({"Male": 1, "Female": 0}).astype(float)
        elif len(unique_vals) == 2:
            mapping = {v: i for i, v in enumerate(sorted(unique_vals))}
            vals = s.map(mapping).astype(float)
        else:
            continue  # skip multi-level categorical for now
    else:
        vals = pd.to_numeric(s, errors="coerce")

    valid = vals.notna() & (~np.isnan(risk))
    if valid.sum() < 100:
        continue

    x = vals[valid].values.astype(float)
    y = risk[valid]

    n_unique = len(np.unique(x[~np.isnan(x)]))
    if n_unique < 2:
        continue

    is_binary = n_unique == 2
    min_group = min((x == 0).sum(), (x == 1).sum()) if is_binary else 0
    if is_binary and min_group < 10:
        continue

    try:
        if is_binary:
            corr, p = pointbiserialr(x, y)
        else:
            corr, p = spearmanr(x, y)
    except Exception:
        continue

    if np.isnan(corr):
        continue

    results.append({
        "col": col, "category": category,
        "corr": corr, "p": p, "n": int(valid.sum()),
        "is_binary": is_binary,
    })

    category_counts[category] = category_counts.get(category, 0) + 1

res = pd.DataFrame(results)

print(f"Total variables tested: {len(res)}")
print(f"\nVariables per category:")
for cat, cnt in sorted(category_counts.items(), key=lambda x: -x[1]):
    n_sig = len(res[(res["category"] == cat) & (res["p"] < 0.05)])
    print(f"  {cat:<40s}: {cnt:4d} variables ({n_sig} significant)")

# Save category mapping for future use
cat_map_rows = []
for _, r in res.iterrows():
    cat_map_rows.append({"column": r["col"], "category": r["category"],
                         "corr": r["corr"], "p": r["p"]})
cat_map_df = pd.DataFrame(cat_map_rows)
cat_map_df.to_csv(OUT_DIR / "phewas_variable_categories.csv", index=False)
print(f"\nSaved variable-category mapping -> {OUT_DIR / 'phewas_variable_categories.csv'}")

# ── Significance threshold (Bonferroni) ──────────────────────────────────────
n_tests = len(res)
bonferroni = 0.05 / n_tests
print(f"\nBonferroni threshold: {bonferroni:.2e}")
print(f"Significant (p<0.05): {(res['p'] < 0.05).sum()}")
print(f"Significant (Bonferroni): {(res['p'] < bonferroni).sum()}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE: PheWAS scatter
# ═══════════════════════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt

CATEGORY_COLORS = {
    "Demographics": "#636EFA",
    "Lab values (index)": "#EF553B",
    "Medical history": "#00CC96",
    "Medications (index)": "#AB63FA",
    "ECG findings": "#FFA15A",
    "Index event": "#19D3F3",
    "Follow-up exam (Echo/ECG/Lab)": "#FF6692",
    "Mental health & QoL": "#B6E880",
    "Derived risk factors": "#FF97FF",
    "Risk factors & Meds (recruitment)": "#FECB52",
    "Risk factor survey": "#72B7B2",
    "Lifestyle counselling": "#636EFA",
    "Medications (follow-up)": "#EF553B",
    "Interview data": "#00CC96",
    "Derived variables": "#AAAAAA",
    "Other recruitment data": "#CCCCCC",
    "Other": "#999999",
}

# Order categories by median |correlation|
cat_order = res.groupby("category")["corr"].apply(lambda x: np.median(np.abs(x))).sort_values(ascending=False).index.tolist()

fig, ax = plt.subplots(figsize=(16, 6))

x_pos = 0
x_positions = []
x_ticks = []
x_tick_positions = []

for cat in cat_order:
    cat_data = res[res["category"] == cat].sort_values("corr", ascending=False)
    if cat_data.empty or len(cat_data) < 2:
        continue

    cat_start = x_pos
    for _, row in cat_data.iterrows():
        color = CATEGORY_COLORS.get(cat, "#999")
        sig = min(-np.log10(max(row["p"], 1e-50)), 20)
        size = max(sig * 5, 8)
        alpha = 0.85 if row["p"] < bonferroni else (0.5 if row["p"] < 0.05 else 0.15)

        ax.scatter(x_pos, row["corr"], s=size, c=color,
                   alpha=alpha, edgecolors="white", linewidths=0.2, zorder=5)
        x_positions.append((x_pos, row))
        x_pos += 1

    cat_mid = (cat_start + x_pos - 1) / 2
    x_ticks.append(cat.replace(" (", "\n(").replace("&", "&\n") if len(cat) > 20 else cat)
    x_tick_positions.append(cat_mid)

    if cat != cat_order[-1]:
        ax.axvline(x_pos - 0.5, color="#eee", linewidth=0.5, zorder=1)
    x_pos += 2

ax.axhline(0, color="grey", linewidth=0.8, alpha=0.4)

# Label top phenotypes
try:
    from adjustText import adjust_text
    LABEL_THRESHOLD = 0.18
    texts = []
    for xp, row in x_positions:
        if abs(row["corr"]) > LABEL_THRESHOLD:
            color = CATEGORY_COLORS.get(row["category"], "#999")
            t = ax.text(xp, row["corr"], row["col"], fontsize=5.5, color=color, fontweight="bold")
            texts.append(t)
    if texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.4),
                    force_text=(0.3, 0.5), expand_text=(1.2, 1.4))
except ImportError:
    pass

ax.set_xticks(x_tick_positions)
ax.set_xticklabels(x_ticks, fontsize=7, fontweight="bold")
ax.set_ylabel("Correlation coefficient", fontsize=11)
ax.set_title(f"PheWAS: Association of {len(res)} clinical phenotypes with DL-ECG predicted risk",
             fontsize=13, fontweight="bold")
ax.grid(True, axis="y", alpha=0.15)
ax.tick_params(axis="x", length=0)
ax.tick_params(axis="y", labelsize=10)

# Legend (only categories with >=5 variables)
from matplotlib.lines import Line2D
legend_cats = [c for c in cat_order if category_counts.get(c, 0) >= 5]
legend_elements = [
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor=CATEGORY_COLORS.get(cat, "#999"),
           markersize=7, label=f"{cat} ({category_counts[cat]})")
    for cat in legend_cats
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=6,
          frameon=True, fancybox=False, edgecolor="#ccc", framealpha=0.95,
          title="Category (n variables)", title_fontsize=7, ncol=2)

ax.text(0.01, 0.02, "Negative correlation\n(lower predicted risk)",
        transform=ax.transAxes, fontsize=7, color="#666", fontstyle="italic", va="bottom")
ax.text(0.01, 0.98, "Positive correlation\n(higher predicted risk)",
        transform=ax.transAxes, fontsize=7, color="#666", fontstyle="italic", va="top")

# Bonferroni annotation
ax.text(0.99, 0.02, f"Bonferroni p < {bonferroni:.1e}\n"
        f"Opaque dots: Bonferroni significant\nSemi-transparent: p<0.05\nFaded: not significant",
        transform=ax.transAxes, fontsize=6, ha="right", va="bottom", color="#888",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ddd", alpha=0.9))

plt.tight_layout()
fig.savefig(OUT_DIR / "fig7_phewas_full_CVD4.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "fig7_phewas_full_CVD4.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSaved -> {OUT_DIR / 'fig7_phewas_full_CVD4.png'}")

# Top correlations
print(f"\nTop 15 positive correlations:")
for _, r in res.sort_values("corr", ascending=False).head(15).iterrows():
    p_str = f"<{bonferroni:.0e}" if r["p"] < bonferroni else f"{r['p']:.2e}"
    print(f"  {r['col']:<35s} {r['category']:<30s} r={r['corr']:+.3f}  p={p_str}")

print(f"\nTop 10 negative correlations:")
for _, r in res.sort_values("corr").head(10).iterrows():
    p_str = f"<{bonferroni:.0e}" if r["p"] < bonferroni else f"{r['p']:.2e}"
    print(f"  {r['col']:<35s} {r['category']:<30s} r={r['corr']:+.3f}  p={p_str}")

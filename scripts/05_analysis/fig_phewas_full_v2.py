"""
Fig. 7 Full PheWAS (v2): Manhattan-style with colored background bands per category.
- Background shading per category
- No x-axis variable labels (too many)
- Top 10 most significant variables labeled with readable names
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

BASE = Path(".")
OUT_DIR = BASE / "results" / "figures"

# ── Load pre-computed PheWAS results ────────────────────────────────────────
res = pd.read_csv(OUT_DIR / "phewas_variable_categories.csv")

# ── Merge categories ──────────────────────────────────────────────────────
CAT_MERGE = {
    "ECG findings": "ECG & Cardiac assessment",
    "Follow-up exam (Echo/ECG/Lab)": "ECG & Cardiac assessment",
    "Derived risk factors": "Risk factors",
    "Risk factor survey": "Risk factors",
    "Risk factors & Meds (recruitment)": "Risk factors",
    "Mental health & QoL": "Mental health, QoL & Interview",
    "Interview data": "Mental health, QoL & Interview",
    "Lab values (index)": "Lab values",
    "Medications (index)": "Baseline medications",
}
CAT_DROP = {"Demographics", "Other recruitment data", "Other", "Derived variables"}
res["category"] = res["category"].map(lambda c: CAT_MERGE.get(c, c))
res = res[~res["category"].isin(CAT_DROP)].reset_index(drop=True)

# ── Per-variable category overrides & deduplication ───────────────────────
# 1) Move CVARLAB* from Risk factors → Lab values (they are core-lab data)
MOVE_TO_LAB = {c for c in res["column"] if c.startswith("CVARLAB") or c in
               {"CVARAPOB", "CVARAPOA1", "CVARLPA", "CVARLOCALFASTINGLU"}}
res.loc[res["column"].isin(MOVE_TO_LAB), "category"] = "Lab values"

# 2) Move medication doses from Risk factors → Baseline medications / Medications (follow-up)
RECRUIT_MEDS = {c for c in res["column"] if c.startswith("RR") and "DOSE" in c.upper()}
RECRUIT_MEDS |= {"RRSACUB"}  # sacubitril flag
res.loc[res["column"].isin(RECRUIT_MEDS), "category"] = "Baseline medications"

# 3) Remove RQ lab duplicates (less accurate questionnaire versions of core-lab data)
RQ_LAB_DUPS = {
    "RQSERUMCREA", "RQSERUMCREAU",          # dup of CVARLABCREA
    "RQLDLCOL", "RQLDLCOL1", "RQLDLCOLU", "RQLDLCOL1U",  # dup of CVARLABLDL*
    "RQBLGLUC", "RQBLGLLE", "RQBLGLLEU", "RQBLGLLEY",     # dup of CVARLABGLUC
    "RQCOL", "RQCOLU",                       # dup of CVARLABTC
    "RQHDLCOL", "RQHDLCOLU",                 # dup of CVARLABHDL
    "RQTRIG", "RQTRIGU",                     # dup of CVARLABTG
    "RQHBA1C",                               # dup of CVARLABHBA1CPERCENT
}
# 4) Remove RS survey duplicates of lab values
RS_LAB_DUPS = {
    "RSHBACT",     # dup of HbA1c
    "RSTCACT", "RSTCTLEV", "RSTCTLEVU",     # dup of total cholesterol
    "RSBGACT", "RSBGTLEV", "RSBGTLEVU",     # dup of blood glucose
    "RSCOL", "RSCOLU",                       # dup of cholesterol
}
# 5) Remove recruitment lab duplicates
RR_LAB_DUPS = {"RRSERUMCREA", "RRSERUMCREAU"}

EXTRA_DUPS = {"CVARLABLDLSAMPSON"}  # near-identical to Friedewald
ALL_DUPS = RQ_LAB_DUPS | RS_LAB_DUPS | RR_LAB_DUPS | EXTRA_DUPS
res = res[~res["column"].isin(ALL_DUPS)].reset_index(drop=True)
print(f"After dedup: {len(res)} variables")

print(f"Total variables: {len(res)}")

n_tests = len(res)
bonferroni = 0.05 / n_tests

# ── Human-readable variable name mapping (top associations) ─────────────────
READABLE_NAMES = {
    # Medical history
    "RQHISTOFHEARTFAIL": "History of heart failure",
    "RQHISTOFDIAB": "History of diabetes",
    "RQHISTOFKIDNEYDISEASE": "History of kidney disease",
    "RQHISTOFCORARTDIS": "History of coronary artery disease",
    "RQHISTOFHYPER": "History of hypertension",
    "RQHISTOFCOPD": "History of COPD",
    "RQPREMATCEREB": "Premature cerebrovascular disease",
    "RQPREMATCAD": "Premature CAD family history",

    # ECG findings
    "QT Prolongation": "QT prolongation",
    "QT Interval (ms)": "QT interval",
    "QRS Duration (ms)": "QRS duration",
    "ST Depression": "ST depression",
    "ST Elevation": "ST elevation",
    "T Wave Inversion": "T-wave inversion",
    "Atrial Fibrillation": "Atrial fibrillation",
    "LBBB": "Left bundle branch block",
    "RBBB": "Right bundle branch block",
    "LVH": "Left ventricular hypertrophy",
    "1 AV block": "First-degree AV block",
    "Left Axis Deviation": "Left axis deviation",
    "Ischaemic": "Ischaemic changes",
    "Sinus Rhythm": "Sinus rhythm",
    "PR Interval (ms)": "PR interval",
    "Q Wave": "Q wave",
    "MI (Old)": "Old myocardial infarction",
    "MI (Acute)": "Acute myocardial infarction",

    # Demographics
    "RQDCHEIGHT": "Height",
    "RQDCBMI": "BMI",
    "RQDCSYSBP": "Systolic BP",
    "RQDCDIASBP": "Diastolic BP",
    "RQDCWEIGHT": "Weight",
    "RQDCWAIST": "Waist circumference",
    "CVARAGE": "Age",

    # Lab values
    "RQCOL": "Total cholesterol",
    "RQHDL": "HDL cholesterol",
    "RQLDL": "LDL cholesterol",
    "RQTRIG": "Triglycerides",
    "RQHBA1C": "HbA1c",
    "RQSERUM": "Serum creatinine",
    "RQBLG": "Blood glucose",
    "RQTROP": "Troponin",
    "RQNTPRO": "NT-proBNP",
    "RQBNP": "BNP",
    "RQCRP": "C-reactive protein",

    # Medications
    "RQASPIRIN": "Aspirin",
    "RQANTICOAGULANTS": "Anticoagulants",
    "RQLIPIDLOWERING": "Lipid-lowering therapy",
    "RQGLUCOSELOWERING": "Glucose-lowering therapy",

    # Derived
    "CVAREGFR": "eGFR",
    "CVARBMI": "BMI",
    "CVARSYSTOL": "Systolic BP",
    "CVARLABHBA1CPERCENT": "HbA1c %",
    "CVARLABHDL": "HDL",
    "CVARLABLDLFRIEDEWALD": "LDL cholesterol",
    "CVARMEDBPLOWERING": "BP-lowering medication",
    "CVARMEDLIPIDLOWERING": "Lipid-lowering medication",
    "CVARMEDGLUCOSELOWERING": "Glucose-lowering medication",
    "CVARCOSMOKING": "Current smoking",
    "CVARSMART2RISK": "SMART2 risk score",
    "CVARLABCREA": "Serum creatinine",
    "CVARDIABETESSRMED": "Diabetes",

    # Index event
    "RQPTCAINFARC": "PCI for infarction",
    "RQCABGINFARC": "CABG for infarction",

    # Follow-up
    "FHLVSYFUNC": "LV systolic function",
    "FHWASECGPER": "ECG performed",

    # Risk factor survey
    "RSGLUCOSELOWERING": "Glucose-lowering therapy",
    "RSBPLOWERING": "BP-lowering therapy",
    "RSLIPIDLOWERING": "Lipid-lowering therapy",
    "RSHBACT": "HbA1c",

    # Follow-up exam
    "FHQTC": "QTc interval",
    "FHHEARRAT": "Heart rate",
    "FHLVEF": "LV ejection fraction",
    "FHVENRAT": "Ventricular rate",
    "FHOGTT": "OGTT status",
    "FHFSCALC": "Fractional shortening",
    "FHLAENLAR": "LA enlargement",

    # Risk factors & Meds (recruitment)
    "RRKIDNEYDIS": "Kidney disease",
    "RRABNORMGLUCMETA": "Abnormal glucose metabolism",

    # Medications (index)
    "RQANTIHYPDRUGS": "Antihypertensive drugs",
    "RQASPIOROTHER": "Aspirin or other antiplatelet",
    "RQDAILYDOSE": "Daily aspirin dose",
    "RQFIBRATES": "Fibrates",
    "RQCHOLABSIN": "Cholesterol absorption inhibitor",

    # Mental health & QoL
    "SBMOB": "Mobility problems",
    "SBEUROQOL1": "EuroQoL VAS score",
    "SBEUROQOL2": "EuroQoL health index",
    "SBPHYSIC": "Physical health summary",
    "SBGLOBAL": "Global QoL score",
    "SBPHY": "Physical functioning",
    "SBACT": "Activity limitations",
    "SBSELF": "Self-care problems",
    "SBPAIN": "Pain/discomfort",
    "SBANX": "Anxiety/depression (EQ-5D)",
    "SBHADSDEP": "HADS depression score",
    "SBHADSANX": "HADS anxiety score",
    "SBLTD": "Daily activity limitation",
    "SBLIFT": "Lifting/carrying difficulty",
    "SBWALK100": "Walking 100m difficulty",
    "SBFEEL": "Feeling downhearted",
    "SBCLIMB": "Climbing stairs difficulty",
    "SBSHORT": "Shortness of breath",
    "SBYARD": "Yard work difficulty",
    "SBGARDEN": "Gardening difficulty",
    "SBWALK": "Walking difficulty",
    "SBDEPRESS": "Feeling depressed",
    "SBEMOTION": "Emotional problems",
    "SBRELAX": "Difficulty relaxing",
    "SBWORRY": "Worrying",
    "SBFRUST": "Frustration",
    "SBDRINKALC": "Alcohol consumption",

    # Lifestyle counselling
    "LCDIABENDO": "Referred to diabetes/endocrinology",
    "LCATTCLUB": "Attended exercise club",
    "LCHELPOTH": "Received help from others",
    "LCFOLLADV": "Following dietary advice",
    "LCREGACT": "Regular physical activity",
    "LCEXPROF": "Exercise with professional",
    "LCREDSUG": "Reducing sugar intake",

    # Follow-up medications
    "MDASP1DOSE": "Aspirin dose",
    "MDNITDOSE": "Nitrate dose",
    "MDINSDOSE": "Insulin dose",
    "MDDIURDOSE1": "Diuretic dose",
    "MDBBDOSE": "Beta-blocker dose",
    "MDACEDOSE": "ACE inhibitor dose",
    "MDSTATDOSE": "Statin dose",

    # Interview
    "IVCDIAB": "Diabetes care visits",
    "IVCOTH": "Other specialist visits",
    "IVCPHYS": "Physiotherapy visits",
    "IVYRSSTUDY": "Years in study",

    # Core lab values (CVAR)
    "CVARAPOB": "Apolipoprotein B",
    "CVARAPOA1": "Apolipoprotein A1",
    "CVARLPA": "Lipoprotein(a)",
    "CVARLOCALFASTINGLU": "Fasting glucose",
    "CVARLABTG": "Triglycerides",
    "CVARLABTC": "Total cholesterol",
    "CVARLABGLUC": "Glucose",

    # Troponin / NT-proBNP (unique in Lab values)
    "RQTROPHSTYN": "High-sensitivity troponin",
    "RQTROPIYN": "Troponin I",
    "RQTROPHSIYN": "HS troponin I",
    "RQTROPTYN": "Troponin T",
    "RQNTPROBNPYN": "NT-proBNP",
    "RQOGTTY": "OGTT performed",

    # Recruitment risk factors
    "RRKIDNEYDIS": "Kidney disease",
    "RRABNORMGLUCMETA": "Abnormal glucose metabolism",
    "RRHYPERTEN": "Hypertension",
    "RROBESE": "Obesity",
    "RRHYPERLIP": "Hyperlipidaemia",
    "RRRETINO": "Retinopathy",
    "RRIGT": "Impaired glucose tolerance",
    "RRANXIE": "Anxiety disorder",
    "RRNIC": "Nicotine use",
    "RRTYPDUR": "Diabetes duration",
    "RRANTIDEP": "Antidepressant use",
    "RRMEDDIS": "Medication discontinuation",
    "RRPOTCHACT": "Potassium channel activator",
    "RRNIDOSE": "Nitrate dose",

    # Recruitment medication doses moved to Baseline medications
    "RRASPDOSE1": "Aspirin dose",
    "RRASPDOSE2": "Aspirin dose (alt)",
    "RRBBDOSE": "Beta-blocker dose",
    "RRACEDOSE": "ACE inhibitor dose",
    "RRDIURDOSE1": "Diuretic dose",
    "RRDIURDOSE2": "Diuretic dose (alt)",
    "RRSTATDOSE": "Statin dose",
    "RRMETFORDOSE": "Metformin dose",
    "RRGLUCDOSE1": "Glucose-lowering dose",
    "RRGLUCDOSE2": "Glucose-lowering dose (alt)",
    "RRCACHBLDOSE": "CCB dose",
    "RRSACUBDOSE1": "Sacubitril dose",
    "RRSACUBDOSE2": "Sacubitril dose (alt)",
    "RRA2DOSE": "ARB dose",
    "RRACDOSE": "Anticoagulant dose",
    "RRMETDOSE": "Metoprolol dose",
    "RRIFDOSE": "Ivabradine dose",
    "RRGLUCINCRETINSDOSE": "Incretin dose",
    "RRGLUCSGLT2DOSE": "SGLT2 inhibitor dose",
    "RRGLUCSULPHONDOSE": "Sulphonylurea dose",

    # Derived extras
    "newavdrinkweek": "Avg drinks per week",
    "CVARHEARTRATE": "Heart rate",
    "CVARPHYSACT": "Physical activity",
    "CVARGUIDELINESTARGETSCORE": "Guideline target score",
    "CVAROGTT120VALUE": "OGTT 120min value",
    "CVAROGTT0VALUE": "OGTT 0min value",
}


def get_readable(col):
    if col in READABLE_NAMES:
        return READABLE_NAMES[col]
    # Try partial matches
    for key, val in READABLE_NAMES.items():
        if col.startswith(key):
            return val
    # Fallback: clean up the raw name
    name = col.replace("_", " ").replace("CVAR", "").replace("RQ", "")
    return name[:35]


# ── Category styling ────────────────────────────────────────────────────────
CATEGORY_COLORS = {
    "ECG & Cardiac assessment":          ("#FFA15A", "#FFF5EC"),
    "Risk factors":                      ("#FF97FF", "#FFF0FF"),
    "Medical history":                   ("#00CC96", "#E6FFF7"),
    "Mental health, QoL & Interview":   ("#B6E880", "#F5FFE8"),
    "Lab values":                        ("#EF553B", "#FFF0ED"),
    "Baseline medications":              ("#AB63FA", "#F5EDFF"),
    "Lifestyle counselling":            ("#636EFA", "#EDEDFF"),
    "Medications (follow-up)":          ("#D98880", "#FFF0ED"),
    "Index event":                      ("#19D3F3", "#EDFBFF"),
}

# Order categories by median |correlation| descending
cat_order = (res.groupby("category")["corr"]
             .apply(lambda x: np.median(np.abs(x)))
             .sort_values(ascending=False).index.tolist())

# Filter out tiny categories
cat_order = [c for c in cat_order if len(res[res["category"] == c]) >= 2]

# ── Build x-positions ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 7))
fig.subplots_adjust(bottom=0.18)  # room for legend + annotations below

x_pos = 0
cat_ranges = []  # (start, end, category_name)
all_points = []  # (x, corr, p, col, category)

for cat in cat_order:
    cat_data = res[res["category"] == cat].sample(frac=1, random_state=42)
    if cat_data.empty:
        continue

    n = len(cat_data)
    # Adaptive spacing: small categories more spread, large ones tighter
    spacing = max(0.4, 2.5 / np.sqrt(n))

    cat_start = x_pos
    for i, (_, row) in enumerate(cat_data.iterrows()):
        all_points.append((x_pos, row["corr"], row["p"], row["column"], cat))
        x_pos += spacing

    cat_ranges.append((cat_start, x_pos - spacing, cat))
    x_pos += 3  # gap between categories

# ── Draw background bands ──────────────────────────────────────────────────
y_min = min(p[1] for p in all_points) - 0.05
y_max = max(p[1] for p in all_points) + 0.05

for start, end, cat in cat_ranges:
    _, bg_color = CATEGORY_COLORS.get(cat, ("#999", "#F0F0F0"))
    rect = Rectangle((start - 0.5, y_min), end - start + 1, y_max - y_min,
                      facecolor=bg_color, edgecolor="none", zorder=0)
    ax.add_patch(rect)

    # (no vertical category labels — legend is sufficient)

# ── Draw scatter points ────────────────────────────────────────────────────
import matplotlib.colors as mcolors

def darken(hex_color, factor=0.7):
    """Return a darker version of a hex color."""
    r, g, b = mcolors.hex2color(hex_color)
    return mcolors.rgb2hex((r * factor, g * factor, b * factor))

for xp, corr, p, col, cat in all_points:
    dot_color, _ = CATEGORY_COLORS.get(cat, ("#999", "#F0F0F0"))
    edge = darken(dot_color, 0.65)
    sig = min(-np.log10(max(p, 1e-50)), 20)
    size = max(sig * 5, 10)

    if p < bonferroni:
        alpha = 0.9
        lw = 0.6
    elif p < 0.05:
        alpha = 0.6
        lw = 0.4
    else:
        alpha = 0.15
        lw = 0.3

    # Draw edge separately so it's always fully opaque
    ax.scatter(xp, corr, s=size, c=edge, alpha=1.0,
               edgecolors="none", linewidths=0, zorder=4)  # opaque edge ring
    ax.scatter(xp, corr, s=max(size * 0.7, 6), c=dot_color, alpha=alpha,
               edgecolors="none", linewidths=0, zorder=5)  # fill with alpha

# ── Label top 2-3 per category by |correlation| ──────────────────────────
LABEL_EXCLUDE = {"RQHISTOFKIDNEYDISEASE", "RRASPDOSE2"}  # skip duplicates
sig_points = [(xp, corr, p, col, cat) for xp, corr, p, col, cat in all_points
              if p < 0.05 and col not in LABEL_EXCLUDE]

# Group by category, pick top 2-3 per category
from collections import defaultdict
cat_top = defaultdict(list)
for pt in sig_points:
    cat_top[pt[4]].append(pt)

label_points = []
for cat, pts in cat_top.items():
    if cat == "Baseline medications":
        # 1 positive + 1 negative
        pos = sorted([p for p in pts if p[1] > 0], key=lambda x: x[1], reverse=True)
        neg = sorted([p for p in pts if p[1] < 0], key=lambda x: x[1])
        label_points.extend(pos[:1])
        label_points.extend(neg[:1])
    elif cat == "ECG & Cardiac assessment":
        # top 3 positive + top 3 negative
        pos = sorted([p for p in pts if p[1] > 0], key=lambda x: x[1], reverse=True)
        neg = sorted([p for p in pts if p[1] < 0], key=lambda x: x[1])
        label_points.extend(pos[:3])
        label_points.extend(neg[:3])
    else:
        ranked = sorted(pts, key=lambda x: abs(x[1]), reverse=True)
        n_pick = 3 if len(pts) >= 10 else 2
        label_points.extend(ranked[:n_pick])

# Uniform angled leader lines: diagonal then horizontal, label at end
# Separate positive and negative correlation labels, stagger within each group
pos_labels = sorted([(xp, corr, p, col, cat) for xp, corr, p, col, cat in label_points if corr >= 0],
                    key=lambda x: x[0])
neg_labels = sorted([(xp, corr, p, col, cat) for xp, corr, p, col, cat in label_points if corr < 0],
                    key=lambda x: x[0])

ANGLE_B = 65   # consistent diagonal angle

for group, sign in [(pos_labels, 1), (neg_labels, -1)]:
    # Stagger: alternate between short and long offsets to avoid overlap
    offsets = [14, 14, 14]
    for i, (xp, corr, p, col, cat) in enumerate(group):
        label = get_readable(col)
        dy = offsets[i % len(offsets)] * sign

        # Baseline medications aspirin: leader line goes left
        if col == "RRASPDOSE1":
            ax.annotate(
                label, (xp, corr),
                textcoords="offset points",
                xytext=(-10, dy),
                fontsize=7.5, color="#222", va="center", ha="right", zorder=10,
                arrowprops=dict(
                    arrowstyle="-",
                    connectionstyle=f"angle,angleA=0,angleB={-ANGLE_B * sign},rad=0",
                    color="#555", lw=0.5,
                ),
            )
        else:
            ax.annotate(
                label, (xp, corr),
                textcoords="offset points",
                xytext=(10, dy),
                fontsize=7.5, color="#222", va="center", ha="left", zorder=10,
                arrowprops=dict(
                    arrowstyle="-",
                    connectionstyle=f"angle,angleA=0,angleB={ANGLE_B * sign},rad=0",
                    color="#555", lw=0.5,
                ),
            )

# ── Axes formatting ────────────────────────────────────────────────────────
ax.axhline(0, color="grey", linewidth=0.8, alpha=0.5, zorder=2)
ax.set_xlim(-2, x_pos)
ax.set_ylim(y_min, y_max)
ax.set_xticks([])  # No x-axis labels
ax.set_ylabel("Correlation with DL-ECG risk score", fontsize=12)
ax.set_title(f"PheWAS: Association of {len(res)} clinical phenotypes with DL-ECG predicted risk",
             fontsize=13, fontweight="bold", pad=8)
ax.tick_params(axis="y", labelsize=10)
ax.grid(True, axis="y", alpha=0.15, zorder=1)

# Remove black border frame (spines)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(left=True, bottom=False)  # keep y ticks, remove x ticks

# ── Legend (below plot, with Bonferroni info underneath) ───────────────────
legend_cats = [c for c in cat_order if len(res[res["category"] == c]) >= 3]
legend_elements = []
for cat in legend_cats:
    dot_color, _ = CATEGORY_COLORS.get(cat, ("#999", "#F0F0F0"))
    n_cat = len(res[res["category"] == cat])
    legend_elements.append(
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=dot_color, markersize=8,
               label=f"{cat} (n={n_cat})")
    )

leg = ax.legend(handles=legend_elements, loc="upper center",
                bbox_to_anchor=(0.5, -0.04), fontsize=9.5,
                frameon=False, ncol=4,
                title="Category", title_fontsize=10.5)

# Bonferroni info below legend
n_bonf = sum(1 for _, _, p, _, _ in all_points if p < bonferroni)
n_sig = sum(1 for _, _, p, _, _ in all_points if p < 0.05)
fig.text(0.5, 0.02,
         f"Bonferroni threshold: p < {bonferroni:.1e}  |  "
         f"Bonferroni significant: {n_bonf}  |  Nominal p<0.05: {n_sig}",
         fontsize=9.5, ha="center", va="bottom", color="#555")

fig.savefig(OUT_DIR / "fig7_phewas_full_CVD4.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "fig7_phewas_full_CVD4.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"\nSaved -> {OUT_DIR / 'fig7_phewas_full_CVD4.png'}")

# Print labeled variables for verification
print(f"\nLabeled variables ({len(label_points)}):")
for xp, corr, p, col, cat in sorted(label_points, key=lambda x: abs(x[1]), reverse=True):
    print(f"  {get_readable(col):<35s} ({cat:<30s}) r={corr:+.3f}  p={p:.2e}")

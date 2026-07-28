"""
Table 1: Baseline demographics and clinical characteristics
Stratified by primary endpoint (event_CVD_Composite_4)

Includes ALL variables from Model 1-3 + ALL 18 ECG biomarkers (Model 4)
+ supplementary variables (diabetes, total cholesterol, eCABG)
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import shutil

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH = Path("data/INTERASPIRE_analysis_dataset.xlsx")
OOF_PATH = Path("results/oof_predictions/oof_CVD4.csv")
EVENT_COL = "event_CVD_Composite_4"
OUT_DIR = Path("results/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_OUT_DIR = Path("results/manuscript_final/main_tables")
FINAL_OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ─────────────────────────────────────────────────────────────────
def fmt_continuous(series, decimals=1):
    s = series.dropna()
    if len(s) == 0:
        return "—"
    return f"{s.mean():.{decimals}f} ± {s.std():.{decimals}f}"


def fmt_n_pct(series, value):
    total = len(series)
    if isinstance(value, list):
        n = series.isin(value).sum()
    else:
        n = (series == value).sum()
    pct = n / total * 100 if total > 0 else 0
    return f"{n} ({pct:.1f}%)"


def fmt_missing(series):
    n_miss = series.isna().sum()
    if n_miss == 0:
        return ""
    return f"[{n_miss} missing]"


def p_continuous(s_event, s_noevent):
    s1 = s_event.dropna()
    s2 = s_noevent.dropna()
    if len(s1) < 3 or len(s2) < 3:
        return ""
    _, p = stats.mannwhitneyu(s1, s2, alternative="two-sided")
    return fmt_p(p)


def p_categorical(col, df_event, df_noevent):
    combined = pd.concat([df_event[col], df_noevent[col]])
    categories = combined.dropna().unique()
    observed = []
    for cat in categories:
        o_e = (df_event[col] == cat).sum()
        o_ne = (df_noevent[col] == cat).sum()
        observed.append([o_e, o_ne])
    observed = np.array(observed)
    observed = observed[observed.sum(axis=1) > 0]
    if observed.shape[0] < 2:
        return ""
    try:
        _, p, _, _ = stats.chi2_contingency(observed)
        return fmt_p(p)
    except ValueError:
        return ""


def fmt_p(p):
    if p < 0.001:
        return "<0.001"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.2f}"


# ── Load data ───────────────────────────────────────────────────────────────
df = pd.read_excel(DATA_PATH, engine="openpyxl")
qc_ids = pd.read_csv(OOF_PATH, usecols=["STUDYID"])["STUDYID"]
df = df[df["STUDYID"].isin(qc_ids)].copy()

# ── Derive variables ────────────────────────────────────────────────────────
# Index event
df["idx_STEMI"] = df["RQINDEX"] == "Acute myocardial infarction STEMI"
df["idx_NSTEMI"] = df["RQINDEX"] == "Acute myocardial infarction Non-STEMI"
df["idx_UA"] = df["RQINDEX"] == "Unstable angina / Acute myocardial ischaemia"
df["idx_ePCI"] = df["RQINDEX"] == "Elective percutaneous transluminal coronary angioplasty"
df["idx_eCABG"] = df["RQINDEX"] == "Elective coronary artery by-pass surgery"

# Binary encodings for display
df["current_smoker_flag"] = (df["CVARCOSMOKING"] == "Yes").astype(int) if df["CVARCOSMOKING"].dtype == object else df["CVARCOSMOKING"]
df["on_statin"] = ~df["RQLIPIDLOWDRUGS"].isin(["No", np.nan])

# Split
df_event = df[df[EVENT_COL] == 1]
df_noevent = df[df[EVENT_COL] == 0]

N = len(df)
N_e = len(df_event)
N_ne = len(df_noevent)

print(f"Total: {N}, Event: {N_e} ({N_e/N*100:.1f}%), No event: {N_ne}")

# ── Build Table 1 ───────────────────────────────────────────────────────────
rows = []
last_header_idx = None

col_overall = f"Overall (N={N})"
col_event = f"Event (N={N_e})"
col_noevent = f"No Event (N={N_ne})"


def add_header(label):
    global last_header_idx
    rows.append({"Variable": f"**{label}**", col_overall: "", col_event: "", col_noevent: "", "P": ""})
    last_header_idx = len(rows) - 1


def add_cont(label, col, dec=1):
    if col not in df.columns:
        return
    vals = pd.to_numeric(df[col], errors="coerce")
    vals_e = pd.to_numeric(df_event[col], errors="coerce")
    vals_ne = pd.to_numeric(df_noevent[col], errors="coerce")
    rows.append({
        "Variable": label,
        col_overall: fmt_continuous(vals, dec),
        col_event: fmt_continuous(vals_e, dec),
        col_noevent: fmt_continuous(vals_ne, dec),
        "P": p_continuous(vals_e, vals_ne),
    })


def add_cat(label, col, positive_val):
    if col not in df.columns:
        return
    rows.append({
        "Variable": label,
        col_overall: fmt_n_pct(df[col], positive_val),
        col_event: fmt_n_pct(df_event[col], positive_val),
        col_noevent: fmt_n_pct(df_noevent[col], positive_val),
        "P": p_categorical(col, df_event, df_noevent),
    })


def add_subcat(label, col, value):
    """Subcategory row (no p-value, indented)."""
    if col not in df.columns:
        return
    rows.append({
        "Variable": f"  {label}",
        col_overall: fmt_n_pct(df[col], value),
        col_event: fmt_n_pct(df_event[col], value),
        col_noevent: fmt_n_pct(df_noevent[col], value),
        "P": "",
    })


# ═══════════════════════════════════════════════════════════════════════════
# DEMOGRAPHICS
# ═══════════════════════════════════════════════════════════════════════════
add_header("Demographics")
add_cont("Age, years", "AGE", 1)
add_cat("Male sex", "RQSEX", "Male")
add_cont("BMI, kg/m²", "RQDCBMI", 1)

# Country distribution (top regions)
rows.append({
    "Variable": "Region",
    col_overall: "", col_event: "", col_noevent: "",
    "P": p_categorical("COUNTRY", df_event, df_noevent),
})
# Group countries into regions for readability
asia = ["Singapore", "China", "Malaysia", "Philippines", "Indonesia", "UAE"]
latam = ["Colombia", "Argentina"]
europe = ["Poland", "Portugal"]
africa = ["Egypt", "Nigeria", "Tanzania", "Kenia"]
df["region"] = df["COUNTRY"].apply(
    lambda x: "Asia" if x in asia else "Latin America" if x in latam
    else "Europe" if x in europe else "Africa" if x in africa else "Other"
)
df_event = df[df[EVENT_COL] == 1]
df_noevent = df[df[EVENT_COL] == 0]
for reg in ["Asia", "Europe", "Latin America", "Africa"]:
    add_subcat(reg, "region", reg)

# ═══════════════════════════════════════════════════════════════════════════
# INDEX EVENT (Model 3 variables)
# ═══════════════════════════════════════════════════════════════════════════
add_header("Index event")
add_subcat("STEMI", "idx_STEMI", True)
add_subcat("NSTEMI", "idx_NSTEMI", True)
add_subcat("Unstable angina", "idx_UA", True)
add_subcat("Elective PCI", "idx_ePCI", True)
add_subcat("Elective CABG", "idx_eCABG", True)
# Overall p for index event
rows[last_header_idx]["P"] = p_categorical("RQINDEX", df_event, df_noevent)

# ═══════════════════════════════════════════════════════════════════════════
# MODEL 1: Clinical risk factors
# ═══════════════════════════════════════════════════════════════════════════
add_header("Clinical risk factors")
add_cont("Systolic BP, mmHg", "RQDCSYSBP", 1)
add_cont("Diastolic BP, mmHg", "RQDCDIASBP", 1)
add_cont("HbA1c, %", "CVARLABHBA1CPERCENT", 1)
add_cont("eGFR, mL/min/1.73m²", "CVAREGFR", 1)
add_cont("HDL-cholesterol, mmol/L", "CVARLABHDL", 2)
add_cont("LDL-cholesterol, mmol/L", "CVARLABLDLFRIEDEWALD", 2)
add_cont("Total cholesterol, mmol/L", "CVARLABTC", 2)
add_cat("Current smoker", "CVARCOSMOKING", "Yes")

# ═══════════════════════════════════════════════════════════════════════════
# MODEL 2: Comorbidities + Medications
# ═══════════════════════════════════════════════════════════════════════════
add_header("Medical history")
add_cat("History of CVD", "RQHISTOFPRECVD", "Yes")
add_cat("History of coronary artery disease", "RQHISTOFCORARTDIS", "Yes")
add_cat("History of heart failure", "RQHISTOFHEARTFAIL", "Yes")
add_cat("History of diabetes", "RQHISTOFDIAB", "Yes")
add_cat("History of atrial fibrillation", "any_af", 1)
add_cat("History of COPD", "RQHISTOFCOPD", "Yes")
add_cat("History of hypertension", "RQHISTOFHYPER", "Yes")

add_header("Medications at baseline")
add_cat("BP-lowering medication", "CVARMEDBPLOWERING", "Yes")
add_cat("Lipid-lowering medication", "CVARMEDLIPIDLOWERING", "Yes")
add_cat("Glucose-lowering medication", "CVARMEDGLUCOSELOWERING", "Yes")
add_cat("Anticoagulant use", "RQANTICOAGULANTS", "Yes")
add_cat("Antiplatelet use", "RQASPIOROTHER", "Yes")

# ═══════════════════════════════════════════════════════════════════════════
# MODEL 4: ALL 18 ECG Biomarkers
# ═══════════════════════════════════════════════════════════════════════════
add_header("ECG findings — Binary")
ecg_binary = [
    ("Sinus Rhythm", "Sinus rhythm"),
    ("Atrial Fibrillation", "Atrial fibrillation"),
    ("LBBB", "Left bundle branch block"),
    ("RBBB", "Right bundle branch block"),
    ("Q Wave", "Q wave"),
    ("ST Elevation", "ST elevation"),
    ("ST Depression", "ST depression"),
    ("T Wave Inversion", "T wave inversion"),
    ("Ischaemic", "Ischaemic changes"),
    ("QT Prolongation", "QT prolongation"),
    ("LVH", "Left ventricular hypertrophy"),
    ("1 AV Block", "First-degree AV block"),
    ("Left Axis Deviation", "Left axis deviation"),
    ("Right Axis Deviation", "Right axis deviation"),
    ("MI (Old)", "Old myocardial infarction"),
    ("MI(Acute)", "Acute myocardial infarction"),
]

for col, label in ecg_binary:
    add_cat(label, col, "Yes")

add_header("ECG findings — Continuous")
ecg_cont = [
    ("PR Interval (ms)", "PR interval, ms"),
    ("QRS Duration (ms)", "QRS duration, ms"),
    ("QT Interval (ms)", "QT interval, ms"),
]
for col, label in ecg_cont:
    add_cont(label, col, 1)

# ── Additional ECG info ─────────────────────────────────────────────────────
add_header("ECG additional (not in model)")
ecg_extra = [
    ("Atrial Flutter", "Atrial flutter"),
    ("RVH", "Right ventricular hypertrophy"),
    ("QT Shortening", "QT shortening"),
    ("2 AV Block", "Second-degree AV block"),
    ("3 AV Block", "Third-degree AV block"),
]
for col, label in ecg_extra:
    if col in df.columns:
        add_cat(label, col, "Yes")

# ═══════════════════════════════════════════════════════════════════════════
# Build and save
# ═══════════════════════════════════════════════════════════════════════════
table1 = pd.DataFrame(rows)
table1.to_csv(OUT_DIR / "table1_CVD_Composite_4.csv", index=False)
shutil.copy2(OUT_DIR / "table1_CVD_Composite_4.csv", FINAL_OUT_DIR / "table1_CVD_Composite_4.csv")

# Pretty print
print("\n" + "=" * 110)
print("TABLE 1: Baseline demographics stratified by CVD_Composite_4")
print("=" * 110)
pd.set_option("display.max_colwidth", 45)
pd.set_option("display.width", 130)
pd.set_option("display.max_rows", 200)
print(table1.to_string(index=False))
print(f"\nSaved to {OUT_DIR / 'table1_CVD_Composite_4.csv'}")
print(f"Copied to {FINAL_OUT_DIR / 'table1_CVD_Composite_4.csv'}")

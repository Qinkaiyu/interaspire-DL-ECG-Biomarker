"""
Generate additional Table 1 variants:
  1. CVD_Composite_6 stratified by Event vs No Event
  2. CVD_Composite_4 stratified by DL-ECG risk tertile (equal-width)
  3. CVD_Composite_6 stratified by DL-ECG risk tertile (equal-width)
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import shutil

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH = Path("data/INTERASPIRE_analysis_dataset.xlsx")
M5_CVD4 = Path("model5_cvd_mace4/E2E_MultiTask_32_2_phase_cvd_mace4_1yr_oof.csv")
M5_CVD6 = Path("model5_cvd_mace6/E2E_MultiTask_32_2_phase_cvd_mace6_1yr_oof.csv")
OUT_DIR = Path("results/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_MAIN_DIR = Path("results/manuscript_final/main_tables")
FINAL_MAIN_DIR.mkdir(parents=True, exist_ok=True)
FINAL_SUPP_DIR = Path("results/manuscript_final/supplementary_tables")
FINAL_SUPP_DIR.mkdir(parents=True, exist_ok=True)


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


def fmt_p(p):
    if p < 0.001:
        return "<0.001"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.2f}"


def p_continuous_2grp(s1, s2):
    s1, s2 = s1.dropna(), s2.dropna()
    if len(s1) < 3 or len(s2) < 3:
        return ""
    _, p = stats.mannwhitneyu(s1, s2, alternative="two-sided")
    return fmt_p(p)


def p_continuous_kw(*groups):
    """Kruskal-Wallis for 3+ groups."""
    cleaned = [g.dropna() for g in groups]
    cleaned = [g for g in cleaned if len(g) >= 3]
    if len(cleaned) < 2:
        return ""
    _, p = stats.kruskal(*cleaned)
    return fmt_p(p)


def p_categorical_2grp(col, df1, df2):
    combined = pd.concat([df1[col], df2[col]])
    categories = combined.dropna().unique()
    observed = []
    for cat in categories:
        observed.append([(df1[col] == cat).sum(), (df2[col] == cat).sum()])
    observed = np.array(observed)
    observed = observed[observed.sum(axis=1) > 0]
    if observed.shape[0] < 2:
        return ""
    try:
        _, p, _, _ = stats.chi2_contingency(observed)
        return fmt_p(p)
    except ValueError:
        return ""


def p_categorical_multi(col, *dfs):
    """Chi-square for 3+ groups."""
    combined = pd.concat([d[col] for d in dfs])
    categories = combined.dropna().unique()
    observed = []
    for cat in categories:
        row = [(d[col] == cat).sum() for d in dfs]
        observed.append(row)
    observed = np.array(observed)
    observed = observed[observed.sum(axis=1) > 0]
    if observed.shape[0] < 2:
        return ""
    try:
        _, p, _, _ = stats.chi2_contingency(observed)
        return fmt_p(p)
    except ValueError:
        return ""


# ── Variable specification (shared across all tables) ──────────────────────
VARIABLE_SPEC = [
    # (type, label, col_or_info, extra)
    # type: "header", "cont", "cat", "subcat"
    ("header", "Demographics", None, None),
    ("cont", "Age, years", "AGE", 1),
    ("cat", "Male sex", "RQSEX", "Male"),
    ("cont", "BMI, kg/m²", "RQDCBMI", 1),
    ("region_block", None, None, None),  # special handling

    ("header", "Index event", None, None),
    ("subcat", "STEMI", "idx_STEMI", True),
    ("subcat", "NSTEMI", "idx_NSTEMI", True),
    ("subcat", "Unstable angina", "idx_UA", True),
    ("subcat", "Elective PCI", "idx_ePCI", True),
    ("subcat_p", "Elective CABG", "idx_eCABG", True),  # last subcat gets p

    ("header", "Clinical risk factors", None, None),
    ("cont", "Systolic BP, mmHg", "RQDCSYSBP", 1),
    ("cont", "Diastolic BP, mmHg", "RQDCDIASBP", 1),
    ("cont", "HbA1c, %", "CVARLABHBA1CPERCENT", 1),
    ("cont", "eGFR, mL/min/1.73m²", "CVAREGFR", 1),
    ("cont", "HDL-cholesterol, mmol/L", "CVARLABHDL", 2),
    ("cont", "LDL-cholesterol, mmol/L", "CVARLABLDLFRIEDEWALD", 2),
    ("cont", "Total cholesterol, mmol/L", "CVARLABTC", 2),
    ("cat", "Current smoker", "CVARCOSMOKING", "Yes"),

    ("header", "Medical history", None, None),
    ("cat", "History of CVD", "RQHISTOFPRECVD", "Yes"),
    ("cat", "History of coronary artery disease", "RQHISTOFCORARTDIS", "Yes"),
    ("cat", "History of heart failure", "RQHISTOFHEARTFAIL", "Yes"),
    ("cat", "History of diabetes", "RQHISTOFDIAB", "Yes"),
    ("cat", "History of atrial fibrillation", "any_af", 1),
    ("cat", "History of COPD", "RQHISTOFCOPD", "Yes"),
    ("cat", "History of hypertension", "RQHISTOFHYPER", "Yes"),

    ("header", "Medications at baseline", None, None),
    ("cat", "BP-lowering medication", "CVARMEDBPLOWERING", "Yes"),
    ("cat", "Lipid-lowering medication", "CVARMEDLIPIDLOWERING", "Yes"),
    ("cat", "Glucose-lowering medication", "CVARMEDGLUCOSELOWERING", "Yes"),
    ("cat", "Anticoagulant use", "RQANTICOAGULANTS", "Yes"),
    ("cat", "Antiplatelet use", "RQASPIOROTHER", "Yes"),

    ("header", "ECG findings — Binary", None, None),
    ("cat", "Sinus rhythm", "Sinus Rhythm", "Yes"),
    ("cat", "Atrial fibrillation", "Atrial Fibrillation", "Yes"),
    ("cat", "Left bundle branch block", "LBBB", "Yes"),
    ("cat", "Right bundle branch block", "RBBB", "Yes"),
    ("cat", "Q wave", "Q Wave", "Yes"),
    ("cat", "ST elevation", "ST Elevation", "Yes"),
    ("cat", "ST depression", "ST Depression", "Yes"),
    ("cat", "T wave inversion", "T Wave Inversion", "Yes"),
    ("cat", "Ischaemic changes", "Ischaemic", "Yes"),
    ("cat", "QT prolongation", "QT Prolongation", "Yes"),
    ("cat", "Left ventricular hypertrophy", "LVH", "Yes"),
    ("cat", "First-degree AV block", "1 AV Block", "Yes"),
    ("cat", "Left axis deviation", "Left Axis Deviation", "Yes"),
    ("cat", "Right axis deviation", "Right Axis Deviation", "Yes"),
    ("cat", "Old myocardial infarction", "MI (Old)", "Yes"),
    ("cat", "Acute myocardial infarction", "MI(Acute)", "Yes"),

    ("header", "ECG findings — Continuous", None, None),
    ("cont", "PR interval, ms", "PR Interval (ms)", 1),
    ("cont", "QRS duration, ms", "QRS Duration (ms)", 1),
    ("cont", "QT interval, ms", "QT Interval (ms)", 1),
]


def build_table_2grp(df_full, df_g1, df_g2, col_all, col_g1, col_g2):
    """Build table for 2-group comparison (Event vs No Event)."""
    rows = []
    last_header_idx = None
    for spec in VARIABLE_SPEC:
        vtype, label, col, extra = spec

        if vtype == "header":
            rows.append({"Variable": f"**{label}**",
                          col_all: "", col_g1: "", col_g2: "", "P": ""})
            last_header_idx = len(rows) - 1

        elif vtype == "region_block":
            rows.append({
                "Variable": "Region",
                col_all: "", col_g1: "", col_g2: "",
                "P": p_categorical_2grp("COUNTRY", df_g1, df_g2),
            })
            for reg in ["Asia", "Europe", "Latin America", "Africa"]:
                rows.append({
                    "Variable": f"  {reg}",
                    col_all: fmt_n_pct(df_full["region"], reg),
                    col_g1: fmt_n_pct(df_g1["region"], reg),
                    col_g2: fmt_n_pct(df_g2["region"], reg),
                    "P": "",
                })

        elif vtype == "cont":
            if col not in df_full.columns:
                continue
            vals = pd.to_numeric(df_full[col], errors="coerce")
            v1 = pd.to_numeric(df_g1[col], errors="coerce")
            v2 = pd.to_numeric(df_g2[col], errors="coerce")
            rows.append({
                "Variable": label,
                col_all: fmt_continuous(vals, extra),
                col_g1: fmt_continuous(v1, extra),
                col_g2: fmt_continuous(v2, extra),
                "P": p_continuous_2grp(v1, v2),
            })

        elif vtype == "cat":
            if col not in df_full.columns:
                continue
            rows.append({
                "Variable": label,
                col_all: fmt_n_pct(df_full[col], extra),
                col_g1: fmt_n_pct(df_g1[col], extra),
                col_g2: fmt_n_pct(df_g2[col], extra),
                "P": p_categorical_2grp(col, df_g1, df_g2),
            })

        elif vtype in ("subcat", "subcat_p"):
            if col not in df_full.columns:
                continue
            rows.append({
                "Variable": f"  {label}",
                col_all: fmt_n_pct(df_full[col], extra),
                col_g1: fmt_n_pct(df_g1[col], extra),
                col_g2: fmt_n_pct(df_g2[col], extra),
                "P": "",
            })
            if vtype == "subcat_p" and last_header_idx is not None:
                rows[last_header_idx]["P"] = p_categorical_2grp("RQINDEX", df_g1, df_g2)

    return pd.DataFrame(rows)


def build_table_3grp(df_full, df_low, df_mid, df_high, col_all, col_low, col_mid, col_high):
    """Build table for 3-group comparison (DL-ECG risk tertile)."""
    rows = []
    last_header_idx = None
    for spec in VARIABLE_SPEC:
        vtype, label, col, extra = spec

        if vtype == "header":
            rows.append({"Variable": f"**{label}**",
                          col_all: "", col_low: "", col_mid: "", col_high: "", "P": ""})
            last_header_idx = len(rows) - 1

        elif vtype == "region_block":
            rows.append({
                "Variable": "Region",
                col_all: "", col_low: "", col_mid: "", col_high: "",
                "P": p_categorical_multi("COUNTRY", df_low, df_mid, df_high),
            })
            for reg in ["Asia", "Europe", "Latin America", "Africa"]:
                rows.append({
                    "Variable": f"  {reg}",
                    col_all: fmt_n_pct(df_full["region"], reg),
                    col_low: fmt_n_pct(df_low["region"], reg),
                    col_mid: fmt_n_pct(df_mid["region"], reg),
                    col_high: fmt_n_pct(df_high["region"], reg),
                    "P": "",
                })

        elif vtype == "cont":
            if col not in df_full.columns:
                continue
            vals = pd.to_numeric(df_full[col], errors="coerce")
            v_l = pd.to_numeric(df_low[col], errors="coerce")
            v_m = pd.to_numeric(df_mid[col], errors="coerce")
            v_h = pd.to_numeric(df_high[col], errors="coerce")
            rows.append({
                "Variable": label,
                col_all: fmt_continuous(vals, extra),
                col_low: fmt_continuous(v_l, extra),
                col_mid: fmt_continuous(v_m, extra),
                col_high: fmt_continuous(v_h, extra),
                "P": p_continuous_kw(v_l, v_m, v_h),
            })

        elif vtype == "cat":
            if col not in df_full.columns:
                continue
            rows.append({
                "Variable": label,
                col_all: fmt_n_pct(df_full[col], extra),
                col_low: fmt_n_pct(df_low[col], extra),
                col_mid: fmt_n_pct(df_mid[col], extra),
                col_high: fmt_n_pct(df_high[col], extra),
                "P": p_categorical_multi(col, df_low, df_mid, df_high),
            })

        elif vtype in ("subcat", "subcat_p"):
            if col not in df_full.columns:
                continue
            rows.append({
                "Variable": f"  {label}",
                col_all: fmt_n_pct(df_full[col], extra),
                col_low: fmt_n_pct(df_low[col], extra),
                col_mid: fmt_n_pct(df_mid[col], extra),
                col_high: fmt_n_pct(df_high[col], extra),
                "P": "",
            })
            if vtype == "subcat_p" and last_header_idx is not None:
                rows[last_header_idx]["P"] = p_categorical_multi("RQINDEX", df_low, df_mid, df_high)

    return pd.DataFrame(rows)


# ── Load data ───────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_excel(DATA_PATH, engine="openpyxl")

# Derive variables
df["idx_STEMI"] = df["RQINDEX"] == "Acute myocardial infarction STEMI"
df["idx_NSTEMI"] = df["RQINDEX"] == "Acute myocardial infarction Non-STEMI"
df["idx_UA"] = df["RQINDEX"] == "Unstable angina / Acute myocardial ischaemia"
df["idx_ePCI"] = df["RQINDEX"] == "Elective percutaneous transluminal coronary angioplasty"
df["idx_eCABG"] = df["RQINDEX"] == "Elective coronary artery by-pass surgery"

asia = ["Singapore", "China", "Malaysia", "Philippines", "Indonesia", "UAE"]
latam = ["Colombia", "Argentina"]
europe = ["Poland", "Portugal"]
africa = ["Egypt", "Nigeria", "Tanzania", "Kenia"]
df["region"] = df["COUNTRY"].apply(
    lambda x: "Asia" if x in asia else "Latin America" if x in latam
    else "Europe" if x in europe else "Africa" if x in africa else "Other"
)


def load_m5_and_assign_tertile(df_base, m5_path, event_col):
    """Load Model 5 OOF, per-fold z-normalize, assign equal-width tertile."""
    oof = pd.read_csv(m5_path)
    # Per-fold z-score normalization
    risk_norm = oof["oof_log_risk"].copy()
    for fold in oof["test_fold"].unique():
        mask = oof["test_fold"] == fold
        vals = oof.loc[mask, "oof_log_risk"]
        risk_norm.loc[mask] = (vals - vals.mean()) / vals.std()
    oof["risk_norm"] = risk_norm

    # Merge with main data
    merged = df_base.merge(oof[["STUDYID", "risk_norm"]], on="STUDYID", how="inner")
    print(f"  Merged N={len(merged)} (M5 for {event_col})")

    # Equal-width tertile on risk range
    rmin, rmax = merged["risk_norm"].min(), merged["risk_norm"].max()
    t1 = rmin + (rmax - rmin) / 3
    t2 = rmin + 2 * (rmax - rmin) / 3
    merged["risk_group"] = pd.cut(
        merged["risk_norm"],
        bins=[rmin - 0.001, t1, t2, rmax + 0.001],
        labels=["Low", "Intermediate", "High"],
    )
    for g in ["Low", "Intermediate", "High"]:
        sub = merged[merged["risk_group"] == g]
        n_evt = sub[event_col].sum() if event_col in sub.columns else "?"
        print(f"    {g}: N={len(sub)}, events={n_evt}")
    return merged


# ═════════════════════════════════════════════════════════════════════════════
# TABLE 1a: CVD_Composite_6 (Event vs No Event)
# ═════════════════════════════════════════════════════════════════════════════
print("\n=== Table 1: CVD_Composite_6 ===")
evt_col6 = "event_CVD_Composite_6"
m5_cvd6_ids = pd.read_csv(M5_CVD6, usecols=["STUDYID"])["STUDYID"]
df6 = df[df["STUDYID"].isin(m5_cvd6_ids)].copy()
df_e6 = df6[df6[evt_col6] == 1]
df_ne6 = df6[df6[evt_col6] == 0]
N6 = len(df6)
N_e6 = len(df_e6)
N_ne6 = len(df_ne6)
print(f"Total: {N6}, Event: {N_e6} ({N_e6/N6*100:.1f}%), No event: {N_ne6}")

col_all6 = f"Overall (N={N6})"
col_evt6 = f"Event (N={N_e6})"
col_nevt6 = f"No Event (N={N_ne6})"

t1_cvd6 = build_table_2grp(df6, df_e6, df_ne6, col_all6, col_evt6, col_nevt6)
out6 = OUT_DIR / "table1_CVD_Composite_6.csv"
t1_cvd6.to_csv(out6, index=False)
shutil.copy2(out6, FINAL_SUPP_DIR / "table1_CVD_Composite_6.csv")
print(f"Saved: {out6}")

# ═════════════════════════════════════════════════════════════════════════════
# TABLE 2a: DL-ECG risk tertile — CVD_Composite_4
# ═════════════════════════════════════════════════════════════════════════════
print("\n=== Table: DL-ECG risk tertile (CVD4) ===")
merged4 = load_m5_and_assign_tertile(df, M5_CVD4, "event_CVD_Composite_4")

df_low4 = merged4[merged4["risk_group"] == "Low"]
df_mid4 = merged4[merged4["risk_group"] == "Intermediate"]
df_high4 = merged4[merged4["risk_group"] == "High"]

N4 = len(merged4)
col_all4t = f"Overall (N={N4})"
col_low4 = f"Low risk (N={len(df_low4)})"
col_mid4 = f"Intermediate risk (N={len(df_mid4)})"
col_high4 = f"High risk (N={len(df_high4)})"

t2_cvd4 = build_table_3grp(merged4, df_low4, df_mid4, df_high4,
                            col_all4t, col_low4, col_mid4, col_high4)

# Add event rate row at top
evt_row = {
    "Variable": "1-year CVD events",
    col_all4t: fmt_n_pct(merged4["event_CVD_Composite_4"], 1),
    col_low4: fmt_n_pct(df_low4["event_CVD_Composite_4"], 1),
    col_mid4: fmt_n_pct(df_mid4["event_CVD_Composite_4"], 1),
    col_high4: fmt_n_pct(df_high4["event_CVD_Composite_4"], 1),
    "P": "",
}
t2_cvd4 = pd.concat([pd.DataFrame([evt_row]), t2_cvd4], ignore_index=True)

out4t = OUT_DIR / "table_dlecg_tertile_CVD4.csv"
t2_cvd4.to_csv(out4t, index=False)
shutil.copy2(out4t, FINAL_MAIN_DIR / "table_dlecg_tertile_CVD4.csv")
print(f"Saved: {out4t}")

# ═════════════════════════════════════════════════════════════════════════════
# TABLE 2b: DL-ECG risk tertile — CVD_Composite_6
# ═════════════════════════════════════════════════════════════════════════════
print("\n=== Table: DL-ECG risk tertile (CVD6) ===")
merged6 = load_m5_and_assign_tertile(df, M5_CVD6, "event_CVD_Composite_6")

df_low6 = merged6[merged6["risk_group"] == "Low"]
df_mid6 = merged6[merged6["risk_group"] == "Intermediate"]
df_high6 = merged6[merged6["risk_group"] == "High"]

N6t = len(merged6)
col_all6t = f"Overall (N={N6t})"
col_low6 = f"Low risk (N={len(df_low6)})"
col_mid6 = f"Intermediate risk (N={len(df_mid6)})"
col_high6 = f"High risk (N={len(df_high6)})"

t2_cvd6 = build_table_3grp(merged6, df_low6, df_mid6, df_high6,
                            col_all6t, col_low6, col_mid6, col_high6)

evt_row6 = {
    "Variable": "1-year CVD events",
    col_all6t: fmt_n_pct(merged6["event_CVD_Composite_6"], 1),
    col_low6: fmt_n_pct(df_low6["event_CVD_Composite_6"], 1),
    col_mid6: fmt_n_pct(df_mid6["event_CVD_Composite_6"], 1),
    col_high6: fmt_n_pct(df_high6["event_CVD_Composite_6"], 1),
    "P": "",
}
t2_cvd6 = pd.concat([pd.DataFrame([evt_row6]), t2_cvd6], ignore_index=True)

out6t = OUT_DIR / "table_dlecg_tertile_CVD6.csv"
t2_cvd6.to_csv(out6t, index=False)
shutil.copy2(out6t, FINAL_SUPP_DIR / "table_dlecg_tertile_CVD6.csv")
print(f"Saved: {out6t}")

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("All tables generated:")
print(f"  1. {out6}")
print(f"  2. {out4t}")
print(f"  3. {out6t}")
print("=" * 70)

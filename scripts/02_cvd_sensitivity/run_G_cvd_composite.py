#!/usr/bin/env python3
"""
Step G: Re-run Models 1–4 with CVD-mortality composite endpoint.

CVD death definition (per protocol):
  DEATHCAUSE_excel2 in {1=CHD, 2=Stroke, 3=Other vascular}
  + code 6 (Other) only if DEATHICD10 starts with 'I' or is '121.*'
  → code 6 cases examined: all have non-CVD ICD (sepsis/pneumonia/GI bleed)
  → effectively: CVD death = DEATHCAUSE in {1, 2, 3}  (N=46)

New composite endpoint (CVD_Composite_4):
  CVD death  OR  HOSPAMI  OR  HOSPHF  OR  HOSPSTROKE  (within 1 year)

Patients with non-CVD death as first event → recoded as censored at death time.

Model 4 uses the 3 consensus-selected ECG features from Steps A–F:
  QT Prolongation (flag), ST Depression, QT Interval (ms, binned)
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from pathlib import Path
import time

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
import statsmodels.api as sm
from scipy import stats
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA_PATH = Path("data/INTERASPIRE_analysis_dataset.xlsx")
OUT_DIR   = Path("results/feature_selection")
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_FOLDS      = 5

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# ── Variable definitions (same as analysis_prognosis_v2.py) ───────────────────
BIN_EDGES = {
    "PR Interval (ms)":[0,120,200,2000], "QRS Duration (ms)":[0,120,1000],
    "QT Interval (ms)":[0,440,480,5000], "FHPRINTER":[0,120,200,2000],
    "FHQRSDURA":[0,120,1000],            "FHQTC":[0,440,480,5000],
    "CVARSYSTOL":[0,120,130,140,180,250],"CVARLABHBA1CPERCENT":[0,5.7,6.5,20],
    "CVARLABLDLFRIEDEWALD":[0,1.8,2.6,3.4,4.9,10],
    "CVARLABHDL":[0,1.0,1.3,1.55,5],   "CVAREGFR":[0,15,30,45,60,90,200],
    "CVARBMI":[0,18.5,25,30,35,100],    "CVARAGE":[0,50,60,70,80,150],
}

M1_VARS = ["CVARAGE","RQSEX","CVARCOSMOKING","CVARBMI","CVARSYSTOL",
           "CVARLABHBA1CPERCENT","CVAREGFR","CVARLABHDL","CVARLABLDLFRIEDEWALD"]
M2_VARS = M1_VARS + ["RQHISTOFPRECVD","RQHISTOFCORARTDIS","RQHISTOFHEARTFAIL",
                     "any_af","CVARMEDBPLOWERING","CVARMEDLIPIDLOWERING",
                     "CVARMEDGLUCOSELOWERING","RQANTICOAGULANTS","RQHISTOFCOPD"]
M3_VARS = M2_VARS + ["idx_STEMI","idx_NSTEMI","idx_UA","idx_ePCI"]

# Model 4: M3 + 3 consensus-selected ECG features (Step F recommendation)
ECG_SELECTED = ["QT Prolongation", "ST Depression", "QT Interval (ms)"]
M4_VARS = M3_VARS + ECG_SELECTED

ECG_ALL_BINARY = [
    "Atrial Fibrillation","LBBB","RBBB","Q Wave","ST Elevation","ST Depression",
    "T Wave Inversion","Ischaemic","QT Prolongation","LVH","1 AV Block",
    "Left Axis Deviation","Right Axis Deviation","MI (Old)","MI(Acute)",
]

MODELS = [
    ("Model 1", M1_VARS),
    ("Model 2", M2_VARS),
    ("Model 3", M3_VARS),
    ("Model 4 (CVD, 3 ECG)", M4_VARS),
]

# ── Load & preprocess ─────────────────────────────────────────────────────────
log("Loading data …")
df = pd.read_excel(DATA_PATH)

def enc(s): return s.map({"Yes":1,"No":0,"yes":1,"no":0})
def binn(s,e): return pd.cut(s,bins=e,labels=range(len(e)-1),include_lowest=True).astype(float)
def is_cvd_icd(icd):
    if pd.isna(icd): return False
    s = str(icd).strip().upper()
    return s.startswith("I") or s.startswith("121")

df["RQSEX"] = df["RQSEX"].map({"Male":1,"Female":0})
for v in ["CVARCOSMOKING","RQHISTOFPRECVD","RQHISTOFCORARTDIS","RQHISTOFHEARTFAIL",
          "CVARMEDBPLOWERING","CVARMEDLIPIDLOWERING","CVARMEDGLUCOSELOWERING",
          "RQANTICOAGULANTS","RQHISTOFCOPD"]:
    df[v] = enc(df[v])
for v in ECG_ALL_BINARY:
    if v in df.columns: df[v] = enc(df[v])

df["idx_STEMI"] =(df["RQINDEX"]=="Acute myocardial infarction STEMI").astype(float)
df["idx_NSTEMI"]=(df["RQINDEX"]=="Acute myocardial infarction Non-STEMI").astype(float)
df["idx_UA"]    =(df["RQINDEX"]=="Unstable angina / Acute myocardial ischaemia").astype(float)
df["idx_ePCI"]  =(df["RQINDEX"]=="Elective percutaneous transluminal coronary angioplasty").astype(float)

# FH ECG fallback
for ai,fh in [("PR Interval (ms)","FHPRINTER"),("QRS Duration (ms)","FHQRSDURA"),("QT Interval (ms)","FHQTC")]:
    m = df[ai].isna()&df[fh].notna(); df.loc[m,ai]=df.loc[m,fh]
m = df["RBBB"].isna()
df.loc[m&(df["FHBUNBRBLO"]=="Right bundle branch block"),"RBBB"]=1
df.loc[m&df["FHBUNBRBLO"].notna()&(df["FHBUNBRBLO"]!="Right bundle branch block"),"RBBB"]=0
df.loc[m&(df["FHBUNBRBLO"]=="Left bundle branch block"),"LBBB"]=1
df.loc[m&df["FHBUNBRBLO"].notna()&(df["FHBUNBRBLO"]!="Left bundle branch block"),"LBBB"]=0
m2 = df["1 AV Block"].isna()
df.loc[m2&(df["FHAVNOBL"]=="I°"),"1 AV Block"]=1
df.loc[m2&df["FHAVNOBL"].notna()&(df["FHAVNOBL"]!="I°"),"1 AV Block"]=0
mlvh = df["LVH"].isna(); flvh = enc(df["FHLEFTVENHYP"])
df.loc[mlvh&flvh.notna(),"LVH"] = flvh[mlvh&flvh.notna()]

for col,edges in BIN_EDGES.items():
    if col in df.columns: df[col]=binn(df[col],edges)

# ── Define CVD composite endpoint ─────────────────────────────────────────────
log("Defining CVD composite endpoint …")

# CVD death flag
cvd_death = (
    df["DEATHCAUSE_excel2"].isin([1,2,3]) |
    ((df["DEATHCAUSE_excel2"]==6) & df["DEATHICD10_excel2"].apply(is_cvd_icd))
)

# Original time (clip 0-365)
df["time_Composite_4"] = df["time_Composite_4"].clip(0,365)

# First component of Composite_4
first_comp = df["first_component_Composite_4"].fillna("")
death_first = (first_comp == "SURVIVAL")
hosp_first  = first_comp.isin(["HOSPAMI","HOSPHF","HOSPSTROKE"])

# New CVD composite event:
#   - CVD death as first event → event=1
#   - Hospitalization (AMI/HF/stroke) as first event → event=1
#   - Non-CVD death as first event → event=0 (censor at death time)
#   - No event → event=0
df["event_CVD4"] = ((death_first & cvd_death) | hosp_first).astype(int)
df["time_CVD4"]  = df["time_Composite_4"]  # same time variable

# Summary
n_allcause_death_first = death_first.sum()
n_cvd_death_first      = (death_first & cvd_death).sum()
n_noncvd_recoded       = (death_first & ~cvd_death).sum()
n_hosp_first           = hosp_first.sum()

log(f"  All-cause deaths as first Composite_4 event: {n_allcause_death_first}")
log(f"    → CVD deaths (kept as events):  {n_cvd_death_first}")
log(f"    → Non-CVD deaths (now censored): {n_noncvd_recoded}")
log(f"  Hospitalisations as first event:  {n_hosp_first}")
log(f"  Total CVD_Composite_4 events:     {df['event_CVD4'].sum()}")

# ── Landmark cohort ───────────────────────────────────────────────────────────
complete = (df["time_CVD4"] >= 365) | (df["event_CVD4"] == 1)
df_lr  = df[complete].reset_index(drop=True)
df_cox = df.copy()
y_lr   = df_lr["event_CVD4"].values.astype(float)

log(f"\n  Landmark LR cohort: N={len(df_lr)}, events={int(y_lr.sum())}")
log(f"  Full Cox cohort:    N={len(df_cox)}, events={int(df_cox['event_CVD4'].sum())}")

# Compare with all-cause composite
log(f"\n  [Compare] All-cause Composite_4: N_lr={complete.sum()}, "
    f"events_lr={df['event_Composite_4'][complete].sum()}")

# ── CV setup ──────────────────────────────────────────────────────────────────
skf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
folds_lr = list(skf.split(df_lr, df_lr["event_CVD4"]))
folds_cx = list(skf.split(df_cox, df_cox["event_CVD4"]))

def prep(Xtr, Xte):
    imp = SimpleImputer(strategy="median"); sc = StandardScaler()
    Xtr_s = sc.fit_transform(imp.fit_transform(Xtr))
    Xte_s = sc.transform(imp.transform(Xte))
    return Xtr_s, Xte_s, list(Xtr.columns)

def lr_cv(data, vlist, folds, evt_col):
    avail = [v for v in vlist if v in data.columns]
    X = data[avail]; y = data[evt_col].values.astype(float)
    aucs, briers, oof = [], [], np.full(len(data), np.nan)
    for tr,te in folds:
        Xtr,Xte,_ = prep(X.iloc[tr], X.iloc[te])
        m = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                               max_iter=1000, random_state=RANDOM_STATE)
        m.fit(Xtr, y[tr]); p = m.predict_proba(Xte)[:,1]
        oof[te] = p
        if len(np.unique(y[te])) > 1: aucs.append(roc_auc_score(y[te],p))
        briers.append(brier_score_loss(y[te],p))
    valid = ~np.isnan(oof)
    try:
        logit = np.log(oof[valid]/(1-oof[valid]+1e-9)+1e-9)
        cm = sm.Logit(y[valid], sm.add_constant(logit)).fit(disp=False)
        cal = cm.params[1]
    except: cal = np.nan
    return np.mean(aucs), np.std(aucs), np.mean(briers), cal, oof, y

def cox_cv(data, vlist, folds, time_col, evt_col):
    avail = [v for v in vlist if v in data.columns]
    X = data[avail]; t = data[time_col].values; e = data[evt_col].values
    cs = []
    for tr,te in folds:
        Xtr,Xte,cols = prep(X.iloc[tr], X.iloc[te])
        tdf = pd.DataFrame(Xtr, columns=cols)
        tdf["_t"] = t[tr]; tdf["_e"] = e[tr]
        cph = CoxPHFitter(penalizer=0.1)
        try:
            cph.fit(tdf, "_t", "_e", show_progress=False)
            risk = cph.predict_partial_hazard(pd.DataFrame(Xte,columns=cols)).values
            cs.append(concordance_index(t[te], -risk, e[te]))
        except: pass
    return (np.mean(cs) if cs else np.nan), (np.std(cs) if cs else np.nan)

# ── Run models ────────────────────────────────────────────────────────────────
log(f"\n── Running 5-fold CV for {len(MODELS)} models (CVD composite) ──")
perf_rows = []; oof_store = {}

for mname, vlist in MODELS:
    auc,auc_sd,brier,cal,oof,ytrue = lr_cv(df_lr, vlist, folds_lr, "event_CVD4")
    c,c_sd = cox_cv(df_cox, vlist, folds_cx, "time_CVD4", "event_CVD4")
    avail_n = len([v for v in vlist if v in df_lr.columns])
    perf_rows.append({
        "Model": mname,
        "Endpoint": "CVD Composite_4",
        "N": len(df_lr), "Events": int(y_lr.sum()),
        "Features": avail_n,
        "AUC mean": round(auc,4), "AUC SD": round(auc_sd,4),
        "AUC 95%CI": f"{auc-1.96*auc_sd:.4f}–{auc+1.96*auc_sd:.4f}",
        "Brier": round(brier,4),
        "Cal slope": round(cal,3) if not np.isnan(cal) else np.nan,
        "Cox C-stat": round(c,4) if not np.isnan(c) else np.nan,
        "Cox C SD": round(c_sd,4),
    })
    oof_store[mname] = (oof, ytrue)
    log(f"  {mname:30s}  AUC={auc:.4f}±{auc_sd:.4f}  "
        f"Cal={cal:.3f}  Cox-C={c:.4f}")

# ΔAUC vs Model 1
base_auc = perf_rows[0]["AUC mean"]; base_sd = perf_rows[0]["AUC SD"]
for row in perf_rows[1:]:
    d = row["AUC mean"] - base_auc
    se = np.sqrt(row["AUC SD"]**2 + base_sd**2)
    z = d/(se+1e-10); p = 2*(1-stats.norm.cdf(abs(z)))
    row["Delta AUC vs M1"] = round(d,4)
    row["p vs M1"] = round(p,4)

perf_df = pd.DataFrame(perf_rows)
perf_df.to_csv(OUT_DIR/"cvd_model_performance.csv", index=False)
log("\nSaved → cvd_model_performance.csv")

# ── Comparison table: CVD vs All-cause ───────────────────────────────────────
log("\n── Loading all-cause results for comparison ──")
try:
    ac = pd.read_csv(OUT_DIR/"model_performance.csv")
    ac_m3 = ac[ac["Model"]=="Model 3 (no ECG)"]
    ac_m4 = ac[ac["Model"]=="4e: Stable ≥50% (3)"]
    log(f"  All-cause M3:  AUC={ac_m3['AUC mean'].values[0]:.4f}  Cox-C={ac_m3['Cox C-stat'].values[0]:.4f}")
    log(f"  All-cause M4e: AUC={ac_m4['AUC mean'].values[0]:.4f}  Cox-C={ac_m4['Cox C-stat'].values[0]:.4f}")
except: log("  (all-cause model_performance.csv not found)")

# ── Plots ─────────────────────────────────────────────────────────────────────
log("\nGenerating plots …")
COLORS = ["#607D8B","#4CAF50","#2196F3","#E91E63"]

# Plot 1: AUC bar chart
fig, ax = plt.subplots(figsize=(9,5))
xpos = np.arange(len(perf_rows))
means = [r["AUC mean"] for r in perf_rows]
sds   = [r["AUC SD"]   for r in perf_rows]
bars  = ax.bar(xpos, means, yerr=sds, color=COLORS, capsize=5, alpha=0.85)
ax.set_xticks(xpos)
ax.set_xticklabels([r["Model"] for r in perf_rows], rotation=15, ha="right", fontsize=9)
ax.set_ylabel("AUC (5-fold CV)")
ax.set_title("CVD Composite Endpoint — Model Discrimination\n"
             "(CVD death / HF hosp / MI hosp / Stroke hosp, 1-year)")
ax.set_ylim(0.55, 0.80)
for bar,m,s in zip(bars,means,sds):
    ax.text(bar.get_x()+bar.get_width()/2, m+s+0.004, f"{m:.4f}",
            ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR/"cvd_auc_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 2: ROC curves
fig, ax = plt.subplots(figsize=(7,7))
ax.plot([0,1],[0,1],"k--",lw=0.8,label="Chance")
for (mname,_),color in zip(MODELS, COLORS):
    oof_arr, ytrue = oof_store[mname]
    valid = ~np.isnan(oof_arr)
    fpr, tpr, _ = roc_curve(ytrue[valid], oof_arr[valid])
    auc = roc_auc_score(ytrue[valid], oof_arr[valid])
    lw = 2.5 if mname == "Model 3" else 1.8
    ax.plot(fpr, tpr, color=color, lw=lw, label=f"{mname}  AUC={auc:.4f}")
ax.set_xlabel("1–Specificity"); ax.set_ylabel("Sensitivity")
ax.set_title("ROC Curves — CVD Composite Endpoint (5-fold CV OOF)")
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim(0,1); ax.set_ylim(0,1)
plt.tight_layout()
plt.savefig(OUT_DIR/"cvd_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()

log("  Plots saved")

# ── Text summary ──────────────────────────────────────────────────────────────
txt = [
    "="*70,
    "CVD Composite Endpoint Analysis  (Step G)",
    f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    "="*70, "",
    "ENDPOINT DEFINITION",
    "-"*50,
    "  CVD death = DEATHCAUSE in {1=CHD, 2=Stroke, 3=Other vascular}",
    "             OR code 6 with cardiovascular ICD-10 (I*)",
    f"  CVD deaths total (cohort):   46",
    f"    - Code 1 (CHD):            36",
    f"    - Code 2 (Stroke):          5",
    f"    - Code 3 (Other vascular):  5",
    f"    - Code 6 + CVD ICD:         0 (all code-6 ICD non-cardiovascular)",
    "",
    "  CVD Composite_4 = CVD death OR HOSPAMI OR HOSPHF OR HOSPSTROKE",
    "",
    "RECODING OF ALL-CAUSE DEATHS",
    "-"*50,
    f"  Deaths as first Composite_4 event:  {n_allcause_death_first}",
    f"    → CVD (kept as events):           {n_cvd_death_first}",
    f"    → Non-CVD (recoded to censored):  {n_noncvd_recoded}",
    f"      (7 COVID, 4 cancer, 6 other-non-CV, 11 unknown cause)",
    "",
    "COHORT",
    "-"*50,
    f"  LR landmark cohort: N={len(df_lr)}, events={int(y_lr.sum())} ({100*y_lr.mean():.1f}%)",
    f"  [All-cause Composite_4 for reference: N=3361, events=195 (5.8%)]",
    "",
    "MODEL PERFORMANCE (5-fold CV, CVD composite endpoint)",
    "-"*50,
]
for row in perf_rows:
    d = row.get("Delta AUC vs M1",""); p = row.get("p vs M1","")
    dstr = f"  ΔAUC={d:+.4f} p={p:.4f}" if isinstance(d,float) else ""
    txt.append(f"  {row['Model']:35s}  AUC={row['AUC mean']:.4f}±{row['AUC SD']:.4f}  "
               f"Cal={row['Cal slope']}  Cox-C={row['Cox C-stat']}{dstr}")
txt += [
    "",
    "COMPARISON: All-cause vs CVD composite (Model 3)",
    "-"*50,
    "  See results_v2/model_performance.csv (Primary endpoint rows)",
    "  for all-cause Composite_4 results to compare directly.",
    "",
    "OUTPUT FILES",
    "-"*50,
    "  results/feature_selection/cvd_model_performance.csv",
    "  results/feature_selection/cvd_auc_comparison.png",
    "  results/feature_selection/cvd_roc_curves.png",
    "="*70,
]

summary = "\n".join(txt)
print("\n"+summary)
(OUT_DIR/"cvd_summary.txt").write_text(summary)
log("✓ Step G done.")

#!/usr/bin/env python3
"""
Step H: Full follow-up CVD composite Cox model.

Composite = CVD death (any time) OR hospitalization (HOSPAMI/HOSPHF/HOSPSTROKE, within 1 year)
- Hospitalization data: available within 1 year only (by study design)
- CVD deaths: extended to full follow-up using DEATHDATE - INTDATE

Compared to 1-year CVD composite (run_G, 167 events):
  + 7 additional CVD deaths occurring after 365 days (no prior hospitalization event)
  = 174 total events

The 3 overlap cases (hospitalization first, then CVD death >365d) are already
events in the 1-year composite (hospitalization as first event) — not double-counted.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from pathlib import Path
import time

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = Path("data/INTERASPIRE_analysis_dataset.xlsx")
OUT_DIR   = Path("results/feature_selection")

RANDOM_STATE = 42
N_FOLDS = 5

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

BIN_EDGES = {
    "PR Interval (ms)":[0,120,200,2000],"QRS Duration (ms)":[0,120,1000],
    "QT Interval (ms)":[0,440,480,5000],"FHPRINTER":[0,120,200,2000],
    "FHQRSDURA":[0,120,1000],"FHQTC":[0,440,480,5000],
    "CVARSYSTOL":[0,120,130,140,180,250],"CVARLABHBA1CPERCENT":[0,5.7,6.5,20],
    "CVARLABLDLFRIEDEWALD":[0,1.8,2.6,3.4,4.9,10],"CVARLABHDL":[0,1.0,1.3,1.55,5],
    "CVAREGFR":[0,15,30,45,60,90,200],"CVARBMI":[0,18.5,25,30,35,100],
    "CVARAGE":[0,50,60,70,80,150],
}
M1_VARS = ["CVARAGE","RQSEX","CVARCOSMOKING","CVARBMI","CVARSYSTOL",
           "CVARLABHBA1CPERCENT","CVAREGFR","CVARLABHDL","CVARLABLDLFRIEDEWALD"]
M2_VARS = M1_VARS + ["RQHISTOFPRECVD","RQHISTOFCORARTDIS","RQHISTOFHEARTFAIL",
                     "any_af","CVARMEDBPLOWERING","CVARMEDLIPIDLOWERING",
                     "CVARMEDGLUCOSELOWERING","RQANTICOAGULANTS","RQHISTOFCOPD"]
M3_VARS = M2_VARS + ["idx_STEMI","idx_NSTEMI","idx_UA","idx_ePCI"]
M4_VARS = M3_VARS + ["QT Prolongation","ST Depression","QT Interval (ms)"]

ECG_BIN = ["Atrial Fibrillation","LBBB","RBBB","Q Wave","ST Elevation","ST Depression",
           "T Wave Inversion","Ischaemic","QT Prolongation","LVH","1 AV Block",
           "Left Axis Deviation","Right Axis Deviation","MI (Old)","MI(Acute)"]
MODELS = [("Model 1",M1_VARS),("Model 2",M2_VARS),
          ("Model 3",M3_VARS),("Model 4 (3 ECG)",M4_VARS)]

# ── Load & preprocess ─────────────────────────────────────────────────────────
log("Loading data …")
df = pd.read_excel(DATA_PATH)

def enc(s): return s.map({"Yes":1,"No":0,"yes":1,"no":0})
def binn(s,e): return pd.cut(s,bins=e,labels=range(len(e)-1),include_lowest=True).astype(float)

df["RQSEX"]=df["RQSEX"].map({"Male":1,"Female":0})
for v in ["CVARCOSMOKING","RQHISTOFPRECVD","RQHISTOFCORARTDIS","RQHISTOFHEARTFAIL",
          "CVARMEDBPLOWERING","CVARMEDLIPIDLOWERING","CVARMEDGLUCOSELOWERING",
          "RQANTICOAGULANTS","RQHISTOFCOPD"]:
    df[v]=enc(df[v])
for v in ECG_BIN:
    if v in df.columns: df[v]=enc(df[v])
df["idx_STEMI"] =(df["RQINDEX"]=="Acute myocardial infarction STEMI").astype(float)
df["idx_NSTEMI"]=(df["RQINDEX"]=="Acute myocardial infarction Non-STEMI").astype(float)
df["idx_UA"]    =(df["RQINDEX"]=="Unstable angina / Acute myocardial ischaemia").astype(float)
df["idx_ePCI"]  =(df["RQINDEX"]=="Elective percutaneous transluminal coronary angioplasty").astype(float)
for ai,fh in [("PR Interval (ms)","FHPRINTER"),("QRS Duration (ms)","FHQRSDURA"),("QT Interval (ms)","FHQTC")]:
    m=df[ai].isna()&df[fh].notna(); df.loc[m,ai]=df.loc[m,fh]
m=df["RBBB"].isna()
df.loc[m&(df["FHBUNBRBLO"]=="Right bundle branch block"),"RBBB"]=1
df.loc[m&df["FHBUNBRBLO"].notna()&(df["FHBUNBRBLO"]!="Right bundle branch block"),"RBBB"]=0
df.loc[m&(df["FHBUNBRBLO"]=="Left bundle branch block"),"LBBB"]=1
df.loc[m&df["FHBUNBRBLO"].notna()&(df["FHBUNBRBLO"]!="Left bundle branch block"),"LBBB"]=0
m2=df["1 AV Block"].isna()
df.loc[m2&(df["FHAVNOBL"]=="I°"),"1 AV Block"]=1
df.loc[m2&df["FHAVNOBL"].notna()&(df["FHAVNOBL"]!="I°"),"1 AV Block"]=0
mlvh=df["LVH"].isna(); flvh=enc(df["FHLEFTVENHYP"])
df.loc[mlvh&flvh.notna(),"LVH"]=flvh[mlvh&flvh.notna()]
for col,edges in BIN_EDGES.items():
    if col in df.columns: df[col]=binn(df[col],edges)

# ── Build full-period CVD composite endpoint ──────────────────────────────────
log("Building full-period CVD composite endpoint …")
intdate   = pd.to_datetime(df["INTDATE_excel2"], errors="coerce")
deathdate = pd.to_datetime(df["DEATHDATE_excel2"], errors="coerce")
df["days_to_death"] = (deathdate - intdate).dt.days

cvd_death  = df["DEATHCAUSE_excel2"].isin([1,2,3])
first_comp = df["first_component_Composite_4"].fillna("")
df["time_Composite_4"] = df["time_Composite_4"].clip(0, 365)

# 1-year CVD composite (same as run_G)
within1y_cvd = (first_comp=="SURVIVAL") & cvd_death
within1y_hosp= first_comp.isin(["HOSPAMI","HOSPHF","HOSPSTROKE"])
df["event_CVD4"] = (within1y_cvd | within1y_hosp).astype(int)
df["time_CVD4"]  = df["time_Composite_4"]

# 7 additional CVD deaths after 365 days (no prior hospitalization event)
# = CVD deaths after 365d that were NOT already hospitalization-first events
after365_new = cvd_death & (df["days_to_death"]>365) & df["days_to_death"].notna() \
               & ~within1y_hosp
log(f"  Additional CVD deaths after 365d (new events): {after365_new.sum()}")

df["event_full"] = df["event_CVD4"].copy()
df["time_full"]  = df["time_CVD4"].copy()
df.loc[after365_new, "event_full"] = 1
df.loc[after365_new, "time_full"]  = df.loc[after365_new, "days_to_death"]

log(f"  1-year CVD composite events:      {df['event_CVD4'].sum()}")
log(f"  Full-period CVD composite events: {df['event_full'].sum()}")
log(f"  Max follow-up time:               {df['time_full'].max():.0f} days")

# ── 5-fold Cox CV ─────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
folds = list(skf.split(df, df["event_full"]))

def cox_cv(data, vlist, folds, time_col, evt_col):
    avail = [v for v in vlist if v in data.columns]
    X = data[avail]; t = data[time_col].values; e = data[evt_col].values
    cs = []
    for tr, te in folds:
        imp = SimpleImputer(strategy="median"); sc = StandardScaler()
        Xtr = sc.fit_transform(imp.fit_transform(X.iloc[tr]))
        Xte = sc.transform(imp.transform(X.iloc[te]))
        cols = avail
        tdf = pd.DataFrame(Xtr, columns=cols)
        tdf["_t"] = t[tr]; tdf["_e"] = e[tr]
        cph = CoxPHFitter(penalizer=0.1)
        try:
            cph.fit(tdf, "_t", "_e", show_progress=False)
            risk = cph.predict_partial_hazard(pd.DataFrame(Xte, columns=cols)).values
            cs.append(concordance_index(t[te], -risk, e[te]))
        except: pass
    return (np.mean(cs) if cs else np.nan), (np.std(cs) if cs else np.nan)

log(f"\n── Cox CV: full-period vs 1-year CVD composite ──")
rows = []
for mname, vlist in MODELS:
    c_full, c_full_sd = cox_cv(df, vlist, folds, "time_full",  "event_full")
    c_1yr,  c_1yr_sd  = cox_cv(df, vlist, folds, "time_CVD4",  "event_CVD4")
    rows.append({
        "Model": mname,
        "Full-period C-stat": round(c_full,4),
        "Full-period C SD":   round(c_full_sd,4),
        "Full-period 95%CI":  f"{c_full-1.96*c_full_sd:.4f}–{c_full+1.96*c_full_sd:.4f}",
        "1-year C-stat":      round(c_1yr,4),
        "1-year C SD":        round(c_1yr_sd,4),
        "Delta C":            round(c_full-c_1yr,4),
    })
    log(f"  {mname:25s}  Full={c_full:.4f}±{c_full_sd:.4f}  "
        f"1yr={c_1yr:.4f}±{c_1yr_sd:.4f}  ΔC={c_full-c_1yr:+.4f}")

perf_df = pd.DataFrame(rows)
perf_df.to_csv(OUT_DIR/"fullperiod_cox.csv", index=False)
log("\nSaved → fullperiod_cox.csv")

# ── Plot: C-stat comparison ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8,5))
x = np.arange(len(rows))
w = 0.35
c_full_vals = [r["Full-period C-stat"] for r in rows]
c_1yr_vals  = [r["1-year C-stat"]      for r in rows]
c_full_sds  = [r["Full-period C SD"]   for r in rows]
c_1yr_sds   = [r["1-year C SD"]        for r in rows]
b1 = ax.bar(x-w/2, c_full_vals, w, yerr=c_full_sds, label="Full-period (174 events)",
            color="#E91E63", capsize=4, alpha=0.85)
b2 = ax.bar(x+w/2, c_1yr_vals,  w, yerr=c_1yr_sds,  label="1-year (167 events)",
            color="#90A4AE", capsize=4, alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels([r["Model"] for r in rows], rotation=10)
ax.set_ylabel("Harrell's C-statistic (5-fold CV)")
ax.set_title("Full-period vs 1-year CVD Composite — Cox Model Comparison\n"
             "(Full-period: +7 CVD deaths after 365 days)")
ax.set_ylim(0.55, 0.80); ax.legend()
ax.axhline(0.5, ls="--", color="gray", lw=0.8)
for b, v, s in zip(b1, c_full_vals, c_full_sds):
    ax.text(b.get_x()+b.get_width()/2, v+s+0.003, f"{v:.4f}",
            ha="center", va="bottom", fontsize=8)
for b, v, s in zip(b2, c_1yr_vals, c_1yr_sds):
    ax.text(b.get_x()+b.get_width()/2, v+s+0.003, f"{v:.4f}",
            ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR/"fullperiod_cstat.png", dpi=150, bbox_inches="tight")
plt.close()
log("Saved → fullperiod_cstat.png")
log("✓ Step H done.")

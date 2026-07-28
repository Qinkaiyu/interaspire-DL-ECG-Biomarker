#!/usr/bin/env python3
"""
Step R: Composite_6 model comparison — all-cause AND CVD endpoints.

Key finding from Steps N/O:
- LASSO/ElasticNet/Stability: 0 ECG variables selected (Best C=0.01269, very aggressive)
- RF: ECG variables show positive permutation importance
- Conclusion: ECG adds marginal incremental value over clinical vars for Composite_6
  because 135 procedural events (HOSPPCI/HOSPCABG) dominate and are not ECG-driven.

This script tests:
  Model 3 (baseline, no ECG)
  + RF top-5 for each endpoint
  + All-cause M4 ECG (QT prolongation, ST depression, QT interval) — for cross-comparison
  + CVD-Composite_4 consensus (QT prolongation, QT interval, LBBB, ST elevation)
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from pathlib import Path
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss
import statsmodels.api as sm
from scipy import stats
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = Path("data/INTERASPIRE_analysis_dataset.xlsx")
OUT_DIR   = Path("results/feature_selection")
RANDOM_STATE=42; N_FOLDS=5

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

BIN_EDGES={"PR Interval (ms)":[0,120,200,2000],"QRS Duration (ms)":[0,120,1000],
    "QT Interval (ms)":[0,440,480,5000],"FHPRINTER":[0,120,200,2000],
    "FHQRSDURA":[0,120,1000],"FHQTC":[0,440,480,5000],
    "CVARSYSTOL":[0,120,130,140,180,250],"CVARLABHBA1CPERCENT":[0,5.7,6.5,20],
    "CVARLABLDLFRIEDEWALD":[0,1.8,2.6,3.4,4.9,10],"CVARLABHDL":[0,1.0,1.3,1.55,5],
    "CVAREGFR":[0,15,30,45,60,90,200],"CVARBMI":[0,18.5,25,30,35,100],
    "CVARAGE":[0,50,60,70,80,150]}
M1_VARS=["CVARAGE","RQSEX","CVARCOSMOKING","CVARBMI","CVARSYSTOL","CVARLABHBA1CPERCENT",
         "CVAREGFR","CVARLABHDL","CVARLABLDLFRIEDEWALD"]
M2_VARS=M1_VARS+["RQHISTOFPRECVD","RQHISTOFCORARTDIS","RQHISTOFHEARTFAIL",
                  "any_af","CVARMEDBPLOWERING","CVARMEDLIPIDLOWERING",
                  "CVARMEDGLUCOSELOWERING","RQANTICOAGULANTS","RQHISTOFCOPD"]
M3_VARS=M2_VARS+["idx_STEMI","idx_NSTEMI","idx_UA","idx_ePCI"]
ECG_ALL=["Atrial Fibrillation","LBBB","RBBB","Q Wave","ST Elevation","ST Depression",
    "T Wave Inversion","Ischaemic","QT Prolongation","LVH","1 AV Block",
    "Left Axis Deviation","Right Axis Deviation","MI (Old)","MI(Acute)",
    "PR Interval (ms)","QRS Duration (ms)","QT Interval (ms)"]
ECG_BIN=ECG_ALL[:-3]

# Read RF importance for ECG vars (from Step O)
rf_c6   = pd.read_csv(OUT_DIR/"c6_rf_importance.csv")
rf_cvd6 = pd.read_csv(OUT_DIR/"cvd6_rf_importance.csv")
rf_c6_top5   = rf_c6.head(5)["raw_col"].tolist()
rf_cvd6_top5 = rf_cvd6.head(5)["raw_col"].tolist()

log("C6 all-cause RF top-5:  " + str(rf_c6_top5))
log("CVD-C6  RF top-5:       " + str(rf_cvd6_top5))

log("Loading data …")
df=pd.read_excel(DATA_PATH)
def enc(s): return s.map({"Yes":1,"No":0,"yes":1,"no":0})
def binn(s,e): return pd.cut(s,bins=e,labels=range(len(e)-1),include_lowest=True).astype(float)
df["RQSEX"]=df["RQSEX"].map({"Male":1,"Female":0})
for v in ["CVARCOSMOKING","RQHISTOFPRECVD","RQHISTOFCORARTDIS","RQHISTOFHEARTFAIL",
          "CVARMEDBPLOWERING","CVARMEDLIPIDLOWERING","CVARMEDGLUCOSELOWERING",
          "RQANTICOAGULANTS","RQHISTOFCOPD"]: df[v]=enc(df[v])
for v in ECG_BIN:
    if v in df.columns: df[v]=enc(df[v])
df["idx_STEMI"]=(df["RQINDEX"]=="Acute myocardial infarction STEMI").astype(float)
df["idx_NSTEMI"]=(df["RQINDEX"]=="Acute myocardial infarction Non-STEMI").astype(float)
df["idx_UA"]=(df["RQINDEX"]=="Unstable angina / Acute myocardial ischaemia").astype(float)
df["idx_ePCI"]=(df["RQINDEX"]=="Elective percutaneous transluminal coronary angioplasty").astype(float)
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

df["time_Composite_6"]=df["time_Composite_6"].clip(0,365)
cvd_death=df["DEATHCAUSE_excel2"].isin([1,2,3])
first6=df["first_component_Composite_6"].fillna("")
df["event_CVD6"]=((first6=="SURVIVAL")&cvd_death |
    first6.isin(["HOSPAMI","HOSPHF","HOSPSTROKE","HOSPPCI","HOSPCABG"])).astype(int)

# ── Landmark cohorts ────────────────────────────────────────────────────────────
complete6    =(df["time_Composite_6"]>=365)|(df["event_Composite_6"]==1)
complete6cvd =(df["time_Composite_6"]>=365)|(df["event_CVD6"]==1)
df6    =df[complete6   ].reset_index(drop=True)
df6cvd =df[complete6cvd].reset_index(drop=True)
log(f"Composite_6 all-cause: N={len(df6)},  events={int(df6['event_Composite_6'].sum())}")
log(f"CVD Composite_6:       N={len(df6cvd)}, events={int(df6cvd['event_CVD6'].sum())}")

# ── CV setup ───────────────────────────────────────────────────────────────────
skf=StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=RANDOM_STATE)
folds6    =list(skf.split(df6,    df6["event_Composite_6"]))
folds6cvd =list(skf.split(df6cvd, df6cvd["event_CVD6"]))

def prep(Xtr,Xte):
    imp=SimpleImputer(strategy="median"); sc=StandardScaler()
    return sc.fit_transform(imp.fit_transform(Xtr)),sc.transform(imp.transform(Xte)),list(Xtr.columns)

def lr_cv(data,vlist,folds,evt):
    avail=[v for v in vlist if v in data.columns]
    X=data[avail]; y=data[evt].values.astype(float)
    aucs,briers=[],[]
    for tr,te in folds:
        Xtr,Xte,_=prep(X.iloc[tr],X.iloc[te])
        m=LogisticRegression(penalty="l2",C=1.0,solver="lbfgs",max_iter=1000,random_state=RANDOM_STATE)
        m.fit(Xtr,y[tr]); p=m.predict_proba(Xte)[:,1]
        if len(np.unique(y[te]))>1: aucs.append(roc_auc_score(y[te],p))
        briers.append(brier_score_loss(y[te],p))
    return np.mean(aucs),np.std(aucs),np.mean(briers)

def cox_cv(data,vlist,folds,tc,ec):
    avail=[v for v in vlist if v in data.columns]
    X=data[avail]; t=data[tc].values; e=data[ec].values; cs=[]
    for tr,te in folds:
        Xtr,Xte,cols=prep(X.iloc[tr],X.iloc[te])
        tdf=pd.DataFrame(Xtr,columns=cols); tdf["_t"]=t[tr]; tdf["_e"]=e[tr]
        cph=CoxPHFitter(penalizer=0.1)
        try:
            cph.fit(tdf,"_t","_e",show_progress=False)
            risk=cph.predict_partial_hazard(pd.DataFrame(Xte,columns=cols)).values
            cs.append(concordance_index(t[te],-risk,e[te]))
        except: pass
    return (np.mean(cs) if cs else np.nan),(np.std(cs) if cs else np.nan)

# ── Model variants ─────────────────────────────────────────────────────────────
# ECG sets:
ECG_M4_ALLCAUSE = ["QT Prolongation","ST Depression","QT Interval (ms)"]   # all-cause M4
ECG_CVD4_CONS   = ["QT Prolongation","QT Interval (ms)","LBBB","ST Elevation"]  # CVD-4 consensus ≥3

VARIANTS_C6=[
    ("M1 (9 vars)",         M1_VARS),
    ("M2 (18 vars)",        M2_VARS),
    ("M3 (22 vars, no ECG)",M3_VARS),
    ("4a: RF-top5 (c6)",    M3_VARS+rf_c6_top5),
    ("4b: M4 ECG (3)",      M3_VARS+ECG_M4_ALLCAUSE),
    ("4c: CVD-4 cons (4)",  M3_VARS+ECG_CVD4_CONS),
]
VARIANTS_CVD6=[
    ("M1 (9 vars)",         M1_VARS),
    ("M2 (18 vars)",        M2_VARS),
    ("M3 (22 vars, no ECG)",M3_VARS),
    ("4a: RF-top5 (cvd6)",  M3_VARS+rf_cvd6_top5),
    ("4b: M4 ECG (3)",      M3_VARS+ECG_M4_ALLCAUSE),
    ("4c: CVD-4 cons (4)",  M3_VARS+ECG_CVD4_CONS),
]

def run_variants(variants, data, folds, tc, evt, label):
    log(f"\n── {label} ──")
    rows=[]
    base_auc=None; base_sd=None
    for mname,vlist in variants:
        ecg_n=len([v for v in vlist if v in ECG_ALL and v in data.columns])
        auc,auc_sd,brier=lr_cv(data,vlist,folds,evt)
        c,c_sd=cox_cv(data,vlist,folds,tc,evt)
        row={"Model":mname,"N ECG feats":ecg_n,
             "AUC mean":round(auc,4),"AUC SD":round(auc_sd,4),
             "AUC 95%CI":f"{auc-1.96*auc_sd:.4f}–{auc+1.96*auc_sd:.4f}",
             "Brier":round(brier,4),"Cox C-stat":round(c,4),"Cox C SD":round(c_sd,4)}
        if base_auc is None:
            base_auc=auc; base_sd=auc_sd
        else:
            d=auc-base_auc; se=np.sqrt(auc_sd**2+base_sd**2)
            z=d/(se+1e-10); p=2*(1-stats.norm.cdf(abs(z)))
            row["Delta AUC vs M1"]=round(d,4); row["p vs M1"]=round(p,4)
        rows.append(row)
        log(f"  {mname:30s} [{ecg_n:2d} ECG]  AUC={auc:.4f}±{auc_sd:.4f}  Cox-C={c:.4f}")
    return rows

rows_c6   =run_variants(VARIANTS_C6,   df6,    folds6,    "time_Composite_6","event_Composite_6","Composite_6 all-cause")
rows_cvd6 =run_variants(VARIANTS_CVD6, df6cvd, folds6cvd, "time_Composite_6","event_CVD6",       "CVD Composite_6")

pd.DataFrame(rows_c6  ).to_csv(OUT_DIR/"c6_model_variants.csv",  index=False)
pd.DataFrame(rows_cvd6).to_csv(OUT_DIR/"cvd6_model_variants.csv",index=False)
log("\nSaved → c6_model_variants.csv, cvd6_model_variants.csv")

# ── Plot: side-by-side AUC ─────────────────────────────────────────────────────
fig,axes=plt.subplots(1,2,figsize=(14,5))
COLORS=["#78909C","#546E7A","#607D8B","#E91E63","#2196F3","#4CAF50"]

for ax,rows,title in [(axes[0],rows_c6,"Composite_6 (all-cause, 328 events)"),
                       (axes[1],rows_cvd6,"CVD Composite_6 (300 events)")]:
    xpos=np.arange(len(rows))
    means=[r["AUC mean"] for r in rows]; sds=[r["AUC SD"] for r in rows]
    bars=ax.bar(xpos,means,yerr=sds,color=COLORS[:len(rows)],capsize=4,alpha=0.85)
    ax.axhline(rows[2]["AUC mean"],ls="--",color="#607D8B",lw=1.2,
               label=f"M3={rows[2]['AUC mean']:.3f}")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{r['Model']}\n[{r['N ECG feats']}]" for r in rows],
                       rotation=20,ha="right",fontsize=8)
    ax.set_ylabel("AUC (5-fold CV)"); ax.set_title(title)
    ax.set_ylim(0.55,0.80); ax.legend(fontsize=8)
    for bar,m,s in zip(bars,means,sds):
        ax.text(bar.get_x()+bar.get_width()/2,m+s+0.003,f"{m:.4f}",
                ha="center",va="bottom",fontsize=7)

plt.suptitle("Composite_6: Model Performance by Endpoint", fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR/"c6_auc_variants.png",dpi=150,bbox_inches="tight")
plt.close()
log("Saved → c6_auc_variants.png")

# ── Final summary ──────────────────────────────────────────────────────────────
print("\n"+"="*70)
print("COMPOSITE_6 — FEATURE SELECTION SUMMARY (Steps N–R)")
print("="*70)
print("\nKEY FINDING:")
print("  LASSO/ElasticNet/Stability selected 0 ECG variables (Best C=0.01269)")
print("  ECG signal present in RF but weak due to 135 procedural events")
print("  (HOSPPCI n=115, HOSPCABG n=20) dominating the endpoint")
print()
print("RF TOP-5 (all-cause C6): " + str(rf_c6_top5))
print("RF TOP-5 (CVD C6):       " + str(rf_cvd6_top5))
print()
print("MODEL PERFORMANCE (all-cause Composite_6):")
for r in rows_c6:
    d=r.get("Delta AUC vs M1",""); ds=f"  ΔvsM1={d:+.4f}" if isinstance(d,float) else ""
    print(f"  {r['Model']:30s} [{r['N ECG feats']:2d}]  AUC={r['AUC mean']:.4f}±{r['AUC SD']:.4f}  Cox-C={r['Cox C-stat']}{ds}")
print()
print("MODEL PERFORMANCE (CVD Composite_6):")
for r in rows_cvd6:
    d=r.get("Delta AUC vs M1",""); ds=f"  ΔvsM1={d:+.4f}" if isinstance(d,float) else ""
    print(f"  {r['Model']:30s} [{r['N ECG feats']:2d}]  AUC={r['AUC mean']:.4f}±{r['AUC SD']:.4f}  Cox-C={r['Cox C-stat']}{ds}")
print()
print("REFERENCE (all-cause Composite_4 / CVD Composite_4):")
print("  All-cause M3:  AUC=0.6933  Cox-C=0.6962")
print("  All-cause M4:  AUC=0.6952  Cox-C=0.7021  [QT prol, ST dep, QT int]")
print("  CVD-4 M3:      AUC=0.6701  Cox-C=0.6905")
print("  CVD-4 M4:      AUC=0.6836  Cox-C=0.6959  [QT prol, ST dep, QT int]")
print("="*70)
log("✓ Step R done.")

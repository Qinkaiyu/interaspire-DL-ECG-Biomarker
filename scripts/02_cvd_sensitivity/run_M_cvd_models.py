#!/usr/bin/env python3
"""Step M: CVD composite model comparison — sensitivity analysis."""
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

DATA_PATH = Path("data/INTERASPIRE_analysis_dataset.xlsx")
OUT_DIR   = Path("results/feature_selection")
RANDOM_STATE=42; N_FOLDS=5

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

lasso = pd.read_csv(OUT_DIR/"cvd_lasso_ecg_coefs.csv")
enet  = pd.read_csv(OUT_DIR/"cvd_enet_ecg_coefs.csv")
rf    = pd.read_csv(OUT_DIR/"cvd_rf_importance.csv")
stab  = pd.read_csv(OUT_DIR/"cvd_stability_selection.csv")

lasso_vars = lasso[lasso["Selected"]==True]["raw_col"].tolist()
enet_vars  = enet[enet["Selected"]==True]["raw_col"].tolist()
rf_top5    = rf.head(5)["raw_col"].tolist()
stab_vars  = stab[stab["Selection freq %"]>=50]["raw_col"].tolist()

all_ecg = lasso["raw_col"].tolist()
method_sets=[set(lasso_vars),set(enet_vars),set(rf_top5),set(stab_vars)]
votes={v:sum(1 for s in method_sets if v in s) for v in all_ecg}
consensus_3=[v for v,n in votes.items() if n>=3]
consensus_2=[v for v,n in votes.items() if n>=2]

log("CVD composite selection summary:")
log(f"  LASSO ({len(lasso_vars)}):      {lasso_vars}")
log(f"  ElasticNet ({len(enet_vars)}):  {enet_vars}")
log(f"  RF top-5:           {rf_top5}")
log(f"  Stable ≥50% ({len(stab_vars)}): {stab_vars}")
log(f"  Consensus ≥3 ({len(consensus_3)}): {consensus_3}")

DISPLAY_ECG={"Atrial Fibrillation":"AF (ECG)","LBBB":"LBBB","RBBB":"RBBB","Q Wave":"Q wave",
    "ST Elevation":"ST elevation","ST Depression":"ST depression",
    "T Wave Inversion":"T-wave inversion","Ischaemic":"Ischaemia",
    "QT Prolongation":"QT prolongation (flag)","LVH":"LVH","1 AV Block":"1st AV block",
    "Left Axis Deviation":"Left axis deviation","Right Axis Deviation":"Right axis deviation",
    "MI (Old)":"Old MI","MI(Acute)":"Acute MI (ECG)",
    "PR Interval (ms)":"PR interval (binned)","QRS Duration (ms)":"QRS duration (binned)",
    "QT Interval (ms)":"QT interval (binned)"}

BIN_EDGES={"PR Interval (ms)":[0,120,200,2000],"QRS Duration (ms)":[0,120,1000],
    "QT Interval (ms)":[0,440,480,5000],"FHPRINTER":[0,120,200,2000],
    "FHQRSDURA":[0,120,1000],"FHQTC":[0,440,480,5000],
    "CVARSYSTOL":[0,120,130,140,180,250],"CVARLABHBA1CPERCENT":[0,5.7,6.5,20],
    "CVARLABLDLFRIEDEWALD":[0,1.8,2.6,3.4,4.9,10],"CVARLABHDL":[0,1.0,1.3,1.55,5],
    "CVAREGFR":[0,15,30,45,60,90,200],"CVARBMI":[0,18.5,25,30,35,100],
    "CVARAGE":[0,50,60,70,80,150]}
M3_VARS=["CVARAGE","RQSEX","CVARCOSMOKING","CVARBMI","CVARSYSTOL","CVARLABHBA1CPERCENT",
    "CVAREGFR","CVARLABHDL","CVARLABLDLFRIEDEWALD","RQHISTOFPRECVD","RQHISTOFCORARTDIS",
    "RQHISTOFHEARTFAIL","any_af","CVARMEDBPLOWERING","CVARMEDLIPIDLOWERING",
    "CVARMEDGLUCOSELOWERING","RQANTICOAGULANTS","RQHISTOFCOPD",
    "idx_STEMI","idx_NSTEMI","idx_UA","idx_ePCI"]
ECG_ALL=["Atrial Fibrillation","LBBB","RBBB","Q Wave","ST Elevation","ST Depression",
    "T Wave Inversion","Ischaemic","QT Prolongation","LVH","1 AV Block",
    "Left Axis Deviation","Right Axis Deviation","MI (Old)","MI(Acute)",
    "PR Interval (ms)","QRS Duration (ms)","QT Interval (ms)"]
ECG_BIN=ECG_ALL[:-3]

log("Loading data ...")
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

cvd_death=df["DEATHCAUSE_excel2"].isin([1,2,3])
first_comp=df["first_component_Composite_4"].fillna("")
df["time_Composite_4"]=df["time_Composite_4"].clip(0,365)
df["event_CVD4"]=((first_comp=="SURVIVAL")&cvd_death |
                   first_comp.isin(["HOSPAMI","HOSPHF","HOSPSTROKE"])).astype(int)
complete=(df["time_Composite_4"]>=365)|(df["event_CVD4"]==1)
df_lr=df[complete].reset_index(drop=True)
df_cox=df.copy()
y_lr=df_lr["event_CVD4"].values.astype(float)
log(f"  CVD LR cohort: {len(df_lr)}, events: {int(y_lr.sum())}")

skf=StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=RANDOM_STATE)
folds_lr=list(skf.split(df_lr,df_lr["event_CVD4"]))
folds_cx=list(skf.split(df_cox,df_cox["event_CVD4"]))

def prep(Xtr,Xte):
    imp=SimpleImputer(strategy="median"); sc=StandardScaler()
    return sc.fit_transform(imp.fit_transform(Xtr)),sc.transform(imp.transform(Xte)),list(Xtr.columns)

def lr_cv(data,vlist,folds,evt):
    avail=[v for v in vlist if v in data.columns]
    X=data[avail]; y=data[evt].values.astype(float)
    aucs,briers,oof=[],[],np.full(len(data),np.nan)
    for tr,te in folds:
        Xtr,Xte,_=prep(X.iloc[tr],X.iloc[te])
        m=LogisticRegression(penalty="l2",C=1.0,solver="lbfgs",max_iter=1000,random_state=RANDOM_STATE)
        m.fit(Xtr,y[tr]); p=m.predict_proba(Xte)[:,1]; oof[te]=p
        if len(np.unique(y[te]))>1: aucs.append(roc_auc_score(y[te],p))
        briers.append(brier_score_loss(y[te],p))
    valid=~np.isnan(oof)
    try:
        logit=np.log(oof[valid]/(1-oof[valid]+1e-9)+1e-9)
        cm=sm.Logit(y[valid],sm.add_constant(logit)).fit(disp=False); cal=cm.params[1]
    except: cal=np.nan
    return np.mean(aucs),np.std(aucs),np.mean(briers),cal,oof,y

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

VARIANTS=[
    ("Model 3 (no ECG)",     M3_VARS),
    ("4a: Consensus ≥3",     M3_VARS+consensus_3),
    ("4b: Stable ≥50% (3)",  M3_VARS+stab_vars),
    ("4c: LASSO (7)",        M3_VARS+lasso_vars),
    ("4d: ElasticNet (6)",   M3_VARS+enet_vars),
    ("4e: RF top-5",         M3_VARS+rf_top5),
    ("4f: All-cause M4 (3)", M3_VARS+["QT Prolongation","ST Depression","QT Interval (ms)"]),
]

log(f"\n── CVD composite: {len(VARIANTS)} model variants ──")
perf_rows=[]; oof_store={}
for mname,vlist in VARIANTS:
    ecg_n=len([v for v in vlist if v in ECG_ALL and v in df_lr.columns])
    auc,auc_sd,brier,cal,oof,ytrue=lr_cv(df_lr,vlist,folds_lr,"event_CVD4")
    c,c_sd=cox_cv(df_cox,vlist,folds_cx,"time_Composite_4","event_CVD4")
    perf_rows.append({"Model":mname,"N ECG feats":ecg_n,
        "AUC mean":round(auc,4),"AUC SD":round(auc_sd,4),
        "AUC 95%CI":f"{auc-1.96*auc_sd:.4f}–{auc+1.96*auc_sd:.4f}",
        "Brier":round(brier,4),"Cal slope":round(cal,3) if not np.isnan(cal) else np.nan,
        "Cox C-stat":round(c,4),"Cox C SD":round(c_sd,4)})
    oof_store[mname]=(oof,ytrue)
    log(f"  {mname:28s} [{ecg_n:2d} ECG]  AUC={auc:.4f}±{auc_sd:.4f}  Cal={cal:.3f}  Cox-C={c:.4f}")

base_auc=perf_rows[0]["AUC mean"]; base_sd=perf_rows[0]["AUC SD"]
for row in perf_rows[1:]:
    d=row["AUC mean"]-base_auc
    se=np.sqrt(row["AUC SD"]**2+base_sd**2)
    z=d/(se+1e-10); p=2*(1-stats.norm.cdf(abs(z)))
    row["Delta AUC vs M3"]=round(d,4); row["p vs M3"]=round(p,4)

pd.DataFrame(perf_rows).to_csv(OUT_DIR/"cvd_model_variants.csv",index=False)
log("\nSaved → cvd_model_variants.csv")

# Consensus table
sel_rows=[]
for v in all_ecg:
    ms=[]
    if v in lasso_vars: ms.append("LASSO")
    if v in enet_vars:  ms.append("ElasticNet")
    if v in rf_top5:    ms.append("RF-top5")
    if v in stab_vars:  ms.append("Stability")
    sel_rows.append({"Variable":DISPLAY_ECG.get(v,v),"raw_col":v,
                     "Votes(4)":len(ms),"Methods":"|".join(ms)})
sel_df=pd.DataFrame(sel_rows).sort_values("Votes(4)",ascending=False)
sel_df.to_csv(OUT_DIR/"cvd_feature_summary.csv",index=False)

# Plot: AUC comparison
COLORS=["#607D8B","#E91E63","#9C27B0","#2196F3","#4CAF50","#FF9800","#F44336"]
fig,ax=plt.subplots(figsize=(11,5))
xpos=np.arange(len(perf_rows))
means=[r["AUC mean"] for r in perf_rows]; sds=[r["AUC SD"] for r in perf_rows]
bars=ax.bar(xpos,means,yerr=sds,color=COLORS[:len(perf_rows)],capsize=4,alpha=0.85)
ax.axhline(base_auc,ls="--",color="#607D8B",lw=1.2,label=f"Model 3={base_auc:.3f}")
ax.set_xticks(xpos)
ax.set_xticklabels([f"{r['Model']}\n[{r['N ECG feats']} ECG]" for r in perf_rows],
                   rotation=20,ha="right",fontsize=8)
ax.set_ylabel("AUC (5-fold CV)")
ax.set_title("CVD Composite Endpoint — ECG Feature Selection (Sensitivity Analysis)")
ax.set_ylim(0.55,0.78); ax.legend(fontsize=9)
for bar,m,s in zip(bars,means,sds):
    ax.text(bar.get_x()+bar.get_width()/2,m+s+0.003,f"{m:.4f}",
            ha="center",va="bottom",fontsize=7.5)
plt.tight_layout()
plt.savefig(OUT_DIR/"cvd_auc_variants.png",dpi=150,bbox_inches="tight")
plt.close()

# Summary print
print("\n"+"="*70)
print("CVD COMPOSITE — FEATURE SELECTION SUMMARY (Steps I–M)")
print("="*70)
print("\nVOTES (4 methods: LASSO, ElasticNet, RF-top5, Stability ≥50%):")
for _,r in sel_df.iterrows():
    if r["Votes(4)"]>0:
        print(f"  [{r['Votes(4)']}/4] {r['Variable']:30s}  {r['Methods']}")
print("\nMODEL VARIANTS (CVD composite, 5-fold CV):")
for row in perf_rows:
    d=row.get("Delta AUC vs M3",""); p=row.get("p vs M3","")
    ds=f"  ΔAUC={d:+.4f}" if isinstance(d,float) else ""
    print(f"  {row['Model']:28s} [{row['N ECG feats']:2d} ECG]  "
          f"AUC={row['AUC mean']:.4f}±{row['AUC SD']:.4f}  Cox-C={row['Cox C-stat']}{ds}")
print("\nCOMPARISON WITH ALL-CAUSE COMPOSITE:")
print("  All-cause M3:  AUC=0.6933  Cox-C=0.6962")
print("  All-cause M4e: AUC=0.6952  Cox-C=0.7021  [3 ECG: QT prol, ST dep, QT int]")
print("="*70)
log("✓ Step M done. Sensitivity analysis complete.")

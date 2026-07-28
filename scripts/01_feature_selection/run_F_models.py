#!/usr/bin/env python3
"""Step F: Candidate model comparison + all plots. Reads selection CSVs from prior steps."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from pathlib import Path
import time, textwrap

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve
import statsmodels.api as sm
from scipy import stats
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA_PATH = Path("data/INTERASPIRE_analysis_dataset.xlsx")
OUT_DIR   = Path("results/feature_selection")

RANDOM_STATE = 42
N_FOLDS = 5
PRIMARY_EVT  = "event_Composite_4"
PRIMARY_TIME = "time_Composite_4"

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# ── Read selection results ────────────────────────────────────────────────────
lasso_sel  = pd.read_csv(OUT_DIR/"lasso_ecg_coefs.csv")
enet_sel   = pd.read_csv(OUT_DIR/"enet_ecg_coefs.csv")
rf_imp     = pd.read_csv(OUT_DIR/"rf_importance.csv")
stab_sel   = pd.read_csv(OUT_DIR/"stability_selection.csv")
univ_sel   = pd.read_csv(OUT_DIR/"ecg_univariate_adjusted.csv")

lasso_vars = lasso_sel[lasso_sel["Selected"]==True]["raw_col"].tolist()
enet_vars  = enet_sel[enet_sel["Selected"]==True]["raw_col"].tolist()
rf_top5    = rf_imp.head(5)["raw_col"].tolist()
stab_vars  = stab_sel[stab_sel["Selection freq %"]>=50]["raw_col"].tolist()
p05_vars   = univ_sel[univ_sel["p-value (adj)"]<0.05]["raw_col"].tolist()
p10_vars   = univ_sel[univ_sel["p-value (adj)"]<0.10]["raw_col"].tolist()

# Consensus: features selected by ≥3 of 5 methods
all_ecg = lasso_sel["raw_col"].tolist()
method_sets = [set(lasso_vars), set(enet_vars), set(rf_top5), set(stab_vars), set(p05_vars)]
votes = {v: sum(1 for s in method_sets if v in s) for v in all_ecg}
consensus_3 = [v for v, n in votes.items() if n >= 3]
consensus_2 = [v for v, n in votes.items() if n >= 2]

log("Selection summary:")
log(f"  LASSO ({len(lasso_vars)}):        {lasso_vars}")
log(f"  ElasticNet ({len(enet_vars)}):    {enet_vars}")
log(f"  RF top-5:             {rf_top5}")
log(f"  Stable ≥50% ({len(stab_vars)}):  {stab_vars}")
log(f"  adj p<0.05 ({len(p05_vars)}):     {p05_vars}")
log(f"  Consensus ≥3 ({len(consensus_3)}): {consensus_3}")
log(f"  Consensus ≥2 ({len(consensus_2)}): {consensus_2}")

DISPLAY_ECG = {
    "Atrial Fibrillation":"AF (ECG)","LBBB":"LBBB","RBBB":"RBBB","Q Wave":"Q wave",
    "ST Elevation":"ST elevation","ST Depression":"ST depression",
    "T Wave Inversion":"T-wave inversion","Ischaemic":"Ischaemia",
    "QT Prolongation":"QT prolongation (flag)","LVH":"LVH","1 AV Block":"1st AV block",
    "Left Axis Deviation":"Left axis deviation","Right Axis Deviation":"Right axis deviation",
    "MI (Old)":"Old MI","MI(Acute)":"Acute MI (ECG)",
    "PR Interval (ms)":"PR interval (binned)","QRS Duration (ms)":"QRS duration (binned)",
    "QT Interval (ms)":"QT interval (binned)",
}

# ── Preprocessing (same as other scripts) ────────────────────────────────────
BIN_EDGES = {
    "PR Interval (ms)":[0,120,200,2000],"QRS Duration (ms)":[0,120,1000],
    "QT Interval (ms)":[0,440,480,5000],"FHPRINTER":[0,120,200,2000],
    "FHQRSDURA":[0,120,1000],"FHQTC":[0,440,480,5000],
    "CVARSYSTOL":[0,120,130,140,180,250],"CVARLABHBA1CPERCENT":[0,5.7,6.5,20],
    "CVARLABLDLFRIEDEWALD":[0,1.8,2.6,3.4,4.9,10],"CVARLABHDL":[0,1.0,1.3,1.55,5],
    "CVAREGFR":[0,15,30,45,60,90,200],"CVARBMI":[0,18.5,25,30,35,100],
    "CVARAGE":[0,50,60,70,80,150],
}
M3_VARS = [
    "CVARAGE","RQSEX","CVARCOSMOKING","CVARBMI","CVARSYSTOL","CVARLABHBA1CPERCENT",
    "CVAREGFR","CVARLABHDL","CVARLABLDLFRIEDEWALD","RQHISTOFPRECVD","RQHISTOFCORARTDIS",
    "RQHISTOFHEARTFAIL","any_af","CVARMEDBPLOWERING","CVARMEDLIPIDLOWERING",
    "CVARMEDGLUCOSELOWERING","RQANTICOAGULANTS","RQHISTOFCOPD",
    "idx_STEMI","idx_NSTEMI","idx_UA","idx_ePCI",
]
ECG_ALL = [
    "Atrial Fibrillation","LBBB","RBBB","Q Wave","ST Elevation","ST Depression",
    "T Wave Inversion","Ischaemic","QT Prolongation","LVH","1 AV Block",
    "Left Axis Deviation","Right Axis Deviation","MI (Old)","MI(Acute)",
    "PR Interval (ms)","QRS Duration (ms)","QT Interval (ms)",
]

log("Loading data …")
df = pd.read_excel(DATA_PATH)
def enc(s): return s.map({"Yes":1,"No":0,"yes":1,"no":0})
def binn(s,e): return pd.cut(s,bins=e,labels=range(len(e)-1),include_lowest=True).astype(float)

df["RQSEX"]=df["RQSEX"].map({"Male":1,"Female":0})
for v in ["CVARCOSMOKING","RQHISTOFPRECVD","RQHISTOFCORARTDIS","RQHISTOFHEARTFAIL",
          "CVARMEDBPLOWERING","CVARMEDLIPIDLOWERING","CVARMEDGLUCOSELOWERING",
          "RQANTICOAGULANTS","RQHISTOFCOPD"]:
    df[v]=enc(df[v])
for v in ECG_ALL[:-3]:
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

df[PRIMARY_TIME]=df[PRIMARY_TIME].clip(0,365)
complete=(df[PRIMARY_TIME]>=365)|(df[PRIMARY_EVT]==1)
df_lr=df[complete].reset_index(drop=True)
df_cox=df.copy()
y_lr=df_lr[PRIMARY_EVT].values.astype(float)
log(f"  LR cohort: {len(df_lr)}, events: {int(y_lr.sum())}")

skf=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
folds_lr =list(skf.split(df_lr,  df_lr[PRIMARY_EVT]))
folds_cox=list(skf.split(df_cox, df_cox[PRIMARY_EVT]))

def prep(Xtr,Xte):
    imp=SimpleImputer(strategy="median"); sc=StandardScaler()
    return sc.fit_transform(imp.fit_transform(Xtr)), sc.transform(imp.transform(Xte)), list(Xtr.columns)

def lr_cv(data, vlist, folds):
    avail=[v for v in vlist if v in data.columns]
    X=data[avail]; y=data[PRIMARY_EVT].values.astype(float)
    aucs,briers=[],[]
    oof=np.full(len(data),np.nan)
    for tr,te in folds:
        Xtr,Xte,_=prep(X.iloc[tr],X.iloc[te])
        m=LogisticRegression(penalty="l2",C=1.0,solver="lbfgs",max_iter=1000,random_state=RANDOM_STATE)
        m.fit(Xtr,y[tr]); p=m.predict_proba(Xte)[:,1]
        oof[te]=p
        if len(np.unique(y[te]))>1: aucs.append(roc_auc_score(y[te],p))
        briers.append(brier_score_loss(y[te],p))
    # calibration slope
    valid=~np.isnan(oof)
    try:
        logit=np.log(oof[valid]/(1-oof[valid]+1e-9)+1e-9)
        cm=sm.Logit(y[valid],sm.add_constant(logit)).fit(disp=False)
        cal=cm.params[1]
    except: cal=np.nan
    return np.mean(aucs),np.std(aucs),np.mean(briers),cal,oof,y

def cox_cv(data, vlist, folds):
    avail=[v for v in vlist if v in data.columns]
    X=data[avail]; t=data[PRIMARY_TIME].values; e=data[PRIMARY_EVT].values
    cs=[]
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

# ── Define model variants ─────────────────────────────────────────────────────
MODEL_VARIANTS = [
    ("Model 3 (no ECG)",       M3_VARS,                     "baseline"),
    ("Model 4 full (18 ECG)",  M3_VARS+ECG_ALL,             "full"),
    ("4a: Consensus ≥3",       M3_VARS+consensus_3,         "consensus3"),
    ("4b: Consensus ≥2",       M3_VARS+consensus_2,         "consensus2"),
    ("4c: LASSO (7)",          M3_VARS+lasso_vars,          "lasso"),
    ("4d: ElasticNet (6)",     M3_VARS+enet_vars,           "enet"),
    ("4e: Stable ≥50% (3)",   M3_VARS+stab_vars,           "stable"),
    ("4f: adj p<0.05 (3)",     M3_VARS+p05_vars,            "p05"),
    ("4g: RF top-5",           M3_VARS+rf_top5,             "rf5"),
]

log(f"\n── Running 5-fold CV for {len(MODEL_VARIANTS)} model variants ──")
perf_rows=[]; oof_store={}
for mname, vlist, tag in MODEL_VARIANTS:
    ecg_n=len([v for v in vlist if v in ECG_ALL and v in df_lr.columns])
    auc,auc_sd,brier,cal,oof,ytrue=lr_cv(df_lr,vlist,folds_lr)
    c,c_sd=cox_cv(df_cox,vlist,folds_cox)
    perf_rows.append({
        "Model":mname, "N ECG feats":ecg_n,
        "AUC mean":round(auc,4),"AUC SD":round(auc_sd,4),
        "AUC 95%CI":f"{auc-1.96*auc_sd:.4f}–{auc+1.96*auc_sd:.4f}",
        "Brier":round(brier,4),"Cal slope":round(cal,3) if not np.isnan(cal) else np.nan,
        "Cox C-stat":round(c,4) if not np.isnan(c) else np.nan,
        "Cox C SD":round(c_sd,4),
        "ECG features":str([DISPLAY_ECG.get(v,v) for v in vlist if v in ECG_ALL]),
    })
    oof_store[mname]=(oof,ytrue)
    log(f"  {mname:30s} [{ecg_n:2d} ECG]  AUC={auc:.4f}±{auc_sd:.4f}  "
        f"Cal={cal:.3f}  Cox-C={c:.4f}")

# Delta AUC vs Model 3
base_auc=perf_rows[0]["AUC mean"]; base_sd=perf_rows[0]["AUC SD"]
for row in perf_rows[1:]:
    d=row["AUC mean"]-base_auc
    se=np.sqrt(row["AUC SD"]**2+base_sd**2)
    z=d/(se+1e-10); p=2*(1-stats.norm.cdf(abs(z)))
    row["Delta AUC vs M3"]=round(d,4)
    row["p vs M3"]=round(p,4)

perf_df=pd.DataFrame(perf_rows)
perf_df.to_csv(OUT_DIR/"model_performance.csv",index=False)
log("\n  Saved model_performance.csv")

# ── NRI/IDI vs Model 3 ────────────────────────────────────────────────────────
oof_m3,y_m3=oof_store["Model 3 (no ECG)"]
nri_rows=[]
for mname,_,_ in MODEL_VARIANTS[1:]:
    oof_new,ytrue=oof_store[mname]
    valid=~np.isnan(oof_new)&~np.isnan(oof_m3)
    if valid.sum()<50: continue
    y2=ytrue[valid].astype(bool); pn=oof_new[valid]; po=oof_m3[valid]
    dp=pn-po
    idi=float(np.mean(dp[y2])-np.mean(dp[~y2]))
    nri=float((np.mean(dp[y2]>0)-np.mean(dp[y2]<0))+(np.mean(dp[~y2]<0)-np.mean(dp[~y2]>0)))
    nri_rows.append({"Model":mname,"NRI (cont)":round(nri,4),"IDI":round(idi,4)})
    log(f"  NRI/IDI {mname:30s}  NRI={nri:+.4f}  IDI={idi:+.4f}")
pd.DataFrame(nri_rows).to_csv(OUT_DIR/"nri_idi.csv",index=False)

# ── Feature selection consensus table ────────────────────────────────────────
sel_rows=[]
for v in all_ecg:
    methods_sel=[]
    if v in lasso_vars: methods_sel.append("LASSO")
    if v in enet_vars:  methods_sel.append("ElasticNet")
    if v in rf_top5:    methods_sel.append("RF-top5")
    if v in stab_vars:  methods_sel.append("Stability")
    if v in p05_vars:   methods_sel.append("Univ p<.05")
    sel_rows.append({
        "Variable":DISPLAY_ECG.get(v,v),"raw_col":v,
        "Votes (of 5)":len(methods_sel),"Methods":"|".join(methods_sel),
        "Univ OR":univ_sel.set_index("raw_col").loc[v,"Adjusted OR"] if v in univ_sel["raw_col"].values else np.nan,
        "Univ p":univ_sel.set_index("raw_col").loc[v,"p-value (adj)"] if v in univ_sel["raw_col"].values else np.nan,
    })
sel_df=pd.DataFrame(sel_rows).sort_values("Votes (of 5)",ascending=False)
sel_df.to_csv(OUT_DIR/"feature_selection_summary.csv",index=False)
log("  Saved feature_selection_summary.csv")

# ── PLOTS ─────────────────────────────────────────────────────────────────────
log("Generating plots …")
COLORS=["#607D8B","#9E9E9E","#E91E63","#2196F3","#4CAF50","#FF9800","#9C27B0","#F44336","#00BCD4"]

# Plot 1: AUC comparison bar chart
fig,ax=plt.subplots(figsize=(12,5))
xpos=np.arange(len(perf_rows))
means=[r["AUC mean"] for r in perf_rows]
sds  =[r["AUC SD"]   for r in perf_rows]
bars=ax.bar(xpos,means,yerr=sds,color=COLORS[:len(perf_rows)],capsize=4,alpha=0.85)
ax.axhline(base_auc,ls="--",color="#607D8B",lw=1.2,label=f"Model 3 = {base_auc:.3f}")
ax.set_xticks(xpos)
xlabels=[f"{r['Model']}\n[{r['N ECG feats']} ECG]" for r in perf_rows]
ax.set_xticklabels(xlabels,rotation=20,ha="right",fontsize=8)
ax.set_ylabel("AUC (5-fold CV)")
ax.set_title("ECG Feature Selection — Model Discrimination Comparison\n(Primary endpoint: 1-year composite)")
ax.set_ylim(0.55,0.78); ax.legend(fontsize=9)
for bar,m,s in zip(bars,means,sds):
    ax.text(bar.get_x()+bar.get_width()/2, m+s+0.003, f"{m:.4f}",
            ha="center",va="bottom",fontsize=7.5)
plt.tight_layout()
plt.savefig(OUT_DIR/"auc_comparison.png",dpi=150,bbox_inches="tight")
plt.close()

# Plot 2: ROC curves (Model 3 + best parsim variants)
key_models=["Model 3 (no ECG)","4c: LASSO (7)","4e: Stable ≥50% (3)","4f: adj p<0.05 (3)","4g: RF top-5"]
fig,ax=plt.subplots(figsize=(7,7))
ax.plot([0,1],[0,1],"k--",lw=0.8,label="Chance")
for mname,color in zip(key_models,["#607D8B","#2196F3","#E91E63","#4CAF50","#FF9800"]):
    if mname not in oof_store: continue
    oof_arr,ytrue=oof_store[mname]
    valid=~np.isnan(oof_arr)
    fpr,tpr,_=roc_curve(ytrue[valid],oof_arr[valid])
    auc=roc_auc_score(ytrue[valid],oof_arr[valid])
    ecg_n=next(r["N ECG feats"] for r in perf_rows if r["Model"]==mname)
    lw=2.5 if "Model 3" in mname else 1.8
    ax.plot(fpr,tpr,color=color,lw=lw,label=f"{mname} [n={ecg_n}]  AUC={auc:.4f}")
ax.set_xlabel("1–Specificity"); ax.set_ylabel("Sensitivity")
ax.set_title("ROC Curves — Key Model Variants (5-fold CV OOF)")
ax.legend(loc="lower right",fontsize=8.5); ax.set_xlim(0,1); ax.set_ylim(0,1)
plt.tight_layout()
plt.savefig(OUT_DIR/"roc_curves.png",dpi=150,bbox_inches="tight")
plt.close()

# Plot 3: Feature selection heatmap
method_names=["LASSO","ElasticNet","RF-top5","Stability ≥50%","Univ p<0.05"]
heat=np.array([
    [1 if v in lasso_vars else 0,
     1 if v in enet_vars  else 0,
     1 if v in rf_top5    else 0,
     1 if v in stab_vars  else 0,
     1 if v in p05_vars   else 0]
    for v in all_ecg
])
vote_order=np.argsort(-heat.sum(axis=1))
heat=heat[vote_order]
ylabels=[DISPLAY_ECG.get(all_ecg[i],all_ecg[i]) for i in vote_order]
vote_counts=heat.sum(axis=1)

fig,ax=plt.subplots(figsize=(9,7))
ax.imshow(heat,cmap="Blues",aspect="auto",vmin=0,vmax=1)
ax.set_xticks(range(5)); ax.set_xticklabels(method_names,rotation=30,ha="right",fontsize=9)
ax.set_yticks(range(len(ylabels)))
ax.set_yticklabels([f"{l}  [{v}/5]" for l,v in zip(ylabels,vote_counts)],fontsize=9)
for i in range(len(ylabels)):
    for j in range(5):
        ax.text(j,i,"✓" if heat[i,j] else "·",ha="center",va="center",
                fontsize=12,color="white" if heat[i,j] else "#cccccc")
ax.set_title("ECG Biomarker Feature Selection — Method Agreement",fontsize=11,pad=10)
plt.tight_layout()
plt.savefig(OUT_DIR/"feature_selection_heatmap.png",dpi=150,bbox_inches="tight")
plt.close()

# Plot 4: Adjusted OR forest plot
fig,ax=plt.subplots(figsize=(8,7))
udf=univ_sel.sort_values("Adjusted OR",ascending=True)
ors=udf["Adjusted OR"].values; lo=udf["OR 95%CI lo"].values
hi=udf["OR 95%CI hi"].values; pvs=udf["p-value (adj)"].values
lbls=udf["ECG variable"].values
bar_c=["#F44336" if p<0.05 else ("#FF9800" if p<0.10 else "#90A4AE") for p in pvs]
ax.barh(range(len(udf)),ors-1,left=1,height=0.6,color=bar_c,alpha=0.8)
ax.errorbar(ors,range(len(udf)),xerr=[ors-lo,hi-ors],fmt="none",color="black",capsize=3,lw=1.2)
ax.axvline(1,ls="--",color="black",lw=0.8)
ax.set_yticks(range(len(udf))); ax.set_yticklabels(lbls,fontsize=9)
ax.set_xlabel("Adjusted Odds Ratio (95% CI)",fontsize=10)
ax.set_title("ECG Biomarkers: Adjusted Association with 1-year Composite Outcome\n(each ECG var added to Model 3)",fontsize=10)
ax.legend(handles=[mpatches.Patch(color="#F44336",label="adj p<0.05"),
                   mpatches.Patch(color="#FF9800",label="adj p<0.10"),
                   mpatches.Patch(color="#90A4AE",label="p≥0.10")],fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR/"ecg_adjusted_OR.png",dpi=150,bbox_inches="tight")
plt.close()

# Plot 5: Stability selection bar
fig,ax=plt.subplots(figsize=(8,6))
ss=stab_sel.sort_values("Selection freq %",ascending=True)
bc=["#F44336" if v>=50 else "#90A4AE" for v in ss["Selection freq %"]]
ax.barh(range(len(ss)),ss["Selection freq %"].values,color=bc,alpha=0.85)
ax.axvline(50,ls="--",color="black",lw=1,label="50% threshold")
ax.set_yticks(range(len(ss))); ax.set_yticklabels(ss["Variable"].values,fontsize=9)
ax.set_xlabel("Bootstrap selection frequency (%)"); ax.legend(fontsize=9)
ax.set_title("ECG Feature Stability (50 bootstrap LASSO runs)")
plt.tight_layout()
plt.savefig(OUT_DIR/"stability_selection.png",dpi=150,bbox_inches="tight")
plt.close()

log("  All plots saved")

# ── Text summary ──────────────────────────────────────────────────────────────
txt_lines=[
    "="*70,
    "ECG Feature Selection Results  (Steps A–E + model comparison)",
    f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    "="*70,"",
    "FEATURE SELECTION CONSENSUS",
    "-"*50,
]
for _,row in sel_df.iterrows():
    txt_lines.append(f"  [{row['Votes (of 5)']}/5] {row['Variable']:30s}  {row['Methods']}"
                     f"  OR={row['Univ OR']:.3f}  p={row['Univ p']:.4f}")

txt_lines+=["","MODEL PERFORMANCE (5-fold CV, logistic, primary endpoint)","-"*50]
for row in perf_rows:
    d=row.get("Delta AUC vs M3",""); p=row.get("p vs M3","")
    delta_str=f"  ΔAUC={d:+.4f} p={p:.4f}" if isinstance(d,float) else ""
    txt_lines.append(
        f"  {row['Model']:30s} [{row['N ECG feats']:2d} ECG]  "
        f"AUC={row['AUC mean']:.4f}±{row['AUC SD']:.4f}  "
        f"Cal={row['Cal slope']}  Cox-C={row['Cox C-stat']}{delta_str}"
    )

txt_lines+=["","RECOMMENDED MODEL 4 COMPOSITION","-"*50,
    "  Based on consensus ≥3 methods + clinical plausibility:"]
rec_ecg=consensus_3 if consensus_3 else stab_vars
for v in rec_ecg:
    txt_lines.append(f"  • {DISPLAY_ECG.get(v,v)}")
txt_lines+=["","Output files:","  "+str(OUT_DIR),"="*70]

summary="\n".join(txt_lines)
print("\n"+summary)
(OUT_DIR/"summary.txt").write_text(summary)
log("✓ Step F done. All results in results/feature_selection/")

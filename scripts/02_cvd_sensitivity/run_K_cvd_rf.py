#!/usr/bin/env python3
"""Step K: Random Forest permutation importance — CVD composite endpoint."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from pathlib import Path
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

DATA_PATH = Path("data/INTERASPIRE_analysis_dataset.xlsx")
OUT_DIR   = Path("results/feature_selection")

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
M3_VARS = ["CVARAGE","RQSEX","CVARCOSMOKING","CVARBMI","CVARSYSTOL","CVARLABHBA1CPERCENT",
           "CVAREGFR","CVARLABHDL","CVARLABLDLFRIEDEWALD","RQHISTOFPRECVD","RQHISTOFCORARTDIS",
           "RQHISTOFHEARTFAIL","any_af","CVARMEDBPLOWERING","CVARMEDLIPIDLOWERING",
           "CVARMEDGLUCOSELOWERING","RQANTICOAGULANTS","RQHISTOFCOPD",
           "idx_STEMI","idx_NSTEMI","idx_UA","idx_ePCI"]
ECG_ALL = ["Atrial Fibrillation","LBBB","RBBB","Q Wave","ST Elevation","ST Depression",
           "T Wave Inversion","Ischaemic","QT Prolongation","LVH","1 AV Block",
           "Left Axis Deviation","Right Axis Deviation","MI (Old)","MI(Acute)",
           "PR Interval (ms)","QRS Duration (ms)","QT Interval (ms)"]
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

log("Loading data ...")
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

cvd_death=df["DEATHCAUSE_excel2"].isin([1,2,3])
first_comp=df["first_component_Composite_4"].fillna("")
df["time_Composite_4"]=df["time_Composite_4"].clip(0,365)
df["event_CVD4"]=((first_comp=="SURVIVAL")&cvd_death |
                   first_comp.isin(["HOSPAMI","HOSPHF","HOSPSTROKE"])).astype(int)
complete=(df["time_Composite_4"]>=365)|(df["event_CVD4"]==1)
df_lr=df[complete].reset_index(drop=True)
y=df_lr["event_CVD4"].values.astype(float)
log(f"  CVD composite LR cohort: {len(df_lr)}, events: {int(y.sum())}")

avail=[v for v in M3_VARS+ECG_ALL if v in df_lr.columns]
imp=SimpleImputer(strategy="median"); sc=StandardScaler()
Xf=sc.fit_transform(imp.fit_transform(df_lr[avail]))
feat_names=avail

log("Fitting Random Forest (200 trees) ...")
rf=RandomForestClassifier(n_estimators=200,max_depth=6,min_samples_leaf=10,
    class_weight="balanced",random_state=42,n_jobs=1)
rf.fit(Xf,y)
log("Computing permutation importance (10 repeats) ...")
perm=permutation_importance(rf,Xf,y,n_repeats=10,scoring="roc_auc",random_state=42,n_jobs=1)
rf_imp=pd.Series(perm.importances_mean,index=feat_names)
rf_std=pd.Series(perm.importances_std,index=feat_names)

rows=[]
for v in ECG_ALL:
    if v not in feat_names: continue
    rows.append({"Variable":DISPLAY_ECG.get(v,v),"raw_col":v,
                 "RF perm imp (mean)":round(rf_imp[v],6),
                 "RF perm imp (std)":round(rf_std[v],6)})
out=pd.DataFrame(rows).sort_values("RF perm imp (mean)",ascending=False)
out.to_csv(OUT_DIR/"cvd_rf_importance.csv",index=False)

log("\nRF permutation importance (CVD composite):")
for _,row in out.iterrows():
    log(f"  {row['Variable']:30s}  {row['RF perm imp (mean)']:+.6f} ± {row['RF perm imp (std)']:.6f}")
top5=[out.iloc[i]["raw_col"] for i in range(min(5,len(out)))]
log(f"\nTop-5: {[DISPLAY_ECG.get(v,v) for v in top5]}")
log("Saved → cvd_rf_importance.csv")
log("✓ Step K done.")

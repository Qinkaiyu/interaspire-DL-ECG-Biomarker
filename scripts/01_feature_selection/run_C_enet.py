#!/usr/bin/env python3
"""Step C: Elastic Net (α=0.5) feature selection for ECG biomarkers."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from pathlib import Path
import time

from sklearn.linear_model import LogisticRegressionCV
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

log("Loading data …")
df = pd.read_excel(DATA_PATH)

def enc(s): return s.map({"Yes":1,"No":0,"yes":1,"no":0})
def binn(s,e): return pd.cut(s,bins=e,labels=range(len(e)-1),include_lowest=True).astype(float)

df["RQSEX"] = df["RQSEX"].map({"Male":1,"Female":0})
for v in ["CVARCOSMOKING","RQHISTOFPRECVD","RQHISTOFCORARTDIS","RQHISTOFHEARTFAIL",
          "CVARMEDBPLOWERING","CVARMEDLIPIDLOWERING","CVARMEDGLUCOSELOWERING",
          "RQANTICOAGULANTS","RQHISTOFCOPD"]:
    df[v] = enc(df[v])
for v in ECG_ALL[:-3]:
    if v in df.columns: df[v] = enc(df[v])
df["idx_STEMI"]  = (df["RQINDEX"]=="Acute myocardial infarction STEMI").astype(float)
df["idx_NSTEMI"] = (df["RQINDEX"]=="Acute myocardial infarction Non-STEMI").astype(float)
df["idx_UA"]     = (df["RQINDEX"]=="Unstable angina / Acute myocardial ischaemia").astype(float)
df["idx_ePCI"]   = (df["RQINDEX"]=="Elective percutaneous transluminal coronary angioplasty").astype(float)

for ai,fh in [("PR Interval (ms)","FHPRINTER"),("QRS Duration (ms)","FHQRSDURA"),("QT Interval (ms)","FHQTC")]:
    m = df[ai].isna()&df[fh].notna(); df.loc[m,ai]=df.loc[m,fh]
m = df["RBBB"].isna()
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

df["time_Composite_4"]=df["time_Composite_4"].clip(0,365)
complete=(df["time_Composite_4"]>=365)|(df["event_Composite_4"]==1)
df_lr=df[complete].reset_index(drop=True)
y=df_lr["event_Composite_4"].values.astype(float)
log(f"  LR cohort: {len(df_lr)}, events: {int(y.sum())}")

avail=[v for v in M3_VARS+ECG_ALL if v in df_lr.columns]
Xf=StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(df_lr[avail]))
feat_names=avail

log("Fitting Elastic Net (α=0.5) with 5-fold inner CV …")
Cs=np.logspace(-3,1,20)
enet_cv=LogisticRegressionCV(
    Cs=Cs, cv=5, penalty="elasticnet", l1_ratios=[0.5],
    solver="saga", scoring="roc_auc", max_iter=3000,
    random_state=42, n_jobs=1,
)
enet_cv.fit(Xf, y)
log(f"  Best C: {enet_cv.C_[0]:.5f}  l1_ratio: {enet_cv.l1_ratio_[0]:.2f}")

coefs=pd.Series(enet_cv.coef_[0], index=feat_names)
rows=[]
for v in ECG_ALL:
    if v not in feat_names: continue
    c=coefs[v]
    rows.append({"Variable":DISPLAY_ECG.get(v,v),"raw_col":v,
                 "ElasticNet coef":round(c,5),"Selected":c!=0})
    log(f"  {DISPLAY_ECG.get(v,v):30s}  coef={c:.5f}  {'← selected' if c!=0 else ''}")

out=pd.DataFrame(rows).sort_values("ElasticNet coef",key=abs,ascending=False)
out.to_csv(OUT_DIR/"enet_ecg_coefs.csv",index=False)
selected=[r["raw_col"] for r in rows if r["Selected"]]
log(f"\nSelected {len(selected)} ECG features: {[DISPLAY_ECG.get(v,v) for v in selected]}")
log("Saved → results/feature_selection/enet_ecg_coefs.csv")
log("✓ Step C done.")

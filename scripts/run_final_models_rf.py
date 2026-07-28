#!/usr/bin/env python3
"""
Final Models 0–4 using server-side fold splits — Random Forest.

Same fold splits and endpoints as LR version, but using RF classifier.
Cox regression unchanged.

Results → results/final_rf/
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from pathlib import Path
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

BASE     = Path(".")
DATA_PATH = BASE / "data" / "INTERASPIRE_analysis_dataset.xlsx"
OUT_DIR   = BASE / "results" / "final_rf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# ── Fold split files ───────────────────────────────────────────────────────────
SPLITS = {
    "全因4结局": {
        "file": BASE / "fold_splits_mace4_1yr.csv",
        "time": "time_Composite_4", "event": "event_Composite_4",
    },
    "CVD4结局": {
        "file": BASE / "fold_splits_cvd_mace4_1yr.csv",
        "time": "time_Composite_4", "event": "event_CVD_Composite_4",
    },
    "全因6结局": {
        "file": BASE / "fold_splits_mace6_1yr.csv",
        "time": "time_Composite_6", "event": "event_Composite_6",
    },
    "CVD6结局": {
        "file": BASE / "fold_splits_cvd_mace6_1yr.csv",
        "time": "time_Composite_6", "event": "event_CVD_Composite_6",
    },
}

# ── Model definitions ──────────────────────────────────────────────────────────
M0_VARS = ["CVARAGE","RQSEX"]
M1_VARS = ["CVARAGE","RQSEX","CVARCOSMOKING","CVARBMI","CVARSYSTOL","CVARLABHBA1CPERCENT",
           "CVAREGFR","CVARLABHDL","CVARLABLDLFRIEDEWALD"]
M2_VARS = M1_VARS + ["RQHISTOFPRECVD","RQHISTOFCORARTDIS","RQHISTOFHEARTFAIL",
                     "any_af","CVARMEDBPLOWERING","CVARMEDLIPIDLOWERING",
                     "CVARMEDGLUCOSELOWERING","RQANTICOAGULANTS","RQHISTOFCOPD"]
M3_VARS = M2_VARS + ["idx_STEMI","idx_NSTEMI","idx_UA","idx_ePCI"]
ECG_M4  = ["QT Prolongation","ST Depression","QT Interval (ms)"]
M4_VARS = M3_VARS + ECG_M4

ECG_BIN = ["Atrial Fibrillation","LBBB","RBBB","Q Wave","ST Elevation","ST Depression",
           "T Wave Inversion","Ischaemic","QT Prolongation","LVH","1 AV Block",
           "Left Axis Deviation","Right Axis Deviation","MI (Old)","MI(Acute)"]

BIN_EDGES = {
    "PR Interval (ms)":[0,120,200,2000],"QRS Duration (ms)":[0,120,1000],
    "QT Interval (ms)":[0,440,480,5000],"FHPRINTER":[0,120,200,2000],
    "FHQRSDURA":[0,120,1000],"FHQTC":[0,440,480,5000],
    "CVARSYSTOL":[0,120,130,140,180,250],"CVARLABHBA1CPERCENT":[0,5.7,6.5,20],
    "CVARLABLDLFRIEDEWALD":[0,1.8,2.6,3.4,4.9,10],"CVARLABHDL":[0,1.0,1.3,1.55,5],
    "CVAREGFR":[0,15,30,45,60,90,200],"CVARBMI":[0,18.5,25,30,35,100],
    "CVARAGE":[0,50,60,70,80,150],
}

MODELS = [
    ("Model 0 (Age+Sex)", M0_VARS),
    ("Model 1", M1_VARS),
    ("Model 2", M2_VARS),
    ("Model 3", M3_VARS),
    ("Model 4 (3 ECG)", M4_VARS),
]

RANDOM_STATE = 42

# ── Load & preprocess main data ───────────────────────────────────────────────
log("Loading data …")
df = pd.read_excel(DATA_PATH)
log(f"  Raw rows: {len(df)}")

def enc(s): return s.map({"Yes":1,"No":0,"yes":1,"no":0})
def binn(s,e): return pd.cut(s,bins=e,labels=range(len(e)-1),include_lowest=True).astype(float)

df["RQSEX"] = df["RQSEX"].map({"Male":1,"Female":0})
for v in ["CVARCOSMOKING","RQHISTOFPRECVD","RQHISTOFCORARTDIS","RQHISTOFHEARTFAIL",
          "CVARMEDBPLOWERING","CVARMEDLIPIDLOWERING","CVARMEDGLUCOSELOWERING",
          "RQANTICOAGULANTS","RQHISTOFCOPD"]:
    df[v] = enc(df[v])
for v in ECG_BIN:
    if v in df.columns: df[v] = enc(df[v])
df["idx_STEMI"]  = (df["RQINDEX"]=="Acute myocardial infarction STEMI").astype(float)
df["idx_NSTEMI"] = (df["RQINDEX"]=="Acute myocardial infarction Non-STEMI").astype(float)
df["idx_UA"]     = (df["RQINDEX"]=="Unstable angina / Acute myocardial ischaemia").astype(float)
df["idx_ePCI"]   = (df["RQINDEX"]=="Elective percutaneous transluminal coronary angioplasty").astype(float)
for ai,fh in [("PR Interval (ms)","FHPRINTER"),("QRS Duration (ms)","FHQRSDURA"),("QT Interval (ms)","FHQTC")]:
    m = df[ai].isna() & df[fh].notna(); df.loc[m, ai] = df.loc[m, fh]
m = df["RBBB"].isna()
df.loc[m & (df["FHBUNBRBLO"]=="Right bundle branch block"), "RBBB"] = 1
df.loc[m & df["FHBUNBRBLO"].notna() & (df["FHBUNBRBLO"]!="Right bundle branch block"), "RBBB"] = 0
df.loc[m & (df["FHBUNBRBLO"]=="Left bundle branch block"), "LBBB"] = 1
df.loc[m & df["FHBUNBRBLO"].notna() & (df["FHBUNBRBLO"]!="Left bundle branch block"), "LBBB"] = 0
m2 = df["1 AV Block"].isna()
df.loc[m2 & (df["FHAVNOBL"]=="I°"), "1 AV Block"] = 1
df.loc[m2 & df["FHAVNOBL"].notna() & (df["FHAVNOBL"]!="I°"), "1 AV Block"] = 0
mlvh = df["LVH"].isna(); flvh = enc(df["FHLEFTVENHYP"])
df.loc[mlvh & flvh.notna(), "LVH"] = flvh[mlvh & flvh.notna()]
for col, edges in BIN_EDGES.items():
    if col in df.columns: df[col] = binn(df[col], edges)

log("Preprocessing done ✓")

# ── CV helpers ─────────────────────────────────────────────────────────────────
def prep(Xtr, Xte):
    imp = SimpleImputer(strategy="median"); sc = StandardScaler()
    return (sc.fit_transform(imp.fit_transform(Xtr)),
            sc.transform(imp.transform(Xte)),
            list(Xtr.columns))

def run_cv(data, vlist, folds, evt_col, time_col):
    """Run RF + Cox CV, return avg + pooled metrics."""
    avail = [v for v in vlist if v in data.columns]
    X = data[avail]
    y = data[evt_col].values.astype(float)
    t = data[time_col].values.astype(float)

    fold_aucs, fold_briers, fold_cs = [], [], []
    oof_probs = np.full(len(data), np.nan)
    oof_risk  = np.full(len(data), np.nan)

    for fold_id in sorted(folds.keys()):
        te_idx = folds[fold_id]
        tr_idx = []
        for fid, idxs in folds.items():
            if fid != fold_id:
                tr_idx.extend(idxs)
        tr_idx = sorted(tr_idx)

        Xtr, Xte, cols = prep(X.iloc[tr_idx], X.iloc[te_idx])
        ytr, yte = y[tr_idx], y[te_idx]
        ttr, tte = t[tr_idx], t[te_idx]

        # RF
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=10,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1)
        rf.fit(Xtr, ytr)
        p = rf.predict_proba(Xte)[:,1]
        oof_probs[te_idx] = p
        if len(np.unique(yte)) > 1:
            fold_aucs.append(roc_auc_score(yte, p))
        fold_briers.append(brier_score_loss(yte, p))

        # Cox
        tdf = pd.DataFrame(Xtr, columns=cols)
        tdf["_t"] = np.clip(ttr, 0.5, None)
        tdf["_e"] = ytr
        cph = CoxPHFitter(penalizer=0.1)
        try:
            cph.fit(tdf, "_t", "_e", show_progress=False)
            risk = cph.predict_partial_hazard(pd.DataFrame(Xte, columns=cols)).values.flatten()
            oof_risk[te_idx] = risk
            fold_cs.append(concordance_index(np.clip(tte, 0.5, None), -risk, yte))
        except:
            pass

    # Average
    avg_auc = np.mean(fold_aucs) if fold_aucs else np.nan
    avg_auc_sd = np.std(fold_aucs) if fold_aucs else np.nan
    avg_brier = np.mean(fold_briers)
    avg_c = np.mean(fold_cs) if fold_cs else np.nan
    avg_c_sd = np.std(fold_cs) if fold_cs else np.nan

    # Pooled
    valid_lr = ~np.isnan(oof_probs)
    pooled_auc = roc_auc_score(y[valid_lr], oof_probs[valid_lr]) if valid_lr.sum() > 0 else np.nan
    pooled_brier = brier_score_loss(y[valid_lr], oof_probs[valid_lr]) if valid_lr.sum() > 0 else np.nan

    valid_cox = ~np.isnan(oof_risk)
    pooled_c = concordance_index(np.clip(t[valid_cox], 0.5, None), -oof_risk[valid_cox], y[valid_cox]) if valid_cox.sum() > 0 else np.nan

    return {
        "avg_auc": avg_auc, "avg_auc_sd": avg_auc_sd,
        "avg_brier": avg_brier,
        "avg_c": avg_c, "avg_c_sd": avg_c_sd,
        "pooled_auc": pooled_auc, "pooled_brier": pooled_brier,
        "pooled_c": pooled_c,
        "fold_aucs": fold_aucs, "fold_cs": fold_cs,
    }

# ── Run all endpoints ──────────────────────────────────────────────────────────
all_rows = []

for ep_name, ep_cfg in SPLITS.items():
    log(f"\n{'='*60}")
    log(f"Endpoint: {ep_name}")

    # Load fold split
    split_df = pd.read_csv(ep_cfg["file"])
    evt_col = ep_cfg["event"]
    time_col = ep_cfg["time"]
    n_total = len(split_df)
    n_events = int(split_df[evt_col].sum())

    log(f"  N={n_total}, events={n_events}")

    # Merge fold assignments with main data
    merged = df.merge(split_df[["STUDYID", "fold", evt_col, time_col]],
                      on="STUDYID", how="inner", suffixes=("_orig", ""))

    # Use event/time from the fold split file (definitive)
    log(f"  Merged: {len(merged)} rows")

    # Build fold dict: {fold_id: [row_indices]}
    folds = {}
    for fold_id in sorted(merged["fold"].unique()):
        folds[fold_id] = merged.index[merged["fold"] == fold_id].tolist()

    fold_desc = ", ".join(f"{k}: {len(v)}" for k, v in sorted(folds.items()))
    log(f"  Folds: {{{fold_desc}}}")

    for mname, vlist in MODELS:
        r = run_cv(merged, vlist, folds, evt_col, time_col)

        row = {
            "结局": ep_name,
            "模型": mname,
            "变量数": len([v for v in vlist if v in merged.columns]),
            "N": n_total,
            "事件数": n_events,
            "Avg_AUC": round(r["avg_auc"], 4),
            "Avg_AUC_SD": round(r["avg_auc_sd"], 4),
            "Avg_AUC_95CI": f"{r['avg_auc']-1.96*r['avg_auc_sd']:.4f}–{r['avg_auc']+1.96*r['avg_auc_sd']:.4f}",
            "Pooled_AUC": round(r["pooled_auc"], 4),
            "Avg_Brier": round(r["avg_brier"], 4),
            "Pooled_Brier": round(r["pooled_brier"], 4),
            "Avg_CoxC": round(r["avg_c"], 4),
            "Avg_CoxC_SD": round(r["avg_c_sd"], 4),
            "Pooled_CoxC": round(r["pooled_c"], 4),
            "Fold_AUCs": "|".join(f"{a:.4f}" for a in r["fold_aucs"]),
            "Fold_CoxCs": "|".join(f"{c:.4f}" for c in r["fold_cs"]),
        }
        all_rows.append(row)

        log(f"  {mname:25s} [{row['变量数']:2d}]  "
            f"AvgAUC={r['avg_auc']:.4f}±{r['avg_auc_sd']:.4f}  "
            f"PoolAUC={r['pooled_auc']:.4f}  "
            f"AvgC={r['avg_c']:.4f}  PoolC={r['pooled_c']:.4f}")

# ── Save ───────────────────────────────────────────────────────────────────────
out_df = pd.DataFrame(all_rows)
out_df.to_csv(OUT_DIR / "final_performance.csv", index=False)
log(f"\nSaved → final_performance.csv")

# ── Per-model per-endpoint CSVs ────────────────────────────────────────────────
for mname, _ in MODELS:
    mdir = OUT_DIR / mname.split("(")[0].strip().replace(" ","_")
    mdir.mkdir(exist_ok=True)
    for ep_name in SPLITS:
        ep_rows = out_df[(out_df["模型"] == mname) & (out_df["结局"] == ep_name)]
        ep_rows.to_csv(mdir / f"{ep_name}.csv", index=False)

log("Per-model CSVs saved ✓")

# ── Print final summary ───────────────────────────────────────────────────────
print("\n" + "="*100)
print("FINAL RESULTS — Server Fold Splits [Random Forest]")
print("="*100)

for ep_name in SPLITS:
    ep_rows = [r for r in all_rows if r["结局"] == ep_name]
    print(f"\n{ep_name}  (N={ep_rows[0]['N']}, events={ep_rows[0]['事件数']})")
    print(f"  {'模型':<28}  {'#var':>4}  {'Avg AUC':>8}  {'Pool AUC':>9}  "
          f"{'Avg C':>7}  {'Pool C':>7}  {'Avg Brier':>9}")
    print("  " + "-"*85)
    for r in ep_rows:
        print(f"  {r['模型']:<28}  {r['变量数']:4d}  {r['Avg_AUC']:8.4f}  {r['Pooled_AUC']:9.4f}  "
              f"{r['Avg_CoxC']:7.4f}  {r['Pooled_CoxC']:7.4f}  {r['Avg_Brier']:9.4f}")

print("\n" + "="*100)

# ── LR vs RF comparison ───────────────────────────────────────────────────────
lr_path = BASE / "results" / "final" / "final_performance.csv"
if lr_path.exists():
    lr_df = pd.read_csv(lr_path)
    print("\n" + "="*100)
    print("RANDOM FOREST vs LOGISTIC REGRESSION — Comparison")
    print("="*100)

    for ep_name in SPLITS:
        rf_ep = [r for r in all_rows if r["结局"] == ep_name]
        lr_ep = lr_df[lr_df["结局"] == ep_name]
        print(f"\n{ep_name}  (N={rf_ep[0]['N']}, events={rf_ep[0]['事件数']})")
        print(f"  {'模型':<28}  {'RF AUC':>7}  {'LR AUC':>7}  {'Δ':>7}  "
              f"{'RF Pool':>7}  {'LR Pool':>7}  {'Δ':>7}")
        print("  " + "-"*80)
        for r in rf_ep:
            lr_row = lr_ep[lr_ep["模型"] == r["模型"]]
            if len(lr_row) > 0:
                lr_auc = lr_row.iloc[0]["Avg_AUC"]
                lr_pool = lr_row.iloc[0]["Pooled_AUC"]
                d_avg = r["Avg_AUC"] - lr_auc
                d_pool = r["Pooled_AUC"] - lr_pool
                print(f"  {r['模型']:<28}  {r['Avg_AUC']:7.4f}  {lr_auc:7.4f}  {d_avg:+7.4f}  "
                      f"{r['Pooled_AUC']:7.4f}  {lr_pool:7.4f}  {d_pool:+7.4f}")

    print("\n" + "="*100)

log("✓ Done.")

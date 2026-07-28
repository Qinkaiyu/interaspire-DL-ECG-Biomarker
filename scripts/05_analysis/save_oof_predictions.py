"""
Re-run 5-fold CV for Models 0-4 on CVD4 and CVD6 endpoints.
Save out-of-fold predicted probabilities for every patient.

Uses Cox PH only (no LR). Outputs:
  - *_cox_prob: Cox predicted 1-year event probability 1-S(365)
  - *_cox_risk: Cox partial hazard (for C-index)
  - landmark: boolean, True if (time >= 365) or (event == 1)

Mirrors run_final_models.py to ensure reproducibility.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from pathlib import Path
import time, pickle

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

BASE = Path(".")
DATA_PATH = BASE / "data" / "INTERASPIRE_analysis_dataset.xlsx"
OUT_DIR = BASE / "results" / "oof_predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LANDMARK_T = 365

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# ── Endpoints (CVD only) ─────────────────────────────────────────────────────
SPLITS = {
    "CVD4": {
        "file": BASE / "fold_splits_cvd_mace4_1yr.csv",
        "time": "time_Composite_4", "event": "event_CVD_Composite_4",
    },
    "CVD6": {
        "file": BASE / "fold_splits_cvd_mace6_1yr.csv",
        "time": "time_Composite_6", "event": "event_CVD_Composite_6",
    },
}

# ── Model definitions (identical to run_final_models.py) ─────────────────────
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
    ("Model_0", M0_VARS),
    ("Model_1", M1_VARS),
    ("Model_2", M2_VARS),
    ("Model_3", M3_VARS),
    ("Model_4", M4_VARS),
]

RANDOM_STATE = 42

# ── Load & preprocess (identical to run_final_models.py) ─────────────────────
log("Loading data ...")
df = pd.read_excel(DATA_PATH, engine="openpyxl")
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

# FH fallback fills (same as original)
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

log("Preprocessing done")

# ── CV helper ────────────────────────────────────────────────────────────────
def prep(Xtr, Xte):
    imp = SimpleImputer(strategy="median"); sc = StandardScaler()
    Xtr_out = sc.fit_transform(imp.fit_transform(Xtr))
    Xte_out = sc.transform(imp.transform(Xte))
    return Xtr_out, Xte_out, imp, sc

def run_cv_save_oof(data, vlist, folds, evt_col, time_col, model_name):
    """Cox-only CV, return oof cox_prob [1-S(365)] and cox_risk [partial hazard]."""
    avail = [v for v in vlist if v in data.columns]
    X = data[avail]
    y = data[evt_col].values.astype(float)
    t = data[time_col].values.astype(float)

    oof_cox_prob = np.full(len(data), np.nan)
    oof_cox_risk = np.full(len(data), np.nan)
    fold_cs = []
    trained_models = {}

    for fold_id in sorted(folds.keys()):
        te_idx = folds[fold_id]
        tr_idx = []
        for fid, idxs in folds.items():
            if fid != fold_id:
                tr_idx.extend(idxs)
        tr_idx = sorted(tr_idx)

        Xtr_s, Xte_s, imp, sc = prep(X.iloc[tr_idx], X.iloc[te_idx])
        ytr, yte = y[tr_idx], y[te_idx]
        ttr, tte = t[tr_idx], t[te_idx]

        cols = list(X.columns)
        tdf = pd.DataFrame(Xtr_s, columns=cols)
        tdf["_t"] = np.clip(ttr, 0.5, None)
        tdf["_e"] = ytr
        cph = CoxPHFitter(penalizer=0.1)
        try:
            cph.fit(tdf, "_t", "_e", show_progress=False)
        except Exception as ex:
            log(f"    Cox failed fold {fold_id}: {ex}")
            trained_models[fold_id] = {"cph": None, "imp": imp, "sc": sc, "cols": cols}
            continue

        te_df = pd.DataFrame(Xte_s, columns=cols)

        # Partial hazard (for C-index)
        risk = cph.predict_partial_hazard(te_df).values.flatten()
        oof_cox_risk[te_idx] = risk
        fold_cs.append(concordance_index(np.clip(tte, 0.5, None), -risk, yte))

        # Predicted 1yr event probability: 1 - S(365)
        sf = cph.predict_survival_function(te_df, times=[LANDMARK_T])
        s365 = sf.values.flatten()
        oof_cox_prob[te_idx] = 1.0 - s365

        trained_models[fold_id] = {"cph": cph, "imp": imp, "sc": sc, "cols": cols}

    # Landmark filter
    lm_mask = (t >= LANDMARK_T) | (y == 1)
    y_1yr = ((y == 1) & (t <= LANDMARK_T)).astype(float)

    avg_c = np.mean(fold_cs) if fold_cs else np.nan
    valid_cox = ~np.isnan(oof_cox_risk)
    pooled_c = (concordance_index(np.clip(t[valid_cox], 0.5, None),
                                   -oof_cox_risk[valid_cox], y[valid_cox])
                if valid_cox.sum() > 0 else np.nan)

    valid_p = ~np.isnan(oof_cox_prob) & lm_mask
    pooled_auc = roc_auc_score(y_1yr[valid_p], oof_cox_prob[valid_p]) if valid_p.sum() > 0 else np.nan

    log(f"  {model_name:20s}  AUC@1yr={pooled_auc:.4f}  AvgC={avg_c:.4f}  PoolC={pooled_c:.4f}  "
        f"(N_lm={int(valid_p.sum())})")

    return oof_cox_prob, oof_cox_risk, trained_models


# ── Run for CVD4 and CVD6 ────────────────────────────────────────────────────
for ep_name, ep_cfg in SPLITS.items():
    log(f"\n{'='*60}")
    log(f"Endpoint: {ep_name}")

    split_df = pd.read_csv(ep_cfg["file"])
    evt_col = ep_cfg["event"]
    time_col = ep_cfg["time"]
    log(f"  N={len(split_df)}, events={int(split_df[evt_col].sum())}")

    merged = df.merge(split_df[["STUDYID", "fold", evt_col, time_col]],
                      on="STUDYID", how="inner", suffixes=("_orig", ""))
    log(f"  Merged: {len(merged)} rows")

    folds = {}
    for fold_id in sorted(merged["fold"].unique()):
        folds[fold_id] = merged.index[merged["fold"] == fold_id].tolist()

    # Collect oof predictions for all models
    t_vals = merged[time_col].values.astype(float)
    y_vals = merged[evt_col].values.astype(float)
    oof_df = merged[["STUDYID", "fold", evt_col, time_col]].copy()
    oof_df["landmark"] = (t_vals >= LANDMARK_T) | (y_vals == 1)

    all_trained = {}
    for mname, vlist in MODELS:
        cox_prob, cox_risk, trained = run_cv_save_oof(
            merged, vlist, folds, evt_col, time_col, mname
        )
        oof_df[f"{mname}_cox_prob"] = cox_prob
        oof_df[f"{mname}_cox_risk"] = cox_risk
        all_trained[mname] = trained

    # Save oof predictions
    oof_path = OUT_DIR / f"oof_{ep_name}.csv"
    oof_df.to_csv(oof_path, index=False)
    log(f"  Saved oof predictions -> {oof_path}")

    # Save trained models
    model_path = OUT_DIR / f"models_{ep_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(all_trained, f)
    log(f"  Saved trained models -> {model_path}")

    # Verification
    lm = oof_df["landmark"]
    y_1yr = ((oof_df[evt_col]==1) & (oof_df[time_col]<=LANDMARK_T)).astype(float)
    log(f"\n  Verification (Pooled AUC@1yr on landmark subset, N={int(lm.sum())}):")
    for mname, _ in MODELS:
        col = f"{mname}_cox_prob"
        valid = ~oof_df[col].isna() & lm
        auc = roc_auc_score(y_1yr[valid], oof_df.loc[valid, col])
        log(f"    {mname:20s}: {auc:.4f}")

log(f"\n{'='*60}")
log("Done. Files saved to results/oof_predictions/")

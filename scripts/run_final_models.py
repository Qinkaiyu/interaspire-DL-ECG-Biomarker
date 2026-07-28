#!/usr/bin/env python3
"""
Final Models 0–4 using server-side fold splits.

Model 0: Age + Sex (2 vars)
Model 1: Basic risk factors (9 vars)
Model 2: + History/medications (18 vars)
Model 3: + Index diagnosis (22 vars)
Model 4: + 3 ECG biomarkers (25 vars)

4 endpoints × 5 models.
All metrics from Cox PH only (no separate LR):
  - C-index: on full data (handles censoring natively)
  - AUC at 1yr: Cox predicted 1-S(365), evaluated on landmark complete-case subset
  - Brier at 1yr: same as above
Results → results/final/
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from pathlib import Path
import time as _time

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

BASE      = Path(".")
DATA_PATH = BASE / "data" / "INTERASPIRE_analysis_dataset.xlsx"
OUT_DIR   = BASE / "results" / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LANDMARK_T = 365  # 1-year landmark

def log(m): print(f"[{_time.strftime('%H:%M:%S')}] {m}", flush=True)

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

log("Preprocessing done")

# ── CV helpers ─────────────────────────────────────────────────────────────────
def prep(Xtr, Xte):
    imp = SimpleImputer(strategy="median"); sc = StandardScaler()
    Xtr_s = sc.fit_transform(imp.fit_transform(Xtr))
    Xte_s = sc.transform(imp.transform(Xte))
    return Xtr_s, Xte_s, list(Xtr.columns), imp, sc


def run_cv(data, vlist, folds, evt_col, time_col):
    """Cox-only CV: C-index on all data, AUC/Brier on landmark complete-case subset."""
    avail = [v for v in vlist if v in data.columns]
    X = data[avail]
    y = data[evt_col].values.astype(float)
    t = data[time_col].values.astype(float)

    # Landmark complete-case mask: know outcome at 1yr
    lm_mask = (t >= LANDMARK_T) | (y == 1)
    # Binary label for 1yr: event within LANDMARK_T
    y_1yr = ((y == 1) & (t <= LANDMARK_T)).astype(float)

    fold_cs, fold_aucs, fold_briers = [], [], []
    oof_risk  = np.full(len(data), np.nan)   # partial hazard (for C-index)
    oof_prob  = np.full(len(data), np.nan)   # 1-S(365) (for AUC/Brier)

    for fold_id in sorted(folds.keys()):
        te_idx = folds[fold_id]
        tr_idx = []
        for fid, idxs in folds.items():
            if fid != fold_id:
                tr_idx.extend(idxs)
        tr_idx = sorted(tr_idx)

        Xtr_s, Xte_s, cols, imp, sc = prep(X.iloc[tr_idx], X.iloc[te_idx])
        ytr, yte = y[tr_idx], y[te_idx]
        ttr, tte = t[tr_idx], t[te_idx]

        # Fit Cox PH
        tdf = pd.DataFrame(Xtr_s, columns=cols)
        tdf["_t"] = np.clip(ttr, 0.5, None)
        tdf["_e"] = ytr
        cph = CoxPHFitter(penalizer=0.1)
        try:
            cph.fit(tdf, "_t", "_e", show_progress=False)
        except Exception:
            continue

        te_df = pd.DataFrame(Xte_s, columns=cols)

        # C-index: partial hazard on full test fold
        risk = cph.predict_partial_hazard(te_df).values.flatten()
        oof_risk[te_idx] = risk
        fold_cs.append(concordance_index(np.clip(tte, 0.5, None), -risk, yte))

        # AUC/Brier at 1yr: predict S(365), evaluate on landmark complete subset
        sf = cph.predict_survival_function(te_df, times=[LANDMARK_T])
        s365 = sf.values.flatten()  # S(365|x) for each test subject
        p_event = 1.0 - s365        # predicted 1yr event probability
        oof_prob[te_idx] = p_event

        # Landmark filter for this fold's test set
        lm_te = lm_mask[te_idx]
        yte_1yr = y_1yr[te_idx]
        if lm_te.sum() > 0 and len(np.unique(yte_1yr[lm_te])) > 1:
            fold_aucs.append(roc_auc_score(yte_1yr[lm_te], p_event[lm_te]))
            fold_briers.append(brier_score_loss(yte_1yr[lm_te], p_event[lm_te]))

    # ── Averages ──
    avg_c    = np.mean(fold_cs) if fold_cs else np.nan
    avg_c_sd = np.std(fold_cs)  if fold_cs else np.nan
    avg_auc    = np.mean(fold_aucs) if fold_aucs else np.nan
    avg_auc_sd = np.std(fold_aucs)  if fold_aucs else np.nan
    avg_brier  = np.mean(fold_briers) if fold_briers else np.nan

    # ── Pooled (all OOF predictions concatenated) ──
    valid_c = ~np.isnan(oof_risk)
    pooled_c = concordance_index(
        np.clip(t[valid_c], 0.5, None), -oof_risk[valid_c], y[valid_c]
    ) if valid_c.sum() > 0 else np.nan

    valid_p = ~np.isnan(oof_prob) & lm_mask
    pooled_auc   = roc_auc_score(y_1yr[valid_p], oof_prob[valid_p]) if valid_p.sum() > 0 else np.nan
    pooled_brier = brier_score_loss(y_1yr[valid_p], oof_prob[valid_p]) if valid_p.sum() > 0 else np.nan

    # Count how many landmark-complete subjects
    n_landmark = int(valid_p.sum())

    return {
        "avg_auc": avg_auc, "avg_auc_sd": avg_auc_sd,
        "avg_brier": avg_brier,
        "avg_c": avg_c, "avg_c_sd": avg_c_sd,
        "pooled_auc": pooled_auc, "pooled_brier": pooled_brier,
        "pooled_c": pooled_c,
        "fold_aucs": fold_aucs, "fold_cs": fold_cs,
        "fold_briers": fold_briers,
        "n_landmark": n_landmark,
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
            "N_landmark": r["n_landmark"],
            "Avg_AUC_1yr": round(r["avg_auc"], 4),
            "Avg_AUC_1yr_SD": round(r["avg_auc_sd"], 4),
            "Avg_AUC_1yr_95CI": f"{r['avg_auc']-1.96*r['avg_auc_sd']:.4f}-{r['avg_auc']+1.96*r['avg_auc_sd']:.4f}",
            "Pooled_AUC_1yr": round(r["pooled_auc"], 4),
            "Avg_Brier_1yr": round(r["avg_brier"], 4),
            "Pooled_Brier_1yr": round(r["pooled_brier"], 4),
            "Avg_CoxC": round(r["avg_c"], 4),
            "Avg_CoxC_SD": round(r["avg_c_sd"], 4),
            "Pooled_CoxC": round(r["pooled_c"], 4),
            "Fold_AUCs_1yr": "|".join(f"{a:.4f}" for a in r["fold_aucs"]),
            "Fold_Briers_1yr": "|".join(f"{b:.4f}" for b in r["fold_briers"]),
            "Fold_CoxCs": "|".join(f"{c:.4f}" for c in r["fold_cs"]),
        }
        all_rows.append(row)

        log(f"  {mname:25s} [{row['变量数']:2d}]  "
            f"AUC@1yr={r['avg_auc']:.4f}+/-{r['avg_auc_sd']:.4f}  "
            f"PoolAUC={r['pooled_auc']:.4f}  "
            f"AvgC={r['avg_c']:.4f}  PoolC={r['pooled_c']:.4f}  "
            f"Brier={r['avg_brier']:.4f}  (N_lm={r['n_landmark']})")

# ── Save ───────────────────────────────────────────────────────────────────────
out_df = pd.DataFrame(all_rows)
out_df.to_csv(OUT_DIR / "final_performance.csv", index=False)
log(f"\nSaved -> final_performance.csv")

# ── Per-model per-endpoint CSVs ────────────────────────────────────────────────
for mname, _ in MODELS:
    mdir = OUT_DIR / mname.split("(")[0].strip().replace(" ","_")
    mdir.mkdir(exist_ok=True)
    for ep_name in SPLITS:
        ep_rows = out_df[(out_df["模型"] == mname) & (out_df["结局"] == ep_name)]
        ep_rows.to_csv(mdir / f"{ep_name}.csv", index=False)

log("Per-model CSVs saved")

# ── Print final summary ───────────────────────────────────────────────────────
print("\n" + "="*100)
print("FINAL RESULTS — Cox PH Only, Server Fold Splits")
print("AUC/Brier: Cox 1-S(365) on landmark complete-case subset")
print("C-index: Cox partial hazard on full data (handles censoring)")
print("="*100)

for ep_name in SPLITS:
    ep_rows = [r for r in all_rows if r["结局"] == ep_name]
    print(f"\n{ep_name}  (N={ep_rows[0]['N']}, events={ep_rows[0]['事件数']}, "
          f"N_landmark={ep_rows[0]['N_landmark']})")
    print(f"  {'模型':<28}  {'#var':>4}  {'AUC@1yr':>8}  {'Pool AUC':>9}  "
          f"{'Avg C':>7}  {'Pool C':>7}  {'Brier@1yr':>9}")
    print("  " + "-"*85)
    for r in ep_rows:
        print(f"  {r['模型']:<28}  {r['变量数']:4d}  {r['Avg_AUC_1yr']:8.4f}  "
              f"{r['Pooled_AUC_1yr']:9.4f}  "
              f"{r['Avg_CoxC']:7.4f}  {r['Pooled_CoxC']:7.4f}  "
              f"{r['Avg_Brier_1yr']:9.4f}")

# ── Cross-endpoint summary ───────────────────────────────────────────────────
print("\n" + "="*100)
print("CROSS-ENDPOINT COMPARISON (Pooled AUC@1yr from Cox)")
print("="*100)
print(f"\n  {'Model':<28}", end="")
for ep_name in SPLITS:
    print(f"  {ep_name:>8}", end="")
print()
print("  " + "-"*70)
for mname, _ in MODELS:
    print(f"  {mname:<28}", end="")
    for ep_name in SPLITS:
        r = [x for x in all_rows if x["结局"]==ep_name and x["模型"]==mname][0]
        print(f"  {r['Pooled_AUC_1yr']:8.4f}", end="")
    print()

print("\n" + "="*100)
print("CROSS-ENDPOINT COMPARISON (Pooled Cox C-statistic)")
print("="*100)
print(f"\n  {'Model':<28}", end="")
for ep_name in SPLITS:
    print(f"  {ep_name:>8}", end="")
print()
print("  " + "-"*70)
for mname, _ in MODELS:
    print(f"  {mname:<28}", end="")
    for ep_name in SPLITS:
        r = [x for x in all_rows if x["结局"]==ep_name and x["模型"]==mname][0]
        print(f"  {r['Pooled_CoxC']:8.4f}", end="")
    print()

print("\n" + "="*100)
log("Done.")

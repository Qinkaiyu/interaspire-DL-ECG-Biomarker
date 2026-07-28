"""
Table 2: Model performance summary for CVD Composite 4.
Includes C-index, AUC, 95% CI, delta C-index, and p-values for incremental models.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index

BASE = Path(".")
OOF_M04 = BASE / "results" / "oof_predictions" / "oof_CVD4.csv"
OOF_M5 = BASE / "model5_cvd_mace4" / "E2E_MultiTask_32_2_phase_cvd_mace4_1yr_oof.csv"
OUT_DIR = BASE / "results" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOT = 1000
SEED = 42
LANDMARK_T = 365

# ── Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv(OOF_M04)
evt_col = "event_CVD_Composite_4"
time_col = "time_Composite_4"

df["landmark"] = (df[time_col] >= LANDMARK_T) | (df[evt_col] == 1)
df["y_1yr"] = ((df[evt_col] == 1) & (df[time_col] <= LANDMARK_T)).astype(float)

# Merge Model 5
m5 = pd.read_csv(OOF_M5)
# Per-fold z-score for Model 5
risk_norm = m5["oof_log_risk"].copy()
for fold in m5["test_fold"].unique():
    mask = m5["test_fold"] == fold
    v = m5.loc[mask, "oof_log_risk"]
    risk_norm.loc[mask] = (v - v.mean()) / v.std()
m5["Model_5_cox_risk"] = risk_norm.values
df = df.merge(m5[["STUDYID", "Model_5_cox_risk"]], on="STUDYID", how="left")

# Model 5 probability (using rank-based mapping for AUC)
df["Model_5_cox_prob"] = df["Model_5_cox_risk"]

print(f"N={len(df)}, events={int(df[evt_col].sum())}")

# ── Model definitions ───────────────────────────────────────────────────────
MODELS = [
    ("Model_0", "Model 0: Age + Sex", 2),
    ("Model_1", "Model 1: + CV risk factors", 9),
    ("Model_2", "Model 2: + Medical history & Meds", 18),
    ("Model_3", "Model 3: + Index event type", 22),
    ("Model_4", "Model 4: + ECG biomarkers", 25),
    ("Model_5", "Model 5: DL-ECG", 1),
]


# ── Bootstrap functions ─────────────────────────────────────────────────────
def boot_metrics(df_sub, risk_col, n_boot=N_BOOT, seed=SEED):
    """Bootstrap AUC and C-index with 95% CI."""
    lm = df_sub[df_sub["landmark"]].copy()
    y_true_auc = lm["y_1yr"].values
    y_score_auc = lm[risk_col].values

    y_time = df_sub[time_col].values
    y_event = df_sub[evt_col].values
    y_risk = df_sub[risk_col].values

    rng = np.random.RandomState(seed)
    aucs, cs = [], []
    n_lm, n_all = len(lm), len(df_sub)

    for _ in range(n_boot):
        # AUC bootstrap on landmark subset
        idx_lm = rng.choice(n_lm, n_lm, replace=True)
        yt, ys = y_true_auc[idx_lm], y_score_auc[idx_lm]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))

        # C-index bootstrap on full data
        idx_all = rng.choice(n_all, n_all, replace=True)
        try:
            c = concordance_index(y_time[idx_all], -y_risk[idx_all], y_event[idx_all])
            cs.append(c)
        except Exception:
            pass

    auc_mean = np.mean(aucs)
    auc_lo, auc_hi = np.percentile(aucs, [2.5, 97.5])
    c_mean = np.mean(cs)
    c_lo, c_hi = np.percentile(cs, [2.5, 97.5])

    return {
        "auc": auc_mean, "auc_lo": auc_lo, "auc_hi": auc_hi,
        "c": c_mean, "c_lo": c_lo, "c_hi": c_hi,
        "aucs_boot": np.array(aucs), "cs_boot": np.array(cs),
    }


def delta_pvalue(boot_a, boot_b):
    """P-value for delta C-index (paired bootstrap)."""
    n = min(len(boot_a), len(boot_b))
    delta = boot_b[:n] - boot_a[:n]
    # Two-sided p-value
    p = 2 * min(np.mean(delta <= 0), np.mean(delta >= 0))
    return p


# ── Compute all metrics ─────────────────────────────────────────────────────
valid = df.dropna(subset=["Model_5_cox_risk"]).copy()
print(f"Valid for all models: N={len(valid)}")

results = []
prev_boot = None
prev_name = None

for model_key, display, n_vars in MODELS:
    risk_col = f"{model_key}_cox_risk"
    if risk_col not in valid.columns:
        # Model 5 uses different column name
        risk_col = f"{model_key}_cox_risk"

    sub = valid.dropna(subset=[risk_col])
    metrics = boot_metrics(sub, risk_col)

    row = {
        "Model": display,
        "N_vars": n_vars,
        "AUC": f"{metrics['auc']:.3f} ({metrics['auc_lo']:.3f}-{metrics['auc_hi']:.3f})",
        "C-index": f"{metrics['c']:.3f} ({metrics['c_lo']:.3f}-{metrics['c_hi']:.3f})",
        "auc_val": metrics["auc"],
        "c_val": metrics["c"],
    }

    # Delta C-index vs previous model
    if prev_boot is not None:
        delta_c = metrics["c"] - prev_c
        p_val = delta_pvalue(prev_boot, metrics["cs_boot"])
        row["Delta C-index"] = f"{delta_c:+.4f}"
        row["P (vs prev)"] = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
    else:
        row["Delta C-index"] = "ref"
        row["P (vs prev)"] = "-"

    # Delta vs Model 0
    if model_key != "Model_0":
        delta_c0 = metrics["c"] - results[0]["c_val"]
        p_c0 = delta_pvalue(first_boot, metrics["cs_boot"])
        row["Delta vs M0"] = f"{delta_c0:+.4f}"
        row["P (vs M0)"] = f"{p_c0:.4f}" if p_c0 >= 0.0001 else "<0.0001"
    else:
        row["Delta vs M0"] = "ref"
        row["P (vs M0)"] = "-"
        first_boot = metrics["cs_boot"]

    prev_boot = metrics["cs_boot"]
    prev_c = metrics["c"]
    results.append(row)
    print(f"  {display}: AUC={metrics['auc']:.3f}, C={metrics['c']:.3f}")

# ── Format and save ─────────────────────────────────────────────────────────
table2 = pd.DataFrame(results)
display_cols = ["Model", "N_vars", "AUC", "C-index", "Delta C-index", "P (vs prev)", "Delta vs M0", "P (vs M0)"]
table2 = table2[display_cols]
table2.columns = ["Model", "Variables", "AUC (95% CI)", "C-index (95% CI)",
                   "ΔC-index (vs prev)", "P-value", "ΔC-index (vs M0)", "P-value (vs M0)"]

table2.to_csv(OUT_DIR / "table2_model_performance_CVD4.csv", index=False)
print(f"\nSaved: {OUT_DIR / 'table2_model_performance_CVD4.csv'}")
print("\n" + table2.to_string(index=False))

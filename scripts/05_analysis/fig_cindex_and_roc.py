"""
Figure 3: Two panels (Models 0–5)
  A) ROC curves — per-fold ROC averaged, with mean AUC (95% CI from folds) in legend
  B) C-index dot plot — mean-fold C-index with 95% CI from folds (AIRE Fig. 4B style)

AUC computed on landmark complete-case subset: (time >= 365) | (event == 1).
Uses Cox 1-S(365) probabilities for Models 0-4.

NOTE: Model 5 (DL) predictions have different scales across folds,
so we use mean-fold metrics (not pooled) for fair comparison.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import re
from sklearn.metrics import roc_curve, roc_auc_score
from lifelines.utils import concordance_index
from scipy import stats as sp_stats

BASE = Path(".")
OOF_M04 = BASE / "results" / "oof_predictions" / "oof_CVD4.csv"
OOF_M5 = BASE / "model5_cvd_mace4" / "E2E_MultiTask_32_2_phase_cvd_mace4_1yr_oof.csv"
TABLE2_PATH = BASE / "results" / "manuscript_final" / "main_tables" / "table2_model_performance_CVD4.csv"
OUT_DIR = BASE / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOT = 1000
RANDOM_STATE = 42

# ── Load & merge ─────────────────────────────────────────────────────────────
df04 = pd.read_csv(OOF_M04)
df5 = pd.read_csv(OOF_M5)
df5 = df5.rename(columns={"oof_log_risk": "Model_5_risk", "test_fold": "fold5"})

merged = df04.merge(df5[["STUDYID", "Model_5_risk", "fold5"]], on="STUDYID", how="inner")

# ── Normalize Model 5 risk per fold (z-score) to align cross-fold scales ─────
merged["Model_5_risk_raw"] = merged["Model_5_risk"].copy()
for fid in sorted(merged["fold5"].unique()):
    mask = merged["fold5"] == fid
    vals = merged.loc[mask, "Model_5_risk"]
    merged.loc[mask, "Model_5_risk"] = (vals - vals.mean()) / vals.std()
merged["Model_5_prob"] = 1 / (1 + np.exp(-merged["Model_5_risk"]))

evt_col = "event_CVD_Composite_4"
time_col = "time_Composite_4"

LANDMARK_T = 365
# Landmark complete-case mask
merged["landmark"] = (merged[time_col] >= LANDMARK_T) | (merged[evt_col] == 1)
# Binary 1yr label for AUC
merged["y_1yr"] = ((merged[evt_col] == 1) & (merged[time_col] <= LANDMARK_T)).astype(float)

print(f"Merged N={len(merged)}, events={int(merged[evt_col].sum())}, "
      f"N_landmark={int(merged['landmark'].sum())}")

# ── Model definitions ────────────────────────────────────────────────────────
MODELS = [
    {"key": "Model_0", "prob": "Model_0_cox_prob", "cox": "Model_0_cox_risk",
     "fold_col": "fold", "display": "Age + Sex"},
    {"key": "Model_1", "prob": "Model_1_cox_prob", "cox": "Model_1_cox_risk",
     "fold_col": "fold", "display": "Model 1"},
    {"key": "Model_2", "prob": "Model_2_cox_prob", "cox": "Model_2_cox_risk",
     "fold_col": "fold", "display": "Model 2"},
    {"key": "Model_3", "prob": "Model_3_cox_prob", "cox": "Model_3_cox_risk",
     "fold_col": "fold", "display": "Model 3"},
    {"key": "Model_4", "prob": "Model_4_cox_prob", "cox": "Model_4_cox_risk",
     "fold_col": "fold", "display": "Model 4"},
    {"key": "Model_5", "prob": "Model_5_prob", "cox": "Model_5_risk",
     "fold_col": "fold5", "display": "Model 5 (DL-ECG)"},
]

COLORS = {
    "Model_0": "#8E8E93",
    "Model_1": "#007AFF",
    "Model_2": "#34C759",
    "Model_3": "#FF9500",
    "Model_4": "#AF52DE",
    "Model_5": "#FF3B30",
}

DISPLAY_TO_TABLE2 = {
    "Age + Sex": "Model 0: Age + Sex",
    "Model 1": "Model 1: + CV risk factors",
    "Model 2": "Model 2: + Medical history & Meds",
    "Model 3": "Model 3: + Index event type",
    "Model 4": "Model 4: + ECG biomarkers",
    "Model 5 (DL-ECG)": "Model 5: DL-ECG",
}


def parse_metric_ci(text):
    m = re.match(r"\s*([0-9.]+)\s*\(([0-9.]+)-([0-9.]+)\)\s*", str(text))
    if not m:
        raise ValueError(f"Could not parse metric CI: {text}")
    return float(m.group(1)), float(m.group(2)), float(m.group(3))

# ── Compute per-fold metrics ─────────────────────────────────────────────────
results = []

for m in MODELS:
    fold_col = m["fold_col"]
    prob_col = m["prob"]
    cox_col = m["cox"]
    folds = sorted(merged[fold_col].unique())

    fold_aucs, fold_cs = [], []
    fold_fprs, fold_tprs = [], []

    for f in folds:
        sub = merged[merged[fold_col] == f]
        y_f = sub["y_1yr"].values.astype(float)
        t_f = sub[time_col].values.astype(float)
        lm_f = sub["landmark"].values

        # AUC on landmark complete-case subset
        v_prob = ~sub[prob_col].isna() & lm_f
        if v_prob.sum() > 0 and len(np.unique(y_f[v_prob])) > 1:
            fold_aucs.append(roc_auc_score(y_f[v_prob], sub.loc[v_prob, prob_col].values))
            fpr_f, tpr_f, _ = roc_curve(y_f[v_prob], sub.loc[v_prob, prob_col].values)
            fold_fprs.append(fpr_f)
            fold_tprs.append(tpr_f)

        # C-index on full data
        v_cox = ~sub[cox_col].isna()
        evt_f = sub[evt_col].values.astype(float)
        if v_cox.sum() > 0:
            try:
                fold_cs.append(concordance_index(
                    np.clip(t_f[v_cox], 0.5, None),
                    -sub.loc[v_cox.values, cox_col].values,
                    evt_f[v_cox]
                ))
            except Exception:
                pass

    mean_auc = np.mean(fold_aucs)
    sd_auc = np.std(fold_aucs, ddof=1)
    ci_auc_lo = mean_auc - 1.96 * sd_auc / np.sqrt(len(fold_aucs))
    ci_auc_hi = mean_auc + 1.96 * sd_auc / np.sqrt(len(fold_aucs))

    mean_c = np.mean(fold_cs)
    sd_c = np.std(fold_cs, ddof=1)
    ci_c_lo = mean_c - 1.96 * sd_c / np.sqrt(len(fold_cs))
    ci_c_hi = mean_c + 1.96 * sd_c / np.sqrt(len(fold_cs))

    # Pooled ROC on landmark complete-case subset
    valid_all = ~merged[prob_col].isna() & merged["landmark"]
    fpr_pooled, tpr_pooled, _ = roc_curve(
        merged.loc[valid_all, "y_1yr"].values,
        merged.loc[valid_all, prob_col].values
    )

    results.append({
        "key": m["key"], "display": m["display"],
        "mean_auc": mean_auc, "auc_lo": ci_auc_lo, "auc_hi": ci_auc_hi,
        "sd_auc": sd_auc,
        "mean_c": mean_c, "c_lo": ci_c_lo, "c_hi": ci_c_hi,
        "sd_c": sd_c,
        "fold_aucs": fold_aucs, "fold_cs": fold_cs,
        "fpr": fpr_pooled, "tpr": tpr_pooled,
    })

    print(f"  {m['display']:22s}  "
          f"AUC={mean_auc:.3f}±{sd_auc:.3f} ({ci_auc_lo:.3f}-{ci_auc_hi:.3f})  "
          f"C={mean_c:.3f}±{sd_c:.3f} ({ci_c_lo:.3f}-{ci_c_hi:.3f})")

res = pd.DataFrame(results)

# ── Sync displayed values to final manuscript Table 2 ───────────────────────
if TABLE2_PATH.exists():
    table2 = pd.read_csv(TABLE2_PATH)
    for i, row in res.iterrows():
        table_name = DISPLAY_TO_TABLE2.get(row["display"])
        if not table_name:
            continue
        hit = table2.loc[table2["Model"] == table_name]
        if hit.empty:
            continue
        auc, auc_lo, auc_hi = parse_metric_ci(hit.iloc[0]["AUC (95% CI)"])
        cidx, c_lo, c_hi = parse_metric_ci(hit.iloc[0]["C-index (95% CI)"])
        res.loc[i, ["mean_auc", "auc_lo", "auc_hi"]] = [auc, auc_lo, auc_hi]
        res.loc[i, ["mean_c", "c_lo", "c_hi"]] = [cidx, c_lo, c_hi]
    print(f"Synced displayed Figure 2 metrics to {TABLE2_PATH}")
else:
    print(f"WARNING: Table 2 not found at {TABLE2_PATH}; using figure-computed metrics.")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10),
                                gridspec_kw={"height_ratios": [1.0, 0.6],
                                             "hspace": 0.45})

# ── Panel A: ROC curves ──────────────────────────────────────────────────────
for _, row in res.iterrows():
    label = (f'{row["display"]}  '
             f'AUC {row["mean_auc"]:.3f} '
             f'(95% CI {row["auc_lo"]:.3f}\u2013{row["auc_hi"]:.3f})')
    ax1.plot(row["fpr"], row["tpr"],
             color=COLORS[row["key"]], linewidth=1.3, label=label)

ax1.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)
ax1.set_xlabel("1 \u2013 Specificity", fontsize=11)
ax1.set_ylabel("Sensitivity", fontsize=11)
ax1.set_title("A   ROC curves", fontsize=14, fontweight="bold", loc="left", pad=60)
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(-0.02, 1.02)
# no equal aspect — vertical layout shares full width
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 1.01),
           ncol=2, frameon=False, columnspacing=1.0)
ax1.tick_params(labelsize=10)

# ── Panel B: C-index dot plot (AIRE style) ────────────────────────────────────
x_pos = np.arange(len(res))

for i, (_, row) in enumerate(res.iterrows()):
    yerr_lo = row["mean_c"] - row["c_lo"]
    yerr_hi = row["c_hi"] - row["mean_c"]

    ax2.errorbar(
        x_pos[i], row["mean_c"],
        yerr=[[yerr_lo], [yerr_hi]],
        fmt="o", color=COLORS[row["key"]],
        markersize=9, capsize=5, capthick=1.8,
        elinewidth=2, markeredgewidth=0, zorder=5,
        label=f'{row["display"]}  {row["mean_c"]:.3f} ({row["c_lo"]:.3f}\u2013{row["c_hi"]:.3f})',
    )

ax2.plot(x_pos, res["mean_c"].values, color="#ccc", linewidth=1, linestyle="-", zorder=2)

ax2.set_xticks(x_pos)
ax2.set_xticklabels([""] * len(x_pos))
ax2.set_ylabel("Cox model C-index", fontsize=11)
ax2.set_title("B   Incremental C-index", fontsize=14, fontweight="bold", loc="left", pad=60)
ax2.set_xlim(-0.5, len(res) - 0.5)

y_min = res["c_lo"].min() - 0.02
y_max = res["c_hi"].max() + 0.02
ax2.set_ylim(y_min, y_max)
ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.02))
ax2.tick_params(labelsize=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(True)
ax2.legend(fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 1.01),
           ncol=2, frameon=False, columnspacing=1.0)

fig.savefig(OUT_DIR / "fig2_roc_and_cindex_CVD4.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "fig2_roc_and_cindex_CVD4.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSaved -> {OUT_DIR / 'fig2_roc_and_cindex_CVD4.png'}")

# Copy to manuscript_final
import shutil
FINAL_DIR = BASE / "results" / "manuscript_final" / "main_figures"
shutil.copy2(OUT_DIR / "fig2_roc_and_cindex_CVD4.pdf", FINAL_DIR)
shutil.copy2(OUT_DIR / "fig2_roc_and_cindex_CVD4.png", FINAL_DIR)

# Save data
out_table = res[["display", "mean_auc", "auc_lo", "auc_hi", "sd_auc",
                  "mean_c", "c_lo", "c_hi", "sd_c"]].copy()
out_table.to_csv(OUT_DIR / "fig2_roc_and_cindex_CVD4_data.csv", index=False)

"""
Supplementary: Calibration curves + Decision Curve Analysis (DCA)
Model 5 probabilities are recalibrated via Platt scaling (logistic recalibration).
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

BASE = Path(".")
OOF_M04 = BASE / "results" / "oof_predictions" / "oof_CVD4.csv"
OOF_M5 = BASE / "model5_cvd_mace4" / "E2E_MultiTask_32_2_phase_cvd_mace4_1yr_oof.csv"
OUT_DIR = BASE / "results" / "supplementary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

evt_col = "event_CVD_Composite_4"

# ── Load & merge ─────────────────────────────────────────────────────────────
df04 = pd.read_csv(OOF_M04)
df5 = pd.read_csv(OOF_M5)
df5 = df5.rename(columns={"oof_log_risk": "Model_5_risk", "test_fold": "fold5"})
merged = df04.merge(df5[["STUDYID", "Model_5_risk", "fold5"]], on="STUDYID", how="inner")

# Normalize Model 5 per fold
for fid in sorted(merged["fold5"].unique()):
    mask = merged["fold5"] == fid
    vals = merged.loc[mask, "Model_5_risk"]
    merged.loc[mask, "Model_5_risk"] = (vals - vals.mean()) / vals.std()

LANDMARK_T = 365
time_col = "time_Composite_4"
merged["landmark"] = (merged[time_col] >= LANDMARK_T) | (merged[evt_col] == 1)
merged["y_1yr"] = ((merged[evt_col] == 1) & (merged[time_col] <= LANDMARK_T)).astype(float)

# Use landmark subset for calibration/DCA
lm = merged["landmark"].values
y = merged["y_1yr"].values.astype(float)

print(f"N_landmark={int(lm.sum())}")

# ── Platt scaling for Model 5 (per-fold to avoid data leakage) ───────────────
merged["Model_5_prob_cal"] = np.nan
for fid in sorted(merged["fold5"].unique()):
    test_mask = merged["fold5"] == fid
    train_mask = ~test_mask

    X_train = merged.loc[train_mask, "Model_5_risk"].values.reshape(-1, 1)
    y_train = y[train_mask]
    X_test = merged.loc[test_mask, "Model_5_risk"].values.reshape(-1, 1)

    platt = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    platt.fit(X_train, y_train)
    merged.loc[test_mask, "Model_5_prob_cal"] = platt.predict_proba(X_test)[:, 1]

m5_cal = merged["Model_5_prob_cal"].values
print(f"Model 5 calibrated prob: mean={np.nanmean(m5_cal):.4f}, "
      f"range=[{np.nanmin(m5_cal):.4f}, {np.nanmax(m5_cal):.4f}]")
print(f"Model 3 prob:            mean={merged['Model_3_cox_prob'].mean():.4f}")
print(f"Actual event rate (lm):  {y[lm].mean():.4f}")

MODELS = [
    ("Model 3", "Model_3_cox_prob", "#FF9500"),
    ("Model 4", "Model_4_cox_prob", "#AF52DE"),
    ("Model 5", "Model_5_prob_cal", "#FF3B30"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# ── Panel A: Calibration (landmark subset) ───────────────────────────────────
for mname, prob_col, color in MODELS:
    prob = merged[prob_col].values
    valid = ~np.isnan(prob) & lm

    fraction_pos, mean_pred = calibration_curve(
        y[valid], prob[valid], n_bins=10, strategy="quantile"
    )
    ax1.plot(mean_pred, fraction_pos, "o-", color=color, label=mname,
             markersize=6, linewidth=1.5)

# Perfect calibration line
max_val = 0.20
ax1.plot([0, max_val], [0, max_val], "k--", alpha=0.4, linewidth=1, label="Perfect")
ax1.set_xlabel("Predicted probability", fontsize=11)
ax1.set_ylabel("Observed event rate", fontsize=11)
ax1.set_title("A  Calibration", fontsize=12, fontweight="bold", loc="left")
ax1.legend(fontsize=9, frameon=True, fancybox=False, edgecolor="#ccc")
ax1.grid(True, alpha=0.15)
ax1.set_xlim(0, max_val)
ax1.set_ylim(0, max_val)
ax1.set_aspect("equal")
ax1.tick_params(labelsize=10)

# ── Panel B: Decision Curve Analysis ─────────────────────────────────────────
thresholds = np.arange(0.005, 0.20, 0.002)
prevalence = y.mean()

# Treat All
nb_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
ax2.plot(thresholds * 100, nb_all, "k-", linewidth=1, alpha=0.5, label="Treat all")
ax2.axhline(0, color="grey", linewidth=0.8, alpha=0.4, label="Treat none")

for mname, prob_col, color in MODELS:
    prob = merged[prob_col].values
    valid = ~np.isnan(prob) & lm
    y_v = y[valid]
    p_v = prob[valid]
    n = len(y_v)

    nb_list = []
    for t in thresholds:
        tp = ((p_v >= t) & (y_v == 1)).sum()
        fp = ((p_v >= t) & (y_v == 0)).sum()
        nb = tp / n - fp / n * t / (1 - t)
        nb_list.append(nb)

    ax2.plot(thresholds * 100, nb_list, color=color, linewidth=2, label=mname)

ax2.set_xlabel("Threshold probability (%)", fontsize=11)
ax2.set_ylabel("Net benefit", fontsize=11)
ax2.set_title("B  Decision Curve Analysis", fontsize=12, fontweight="bold", loc="left")
ax2.legend(fontsize=9, loc="upper right", frameon=True, fancybox=False, edgecolor="#ccc")
ax2.grid(True, alpha=0.15)
ax2.set_xlim(0.5, 20)
ax2.set_ylim(-0.01, max(nb_all[0] * 1.1, 0.06))
ax2.tick_params(labelsize=10)

# Annotation
ax2.text(0.5, 0.5, "Higher net benefit = better\nclinical utility at that threshold",
         transform=ax2.transAxes, fontsize=7.5, ha="center", va="center",
         color="#666", fontstyle="italic",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ddd", alpha=0.8))

plt.tight_layout(w_pad=3)
fig.savefig(OUT_DIR / "supp_calibration_dca_CVD4.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "supp_calibration_dca_CVD4.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved -> {OUT_DIR / 'supp_calibration_dca_CVD4.png'}")

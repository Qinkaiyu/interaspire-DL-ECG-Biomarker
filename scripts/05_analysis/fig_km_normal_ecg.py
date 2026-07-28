"""
Fig. 5: KM curves in Normal ECG subset (Raghunath 2020 style)
Key question: Can Model 5 still discriminate within patients whose ECG is read as normal?
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

BASE = Path(".")
DATA_PATH = BASE / "data" / "INTERASPIRE_analysis_dataset.xlsx"
OOF_M04 = BASE / "results" / "oof_predictions" / "oof_CVD4.csv"
OOF_M5 = BASE / "model5_cvd_mace4" / "E2E_MultiTask_32_2_phase_cvd_mace4_1yr_oof.csv"
OUT_DIR = BASE / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

evt_col = "event_CVD_Composite_4"
time_col = "time_Composite_4"

# ── Load data ────────────────────────────────────────────────────────────────
raw = pd.read_excel(DATA_PATH, engine="openpyxl")
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


# Merge ECG columns from raw data
ecg_cols_needed = [
    "STUDYID", "Sinus Rhythm", "Atrial Fibrillation", "Atrial Flutter",
    "Supraventricular Tachycardia", "RBBB", "LBBB",
    "1 AV Block", "2 AV Block", "3 AV Block",
    "Left Axis Deviation", "Right Axis Deviation", "Q Wave",
    "ST Elevation", "ST Depression", "T Wave Inversion",
    "Ischaemic", "LVH", "RVH", "QT Prolongation",
    "MI (Old)", "MI(Acute)",
]
ecg_cols_available = [c for c in ecg_cols_needed if c in raw.columns]
merged = merged.merge(raw[ecg_cols_available], on="STUDYID", how="left")

# ── Define Normal ECG (Strict) ───────────────────────────────────────────────
abnormal_cols = [c for c in ecg_cols_available if c not in ["STUDYID", "Sinus Rhythm"]]

has_ecg = merged["Sinus Rhythm"].notna()
is_normal = (merged["Sinus Rhythm"] == "Yes")
for col in abnormal_cols:
    is_normal = is_normal & ((merged[col] == "No") | merged[col].isna())

merged["ecg_status"] = "Abnormal"
merged.loc[has_ecg & is_normal, "ecg_status"] = "Normal"

print("=== ECG status ===")
for status in ["Normal", "Abnormal"]:
    sub = merged[merged["ecg_status"] == status]
    ev = int(sub[evt_col].sum())
    print(f"  {status:10s}: N={len(sub):5d}, events={ev:3d}, rate={ev/len(sub)*100:.1f}%")

# ── Risk stratification within Normal ECG subset ─────────────────────────────
df_normal = merged[merged["ecg_status"] == "Normal"].copy()
df_abnormal = merged[merged["ecg_status"] == "Abnormal"].copy()

print(f"\n=== Normal ECG subset: N={len(df_normal)}, events={int(df_normal[evt_col].sum())} ===")

# Equal-width tertile on Model 5 risk
risk = df_normal["Model_5_risk"].values
r_min, r_max = risk.min(), risk.max()
width = (r_max - r_min) / 3
cuts = [r_min, r_min + width, r_min + 2 * width, r_max]

df_normal["risk_group"] = pd.cut(
    df_normal["Model_5_risk"], bins=cuts,
    labels=["Low", "Intermediate", "High"], include_lowest=True
)

print(f"\nEqual-width tertile within Normal ECG:")
print(f"{'Group':<18s} {'N':>6s} {'Events':>7s} {'Rate':>7s}")
print("-" * 40)
for g in ["Low", "Intermediate", "High"]:
    sub = df_normal[df_normal["risk_group"] == g]
    n = len(sub)
    ev = int(sub[evt_col].sum())
    rate = ev / n * 100 if n > 0 else 0
    print(f"{g:<18s} {n:6d} {ev:7d} {rate:6.1f}%")

# Log-rank
lr = multivariate_logrank_test(df_normal[time_col], df_normal["risk_group"], df_normal[evt_col])
print(f"\nLog-rank: chi2={lr.test_statistic:.1f}, p={lr.p_value:.2e}")

# Mean-fold C-index within Normal ECG
fold_cs_normal, fold_aucs_normal = [], []
fold_cs_abnormal, fold_aucs_abnormal = [], []

for fid in sorted(df_normal["fold5"].unique()):
    # Normal
    sub_n = df_normal[df_normal["fold5"] == fid]
    y_n = sub_n[evt_col].values.astype(float)
    t_n = sub_n[time_col].values.astype(float)
    r_n = sub_n["Model_5_risk"].values
    if len(np.unique(y_n)) > 1:
        fold_cs_normal.append(concordance_index(np.clip(t_n, 0.5, None), -r_n, y_n))
        fold_aucs_normal.append(roc_auc_score(y_n, 1/(1+np.exp(-r_n))))

    # Abnormal
    sub_a = df_abnormal[df_abnormal["fold5"] == fid]
    y_a = sub_a[evt_col].values.astype(float)
    t_a = sub_a[time_col].values.astype(float)
    r_a = sub_a["Model_5_risk"].values
    if len(np.unique(y_a)) > 1:
        fold_cs_abnormal.append(concordance_index(np.clip(t_a, 0.5, None), -r_a, y_a))
        fold_aucs_abnormal.append(roc_auc_score(y_a, 1/(1+np.exp(-r_a))))

print(f"\nModel 5 performance:")
print(f"  Normal ECG:   Mean C={np.mean(fold_cs_normal):.3f}±{np.std(fold_cs_normal):.3f}  "
      f"Mean AUC={np.mean(fold_aucs_normal):.3f}±{np.std(fold_aucs_normal):.3f}")
print(f"  Abnormal ECG: Mean C={np.mean(fold_cs_abnormal):.3f}±{np.std(fold_cs_abnormal):.3f}  "
      f"Mean AUC={np.mean(fold_aucs_abnormal):.3f}±{np.std(fold_aucs_abnormal):.3f}")

# ── Also do Abnormal ECG subset ──────────────────────────────────────────────
risk_a = df_abnormal["Model_5_risk"].values
r_min_a, r_max_a = risk_a.min(), risk_a.max()
width_a = (r_max_a - r_min_a) / 3
cuts_a = [r_min_a, r_min_a + width_a, r_min_a + 2 * width_a, r_max_a]

df_abnormal["risk_group"] = pd.cut(
    df_abnormal["Model_5_risk"], bins=cuts_a,
    labels=["Low", "Intermediate", "High"], include_lowest=True
)

print(f"\n=== Abnormal ECG subset: N={len(df_abnormal)}, events={int(df_abnormal[evt_col].sum())} ===")
print(f"{'Group':<18s} {'N':>6s} {'Events':>7s} {'Rate':>7s}")
print("-" * 40)
for g in ["Low", "Intermediate", "High"]:
    sub = df_abnormal[df_abnormal["risk_group"] == g]
    n = len(sub)
    ev = int(sub[evt_col].sum())
    rate = ev / n * 100 if n > 0 else 0
    print(f"{g:<18s} {n:6d} {ev:7d} {rate:6.1f}%")

lr_a = multivariate_logrank_test(df_abnormal[time_col], df_abnormal["risk_group"], df_abnormal[evt_col])
print(f"Log-rank: chi2={lr_a.test_statistic:.1f}, p={lr_a.p_value:.2e}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE: 2 panels — Normal ECG (left) + Abnormal ECG (right)
# ═══════════════════════════════════════════════════════════════════════════════
COLORS = {"Low": "#2ecc71", "Intermediate": "#3498db", "High": "#e74c3c"}

# Type sizes; legend: FS_LEGEND + ax.legend(...) in the loop below.
FS_TITLE = 16
FS_AXIS = 14
FS_TICK = 14
FS_LEGEND = 14
FS_PVAL = 14

# Two portrait panels (per-panel size matches fig_km_curves.py)
FIG_W_PANEL, FIG_H = 6.8, 9.2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * FIG_W_PANEL, FIG_H))

for ax, df_sub, title_suffix, lr_res in [
    (ax1, df_normal, "Normal ECG", lr),
    (ax2, df_abnormal, "Abnormal ECG", lr_a),
]:
    for grp in ["Low", "Intermediate", "High"]:
        sub = df_sub[df_sub["risk_group"] == grp]
        kmf = KaplanMeierFitter()
        kmf.fit(sub[time_col], event_observed=sub[evt_col], label=grp)
        n = len(sub)
        ev = int(sub[evt_col].sum())
        rate = ev / n * 100 if n > 0 else 0
        kmf.plot_survival_function(
            ax=ax, color=COLORS[grp], linewidth=2,
            ci_alpha=0.12, ci_show=True,
            label=f"{grp} (n={n}, {ev} events, {rate:.1f}%)"
        )

    # P-value annotation
    p_text = "p < 0.001" if lr_res.p_value < 0.001 else f"p = {lr_res.p_value:.3f}"
    ax.text(0.98, 0.95, f"Log-rank {p_text}",
            transform=ax.transAxes, fontsize=FS_PVAL,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="#ccc"))

    n_total = len(df_sub)
    ev_total = int(df_sub[evt_col].sum())
    ax.set_title(f"{title_suffix} (N={n_total}, {ev_total} events)",
                 fontsize=FS_TITLE, fontweight="bold", pad=55)
    ax.set_xlabel("Time (days)", fontsize=FS_AXIS)
    ax.set_ylabel("Survival probability", fontsize=FS_AXIS)
    ax.set_xlim(0, 370)
    ax.set_ylim(0.82, 1.01)
    ax.tick_params(labelsize=FS_TICK)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 0.96),
        frameon=False,
        ncol=1,
        handlelength=2.2,
        borderaxespad=0.0,
        prop={"size": FS_LEGEND, "weight": "bold"},
    )

plt.tight_layout(w_pad=3)
fig.savefig(OUT_DIR / "fig4_km_normal_ecg_CVD4.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "fig4_km_normal_ecg_CVD4.png", dpi=300, bbox_inches="tight")
plt.close()

# Copy to manuscript_final
import shutil
FINAL_DIR = BASE / "results" / "manuscript_final" / "main_figures"
shutil.copy2(OUT_DIR / "fig4_km_normal_ecg_CVD4.pdf", FINAL_DIR)
shutil.copy2(OUT_DIR / "fig4_km_normal_ecg_CVD4.png", FINAL_DIR)

print(f"\nSaved -> {OUT_DIR / 'fig4_km_normal_ecg_CVD4.png'}")

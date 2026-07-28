"""
Figure 3: Kaplan-Meier curves by Model 5 risk groups (0-365 days).
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

BASE = Path(".")
OOF_M04 = BASE / "results" / "oof_predictions" / "oof_CVD4.csv"
OOF_M5 = BASE / "model5_cvd_mace4" / "E2E_MultiTask_32_2_phase_cvd_mace4_1yr_oof.csv"
OUT_DIR = BASE / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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


evt_col = "event_CVD_Composite_4"
time_col = "time_Composite_4"

# ── Tertile stratification (equal-width on risk value range) ──────────────────
risk_col = "Model_5_risk"
r_min = merged[risk_col].min()
r_max = merged[risk_col].max()
width = (r_max - r_min) / 3
cuts = [r_min, r_min + width, r_min + 2 * width, r_max]

merged["risk_group"] = pd.cut(
    merged[risk_col],
    bins=cuts,
    labels=["Low", "Intermediate", "High"],
    include_lowest=True,
)

print(f"Equal-width cutpoints: {[f'{c:.3f}' for c in cuts]}")
print(f"\n{'Group':<22s} {'N':>6s} {'Events':>7s} {'Rate':>7s}")
print("-" * 45)
for grp in ["Low", "Intermediate", "High"]:
    sub = merged[merged["risk_group"] == grp]
    n = len(sub)
    ev = int(sub[evt_col].sum())
    rate = ev / n * 100
    print(f"{grp:<22s} {n:6d} {ev:7d} {rate:6.1f}%")

total_ev = int(merged[evt_col].sum())
print(f"{'Total':<22s} {len(merged):6d} {total_ev:7d} {total_ev/len(merged)*100:6.1f}%")

# ── Log-rank test ────────────────────────────────────────────────────────────
lr_result = multivariate_logrank_test(
    merged[time_col], merged["risk_group"], merged[evt_col]
)
print(f"\nLog-rank test: chi2={lr_result.test_statistic:.1f}, p={lr_result.p_value:.2e}")

# ── KM plot ──────────────────────────────────────────────────────────────────
COLORS = {
    "Low": "#2ecc71",
    "Intermediate": "#3498db",
    "High": "#e74c3c",
}

# Type sizes for KM panels; legend text: adjust FS_LEGEND and ax.legend(...) below.
FS_TITLE = 16
FS_AXIS = 14
FS_TICK = 14
FS_LEGEND = 14
FS_PVAL = 14

# Portrait canvas (height > width); adjust here if you want taller/narrower.
FIG_W, FIG_H = 6.8, 9.2
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

kmfs = {}

for grp in ["Low", "Intermediate", "High"]:
    sub = merged[merged["risk_group"] == grp]
    kmf = KaplanMeierFitter()
    kmf.fit(sub[time_col], event_observed=sub[evt_col], label=grp)
    kmfs[grp] = (kmf, sub)

    # Plot manually to control legend properly
    times = kmf.survival_function_.index.values
    surv = kmf.survival_function_.values.flatten()
    ci_lo = kmf.confidence_interval_survival_function_.iloc[:, 0].values
    ci_hi = kmf.confidence_interval_survival_function_.iloc[:, 1].values

    ax.step(times, surv, where="post", color=COLORS[grp], linewidth=2)
    ax.fill_between(times, ci_lo, ci_hi, step="post", color=COLORS[grp], alpha=0.12)

# Formatting
ax.set_xlabel("Time (days)", fontsize=FS_AXIS)
ax.set_ylabel("Survival probability", fontsize=FS_AXIS)
ax.set_xlim(0, 370)
ax.set_ylim(0.82, 1.01)
ax.tick_params(labelsize=FS_TICK)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# P-value annotation
if lr_result.p_value < 0.001:
    p_text = "p < 0.001"
else:
    p_text = f"p = {lr_result.p_value:.3f}"
ax.text(0.98, 0.95, f"Log-rank {p_text}",
        transform=ax.transAxes, fontsize=FS_PVAL,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="#ccc"))

# Custom legend with proper line markers
from matplotlib.lines import Line2D
legend_handles = []
for grp in ["Low", "Intermediate", "High"]:
    _, sub = kmfs[grp]
    n = len(sub)
    ev = int(sub[evt_col].sum())
    rate = ev / n * 100
    handle = Line2D([0], [0], color=COLORS[grp], linewidth=2,
                    label=f"{grp} (n={n}, {ev} events, {rate:.1f}%)")
    legend_handles.append(handle)

n_tot = len(merged)
ev_tot = int(merged[evt_col].sum())
ax.set_title(
    f"Overall CVD Composite 1 (N={n_tot}, {ev_tot} events)",
    fontsize=FS_TITLE,
    fontweight="bold",
    pad=55,
)
ax.legend(
    handles=legend_handles,
    loc="lower left",
    bbox_to_anchor=(0.0, 0.96),
    frameon=False,
    ncol=1,
    handlelength=2.2,
    borderaxespad=0.0,
    prop={"size": FS_LEGEND, "weight": "bold"},
)

plt.tight_layout()

fig.savefig(OUT_DIR / "fig3_km_quartile_CVD4.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "fig3_km_quartile_CVD4.png", dpi=300, bbox_inches="tight")
plt.close()

# Copy to manuscript_final
import shutil
FINAL_DIR = BASE / "results" / "manuscript_final" / "main_figures"
shutil.copy2(OUT_DIR / "fig3_km_quartile_CVD4.pdf", FINAL_DIR)
shutil.copy2(OUT_DIR / "fig3_km_quartile_CVD4.png", FINAL_DIR)

print(f"\nSaved -> {OUT_DIR / 'fig3_km_quartile_CVD4.png'}")

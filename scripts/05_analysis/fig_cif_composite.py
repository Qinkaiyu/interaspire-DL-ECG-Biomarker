"""
Fig 4b: Cumulative Incidence curves by DL-ECG risk tertile
for CVD_Composite_4 (1-year).

This is 1 - KM survival, plotted as cumulative incidence (y-axis going up).
Note: For the composite endpoint, there are no competing risks (all 4 events
count), so CIF = 1 - KM. This is just a different visual presentation.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from matplotlib.transforms import blended_transform_factory

BASE = Path(".")
OOF_M04 = BASE / "results" / "oof_predictions" / "oof_CVD4.csv"
OOF_M5 = BASE / "model5_cvd_mace4" / "E2E_MultiTask_32_2_phase_cvd_mace4_1yr_oof.csv"
OUT_DIR = BASE / "results" / "figures"

# ── Load & merge ─────────────────────────────────────────────────────────────
df04 = pd.read_csv(OOF_M04)
df5 = pd.read_csv(OOF_M5)
df5 = df5.rename(columns={"oof_log_risk": "Model_5_risk", "test_fold": "fold5"})
merged = df04.merge(df5[["STUDYID", "Model_5_risk", "fold5"]], on="STUDYID", how="inner")

# Per-fold z-normalize
for fid in sorted(merged["fold5"].unique()):
    mask = merged["fold5"] == fid
    vals = merged.loc[mask, "Model_5_risk"]
    merged.loc[mask, "Model_5_risk"] = (vals - vals.mean()) / vals.std()

evt_col = "event_CVD_Composite_4"
time_col = "time_Composite_4"

# Equal-width tertile
r_min, r_max = merged["Model_5_risk"].min(), merged["Model_5_risk"].max()
width = (r_max - r_min) / 3
merged["risk_group"] = pd.cut(
    merged["Model_5_risk"],
    bins=[r_min, r_min + width, r_min + 2 * width, r_max],
    labels=["Low", "Intermediate", "High"],
    include_lowest=True,
)

# Log-rank
lr = multivariate_logrank_test(merged[time_col], merged["risk_group"], merged[evt_col])

# ── Plot CIF (1 - survival) ─────────────────────────────────────────────────
COLORS = {"Low": "#2ecc71", "Intermediate": "#3498db", "High": "#e74c3c"}

fig, ax = plt.subplots(figsize=(8, 6))

kmfs = {}
for grp in ["Low", "Intermediate", "High"]:
    sub = merged[merged["risk_group"] == grp]
    kmf = KaplanMeierFitter()
    kmf.fit(sub[time_col], event_observed=sub[evt_col], label=grp)
    kmfs[grp] = (kmf, sub)

    # CIF = 1 - survival
    times = kmf.survival_function_.index.values
    surv = kmf.survival_function_.values.flatten()
    cif = 1 - surv

    # CI band
    ci_lo = 1 - kmf.confidence_interval_survival_function_.iloc[:, 1].values  # upper surv -> lower CIF
    ci_hi = 1 - kmf.confidence_interval_survival_function_.iloc[:, 0].values  # lower surv -> upper CIF

    n = len(sub)
    ev = int(sub[evt_col].sum())
    rate = ev / n * 100
    label = f"{grp} (n={n}, {ev} events, {rate:.1f}%)"

    ax.plot(times, cif, color=COLORS[grp], linewidth=2, label=label)
    ax.fill_between(times, ci_lo, ci_hi, color=COLORS[grp], alpha=0.12)

ax.set_xlabel("Time (days)", fontsize=12)
ax.set_ylabel("Cumulative incidence", fontsize=12)
ax.set_xlim(0, 370)
ax.set_ylim(0, 0.22)
ax.grid(True, alpha=0.15)
ax.tick_params(labelsize=10)

# P-value
p_text = "p < 0.001" if lr.p_value < 0.001 else f"p = {lr.p_value:.3f}"
ax.text(0.02, 0.97, f"Log-rank {p_text}\n(chi2={lr.test_statistic:.1f})",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="#ccc"))

ax.legend(loc="upper left", fontsize=9, frameon=True, fancybox=False,
          edgecolor="#ccc", framealpha=0.95)

ax.set_title("Cumulative incidence by DL-ECG risk tertile\n(CVD Composite-4, 1-year)",
             fontsize=12, fontweight="bold")

# ── Number at risk table ────────────────────────────────────────────────────
n_at_risk_times = [0, 60, 120, 180, 240, 300, 365]
groups_order = ["High", "Intermediate", "Low"]

trans = blended_transform_factory(ax.transData, ax.transAxes)
row_gap = 0.055
y_base = -0.14

ax.text(-30, y_base + row_gap, "Number at risk", fontsize=9, fontweight="bold",
        transform=trans, va="center", ha="left")

for i, grp in enumerate(groups_order):
    _, sub = kmfs[grp]
    y_pos = y_base - i * row_gap
    ax.text(-30, y_pos, grp, fontsize=8.5, fontweight="bold",
            color=COLORS[grp], transform=trans, va="center", ha="left")
    for tp in n_at_risk_times:
        n_r = int(((sub[time_col] > tp) | ((sub[time_col] == tp) & (sub[evt_col] == 0))).sum())
        ax.text(tp, y_pos, str(n_r), fontsize=8, transform=trans, va="center", ha="center")

fig.subplots_adjust(bottom=0.30)

out_path = OUT_DIR / "fig4b_cif_composite_CVD4"
fig.savefig(str(out_path) + ".pdf", dpi=300, bbox_inches="tight")
fig.savefig(str(out_path) + ".png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved -> {out_path}.png")

"""
Individual KM curves for each component endpoint of CVD_Composite_4,
stratified by DL-ECG risk tertile. Each component gets its own figure
with number-at-risk table.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH = Path("data/INTERASPIRE_analysis_dataset.xlsx")
M5_PATH = Path("model5_cvd_mace4/E2E_MultiTask_32_2_phase_cvd_mace4_1yr_oof.csv")
OUT_DIR = Path("results/component_endpoints")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load & merge ────────────────────────────────────────────────────────────
df = pd.read_excel(DATA_PATH, engine="openpyxl")

oof5 = pd.read_csv(M5_PATH)
risk_norm = oof5["oof_log_risk"].copy()
for fold in oof5["test_fold"].unique():
    m = oof5["test_fold"] == fold
    v = oof5.loc[m, "oof_log_risk"]
    risk_norm.loc[m] = (v - v.mean()) / v.std()
oof5["risk_norm"] = risk_norm

merged = df.merge(oof5[["STUDYID", "risk_norm"]], on="STUDYID", how="inner")

# Derive component events from first-event logic
first_comp = merged["first_component_Composite_4"].fillna("")
merged["comp_cv_death"] = ((first_comp == "SURVIVAL") & (merged["cvd_death_flag"] == 1)).astype(int)
merged["comp_mi"] = (first_comp == "HOSPAMI").astype(int)
merged["comp_hf"] = (first_comp == "HOSPHF").astype(int)
merged["comp_stroke"] = (first_comp == "HOSPSTROKE").astype(int)

TIME_COL = "time_Composite_4"

# Equal-width tertile
rmin, rmax = merged["risk_norm"].min(), merged["risk_norm"].max()
t1 = rmin + (rmax - rmin) / 3
t2 = rmin + 2 * (rmax - rmin) / 3
merged["risk_group"] = pd.cut(
    merged["risk_norm"],
    bins=[rmin - 0.001, t1, t2, rmax + 0.001],
    labels=["Low", "Intermediate", "High"],
)

COMPONENTS = [
    ("CV death", "comp_cv_death", 24),
    ("Myocardial infarction", "comp_mi", 67),
    ("Heart failure", "comp_hf", 52),
    ("Stroke", "comp_stroke", 24),
]

COLORS = {"Low": "#2ca02c", "Intermediate": "#ff7f0e", "High": "#d62728"}
RISK_TABLE_TIMES = [0, 60, 120, 180, 240, 300, 365]

for comp_name, evt_col, n_evt in COMPONENTS:
    fig, ax = plt.subplots(figsize=(8, 6))

    group_info = []
    for g in ["Low", "Intermediate", "High"]:
        gs = merged[merged["risk_group"] == g]
        ne = int(gs[evt_col].sum())
        rate = ne / len(gs) * 100
        label = f"{g} (n={len(gs)}, {ne} events, {rate:.1f}%)"
        group_info.append((g, len(gs), ne, rate))

        kmf = KaplanMeierFitter()
        kmf.fit(gs[TIME_COL], gs[evt_col], label=g)
        kmf.plot_survival_function(ax=ax, color=COLORS[g], ci_alpha=0.12,
                                    label=label, linewidth=1.5)

    # Log-rank
    lr = multivariate_logrank_test(merged[TIME_COL], merged["risk_group"], merged[evt_col])
    p_text = "p < 0.001" if lr.p_value < 0.001 else f"p = {lr.p_value:.3f}"
    ax.text(0.98, 0.98, f"Log-rank {p_text}\n(chi2={lr.test_statistic:.1f})",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="gray"))

    ax.set_xlim(0, 370)
    ax.set_xlabel("Time (days)", fontsize=12)
    ax.set_ylabel("Event-free probability", fontsize=12)
    ax.set_title(f"DL-ECG risk stratification — {comp_name}\n"
                 f"(n={n_evt} first events, CVD_Composite_4 model)",
                 fontsize=13)
    ax.legend(loc="lower left", fontsize=9)

    # Number at risk table as text annotation below the plot
    table_lines = ["Number at risk"]
    for g in ["High", "Intermediate", "Low"]:
        gs = merged[merged["risk_group"] == g]
        counts = [str((gs[TIME_COL] >= t).sum()) for t in RISK_TABLE_TIMES]
        table_lines.append(f"  {g:>12s}: " + "  ".join(f"{c:>5s}" for c in counts))
    table_text = "\n".join(table_lines)
    ax.text(0.02, -0.15, table_text, transform=ax.transAxes, fontsize=7,
            fontfamily="monospace", va="top")

    fname = OUT_DIR / f"km_{evt_col}_CVD4.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    fig.savefig(fname.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")

    # Print event rate summary
    print(f"  {comp_name}: ", end="")
    for g, n, ne, rate in group_info:
        print(f"{g}={rate:.1f}% ", end="")
    print(f"| Log-rank p={lr.p_value:.2e}")

print("\nDone.")

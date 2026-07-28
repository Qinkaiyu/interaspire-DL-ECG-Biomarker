"""
Component endpoint analysis for CVD_Composite_4.

Derives individual component events from `first_component_Composite_4`,
using the FIRST event logic from the composite endpoint construction.
All endpoints share `time_Composite_4` as the time variable.

Components:
  - CV death (SURVIVAL + cvd_death_flag=1): 24 events
  - MI (HOSPAMI): 67 events
  - HF (HOSPHF): 52 events
  - Stroke (HOSPSTROKE): 24 events

For each component, patients with a DIFFERENT first event are censored
at time_Composite_4 (competing risk censoring).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from lifelines.utils import concordance_index
from lifelines.statistics import multivariate_logrank_test
from lifelines import KaplanMeierFitter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH = Path("data/INTERASPIRE_analysis_dataset.xlsx")
M5_PATH = Path("model5_cvd_mace4/E2E_MultiTask_32_2_phase_cvd_mace4_1yr_oof.csv")
OOF_M04 = Path("results/oof_predictions/oof_CVD4.csv")
OUT_DIR = Path("results/component_endpoints")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Component definitions: (display_name, first_component_value)
# "SURVIVAL" means death was the first event; combined with cvd_death_flag=1
COMPONENTS = [
    ("CV death", "SURVIVAL"),
    ("MI", "HOSPAMI"),
    ("HF", "HOSPHF"),
    ("Stroke", "HOSPSTROKE"),
]

# ── Load data ───────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_excel(DATA_PATH, engine="openpyxl")

# Load Model 5 OOF + per-fold z-normalize
oof5 = pd.read_csv(M5_PATH)
risk_norm = oof5["oof_log_risk"].copy()
for fold in oof5["test_fold"].unique():
    m = oof5["test_fold"] == fold
    v = oof5.loc[m, "oof_log_risk"]
    risk_norm.loc[m] = (v - v.mean()) / v.std()
oof5["risk_norm"] = risk_norm

# Load Models 0-4 OOF
oof04 = pd.read_csv(OOF_M04)

# Merge
merged = df.merge(oof5[["STUDYID", "risk_norm", "test_fold"]], on="STUDYID", how="inner")
merged = merged.merge(
    oof04[["STUDYID", "fold", "Model_0_cox_risk", "Model_1_cox_risk",
            "Model_2_cox_risk", "Model_3_cox_risk", "Model_4_cox_risk"]],
    on="STUDYID", how="inner",
)
print(f"Merged N={len(merged)}")

# Per-fold z-score normalization for ALL model risks (pooled C-index requires this)
cox_risk_cols = ["Model_0_cox_risk", "Model_1_cox_risk", "Model_2_cox_risk",
                 "Model_3_cox_risk", "Model_4_cox_risk"]
for col in cox_risk_cols:
    norm_col = col.replace("_cox_risk", "_norm")
    merged[norm_col] = merged[col].copy()
    for fold_val in merged["fold"].unique():
        mask = merged["fold"] == fold_val
        vals = merged.loc[mask, col]
        std = vals.std()
        if std > 0:
            merged.loc[mask, norm_col] = (vals - vals.mean()) / std
        else:
            merged.loc[mask, norm_col] = 0.0
print("Per-fold z-score normalization applied to Models 0-4")

# Derive component event columns from first_component_Composite_4
# For each component: event=1 if that component was the FIRST event
# For all patients (event or not): time = time_Composite_4
first_comp = merged["first_component_Composite_4"].fillna("")

merged["comp_cv_death"] = ((first_comp == "SURVIVAL") & (merged["cvd_death_flag"] == 1)).astype(int)
merged["comp_mi"] = (first_comp == "HOSPAMI").astype(int)
merged["comp_hf"] = (first_comp == "HOSPHF").astype(int)
merged["comp_stroke"] = (first_comp == "HOSPSTROKE").astype(int)

# All use the same time variable
TIME_COL = "time_Composite_4"

COMPONENT_COLS = {
    "CV death": "comp_cv_death",
    "MI": "comp_mi",
    "HF": "comp_hf",
    "Stroke": "comp_stroke",
}

# Verify
print("\nComponent event counts (first-event basis):")
for name, col in COMPONENT_COLS.items():
    n = merged[col].sum()
    print(f"  {name}: {n} events ({n/len(merged)*100:.1f}%)")
total = sum(merged[col].sum() for col in COMPONENT_COLS.values())
print(f"  Total: {total} (CVD_Composite_4 events: {merged['event_CVD_Composite_4'].sum()})")


def compute_pooled_c_bootstrap(data, risk_col, evt_col, n_boot=1000, seed=42):
    """Pooled C-index with bootstrap 95% CI (all OOF predictions together)."""
    valid = data[[risk_col, evt_col, TIME_COL]].dropna()
    if valid[evt_col].sum() < 3:
        return np.nan, np.nan, np.nan
    try:
        c = concordance_index(valid[TIME_COL], -valid[risk_col], valid[evt_col])
    except Exception:
        return np.nan, np.nan, np.nan

    rng = np.random.RandomState(seed)
    boot_cs = []
    n = len(valid)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        b = valid.iloc[idx]
        if b[evt_col].sum() < 3:
            continue
        try:
            bc = concordance_index(b[TIME_COL], -b[risk_col], b[evt_col])
            boot_cs.append(bc)
        except Exception:
            continue
    if len(boot_cs) < 100:
        return c, np.nan, np.nan
    lo = np.percentile(boot_cs, 2.5)
    hi = np.percentile(boot_cs, 97.5)
    return c, lo, hi


# ── Analysis ────────────────────────────────────────────────────────────────
model_risks = {
    "Model 0": "Model_0_norm",
    "Model 1": "Model_1_norm",
    "Model 2": "Model_2_norm",
    "Model 3": "Model_3_norm",
    "Model 4": "Model_4_norm",
    "Model 5 (DL-ECG)": "risk_norm",
}

results = []

for comp_name, evt_col in COMPONENT_COLS.items():
    n_events = int(merged[evt_col].sum())
    print(f"\n{'='*60}")
    print(f"Endpoint: {comp_name} ({n_events} events, {n_events/len(merged)*100:.2f}%)")
    print(f"{'='*60}")

    row = {"Endpoint": comp_name, "N": len(merged), "Events": n_events,
           "Event rate": f"{n_events/len(merged)*100:.1f}%"}

    for model_name, risk_col in model_risks.items():
        c, lo, hi = compute_pooled_c_bootstrap(merged, risk_col, evt_col)
        ci_str = f"({lo:.3f}-{hi:.3f})" if not np.isnan(lo) else ""
        print(f"  {model_name}: C={c:.3f} {ci_str}")
        row[f"{model_name} C"] = round(c, 3) if not np.isnan(c) else np.nan
        row[f"{model_name} 95CI"] = ci_str

    results.append(row)

    # ── KM by DL-ECG risk tertile ──────────────────────────────────
    rmin, rmax = merged["risk_norm"].min(), merged["risk_norm"].max()
    t1 = rmin + (rmax - rmin) / 3
    t2 = rmin + 2 * (rmax - rmin) / 3
    merged["_grp"] = pd.cut(
        merged["risk_norm"],
        bins=[rmin - 0.001, t1, t2, rmax + 0.001],
        labels=["Low", "Intermediate", "High"],
    )

    print(f"\n  Risk stratification ({comp_name}):")
    for g in ["Low", "Intermediate", "High"]:
        gs = merged[merged["_grp"] == g]
        ne = int(gs[evt_col].sum())
        rate = ne / len(gs) * 100 if len(gs) > 0 else 0
        print(f"    {g}: N={len(gs)}, events={ne}, rate={rate:.2f}%")

    # Log-rank
    if merged[evt_col].sum() >= 10:
        lr = multivariate_logrank_test(merged[TIME_COL], merged["_grp"], merged[evt_col])
        print(f"  Log-rank: chi2={lr.test_statistic:.1f}, p={lr.p_value:.2e}")

# ── Also add CVD_Composite_4 as reference ──────────────────────────────────
print(f"\n{'='*60}")
print(f"Reference: CVD_Composite_4 (167 events)")
print(f"{'='*60}")
ref_row = {"Endpoint": "CVD_Composite_4", "N": len(merged),
           "Events": int(merged["event_CVD_Composite_4"].sum()),
           "Event rate": f"{merged['event_CVD_Composite_4'].mean()*100:.1f}%"}
for model_name, risk_col in model_risks.items():
    c, lo, hi = compute_pooled_c_bootstrap(merged, risk_col, "event_CVD_Composite_4")
    ci_str = f"({lo:.3f}-{hi:.3f})" if not np.isnan(lo) else ""
    print(f"  {model_name}: C={c:.3f} {ci_str}")
    ref_row[f"{model_name} C"] = round(c, 3)
    ref_row[f"{model_name} 95CI"] = ci_str
results.append(ref_row)

# ── Save summary ──────────────────────────────────────────────────────────
res_df = pd.DataFrame(results)
out_csv = OUT_DIR / "component_endpoint_cindex.csv"
res_df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")

# ── Combined KM figure (2x2) ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
colors = {"Low": "#2ca02c", "Intermediate": "#ff7f0e", "High": "#d62728"}
panel_labels = ["A", "B", "C", "D"]

for idx, (comp_name, evt_col) in enumerate(COMPONENT_COLS.items()):
    ax = axes[idx]
    n_events = int(merged[evt_col].sum())

    rmin, rmax = merged["risk_norm"].min(), merged["risk_norm"].max()
    t1 = rmin + (rmax - rmin) / 3
    t2 = rmin + 2 * (rmax - rmin) / 3
    merged["_grp"] = pd.cut(
        merged["risk_norm"],
        bins=[rmin - 0.001, t1, t2, rmax + 0.001],
        labels=["Low", "Intermediate", "High"],
    )

    for g in ["Low", "Intermediate", "High"]:
        gs = merged[merged["_grp"] == g]
        if len(gs) < 5:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(gs[TIME_COL], gs[evt_col], label=g)
        ne = int(gs[evt_col].sum())
        rate = ne / len(gs) * 100
        label = f"{g} (n={len(gs)}, {ne} events, {rate:.1f}%)"
        kmf.plot_survival_function(ax=ax, color=colors[g], ci_alpha=0.12, label=label)

    ax.set_xlim(0, 370)
    ax.set_xlabel("Time (days)", fontsize=10)
    ax.set_ylabel("Event-free probability", fontsize=10)
    ax.set_title(f"{panel_labels[idx]}  {comp_name} (n={n_events} events)",
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(loc="lower left", fontsize=8)

    if n_events >= 10:
        lr = multivariate_logrank_test(merged[TIME_COL], merged["_grp"], merged[evt_col])
        p_text = "p < 0.001" if lr.p_value < 0.001 else f"p = {lr.p_value:.3f}"
        ax.text(0.98, 0.98, f"Log-rank {p_text}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.suptitle("DL-ECG risk stratification by individual endpoint component\n"
             "(derived from CVD_Composite_4 first-event logic)",
             fontsize=14, y=1.01)
plt.tight_layout()
combined = OUT_DIR / "km_combined_components_CVD4.png"
fig.savefig(combined, dpi=200, bbox_inches="tight")
fig.savefig(combined.with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig)
print(f"\nCombined KM figure: {combined}")

"""
Cumulative Incidence Function (CIF) analysis for CVD_Composite_4 components.

Uses Aalen-Johansen estimator to properly account for competing risks:
  - When analyzing HF, patients who first had MI/CV death/Stroke are treated
    as competing events (not censored), giving the TRUE cumulative probability.

Outputs:
  1. CIF curves by DL-ECG risk tertile for each component (4 individual figures)
  2. Combined 2x2 CIF figure
  3. Comparison figure: KM (cause-specific) vs CIF (competing risks) side by side
"""

import pandas as pd
import numpy as np
from pathlib import Path
from lifelines import AalenJohansenFitter
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
print("Loading data...")
df = pd.read_excel(DATA_PATH, engine="openpyxl")

oof5 = pd.read_csv(M5_PATH)
risk_norm = oof5["oof_log_risk"].copy()
for fold in oof5["test_fold"].unique():
    m = oof5["test_fold"] == fold
    v = oof5.loc[m, "oof_log_risk"]
    risk_norm.loc[m] = (v - v.mean()) / v.std()
oof5["risk_norm"] = risk_norm

merged = df.merge(oof5[["STUDYID", "risk_norm"]], on="STUDYID", how="inner")
print(f"Merged N={len(merged)}")

TIME_COL = "time_Composite_4"

# ── Encode event type as integer for competing risks ────────────────────────
# 0 = censored (no event)
# 1 = CV death
# 2 = MI
# 3 = HF
# 4 = Stroke
first_comp = merged["first_component_Composite_4"].fillna("")
merged["event_type"] = 0  # default: censored
merged.loc[(first_comp == "SURVIVAL") & (merged["cvd_death_flag"] == 1), "event_type"] = 1
merged.loc[first_comp == "HOSPAMI", "event_type"] = 2
merged.loc[first_comp == "HOSPHF", "event_type"] = 3
merged.loc[first_comp == "HOSPSTROKE", "event_type"] = 4
# Non-CVD deaths (SURVIVAL but cvd_death_flag != 1) remain as censored (0)

EVENT_LABELS = {1: "CV death", 2: "MI", 3: "HF", 4: "Stroke"}

print("\nEvent type distribution:")
for code, label in EVENT_LABELS.items():
    n = (merged["event_type"] == code).sum()
    print(f"  {code} ({label}): {n}")
print(f"  0 (censored): {(merged['event_type'] == 0).sum()}")

# ── Equal-width tertile ─────────────────────────────────────────────────────
rmin, rmax = merged["risk_norm"].min(), merged["risk_norm"].max()
t1 = rmin + (rmax - rmin) / 3
t2 = rmin + 2 * (rmax - rmin) / 3
merged["risk_group"] = pd.cut(
    merged["risk_norm"],
    bins=[rmin - 0.001, t1, t2, rmax + 0.001],
    labels=["Low", "Intermediate", "High"],
)

COLORS = {"Low": "#2ca02c", "Intermediate": "#ff7f0e", "High": "#d62728"}

# ── CIF for each component, by risk group ───────────────────────────────────
print("\n=== Cumulative Incidence Function (Aalen-Johansen) ===")

fig_combined, axes_combined = plt.subplots(2, 2, figsize=(14, 10))
axes_combined = axes_combined.flatten()
panel_labels = ["A", "B", "C", "D"]

for idx, (event_code, comp_name) in enumerate(EVENT_LABELS.items()):
    print(f"\n--- {comp_name} (event_type={event_code}) ---")

    # Individual figure
    fig_ind, ax_ind = plt.subplots(figsize=(8, 6))
    ax_comb = axes_combined[idx]

    for g in ["Low", "Intermediate", "High"]:
        gs = merged[merged["risk_group"] == g].copy()
        n_total = len(gs)
        n_events = (gs["event_type"] == event_code).sum()
        rate = n_events / n_total * 100

        # AalenJohansenFitter for the event of interest
        ajf = AalenJohansenFitter(calculate_variance=True)
        ajf.fit(gs[TIME_COL], gs["event_type"], event_of_interest=event_code)

        # CIF values and 95% CI
        ci = ajf.cumulative_density_
        times = ci.index.values
        cif_vals = ci.values.flatten()

        # Confidence intervals from AJ variance
        ci_lo = ajf.confidence_interval_cumulative_density_.iloc[:, 0].values
        ci_hi = ajf.confidence_interval_cumulative_density_.iloc[:, 1].values

        label = f"{g} (n={n_total}, {n_events} events, {rate:.1f}%)"

        # Plot on individual figure with 95% CI band
        ax_ind.plot(times, cif_vals, color=COLORS[g], linewidth=1.5, label=label)
        ax_ind.fill_between(times, ci_lo, ci_hi, color=COLORS[g], alpha=0.12)

        # Plot on combined figure with 95% CI band
        ax_comb.plot(times, cif_vals, color=COLORS[g], linewidth=1.5, label=label)
        ax_comb.fill_between(times, ci_lo, ci_hi, color=COLORS[g], alpha=0.12)

        # Print 1-year CIF
        cif_1yr = cif_vals[-1] if len(cif_vals) > 0 else 0
        print(f"  {g}: N={n_total}, events={n_events}, 1yr CIF={cif_1yr*100:.2f}%")

    # Individual figure formatting
    ax_ind.set_xlim(0, 370)
    ax_ind.set_ylim(0, None)
    ax_ind.set_xlabel("Time (days)", fontsize=12)
    ax_ind.set_ylabel("Cumulative incidence", fontsize=12)
    ax_ind.set_title(f"Cumulative Incidence — {comp_name}\n"
                     f"(Aalen-Johansen, competing risks accounted)",
                     fontsize=13)
    ax_ind.legend(loc="upper left", fontsize=9)

    fname = OUT_DIR / f"cif_{comp_name.lower().replace(' ', '_')}_CVD4.png"
    fig_ind.savefig(fname, dpi=200, bbox_inches="tight")
    fig_ind.savefig(fname.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig_ind)
    print(f"  Saved: {fname}")

    # Combined figure formatting
    n_events_total = (merged["event_type"] == event_code).sum()
    ax_comb.set_xlim(0, 370)
    ax_comb.set_ylim(0, None)
    ax_comb.set_xlabel("Time (days)", fontsize=10)
    ax_comb.set_ylabel("Cumulative incidence", fontsize=10)
    ax_comb.set_title(f"{panel_labels[idx]}  {comp_name} (n={n_events_total} events)",
                      fontsize=12, fontweight="bold", loc="left")
    ax_comb.legend(loc="upper left", fontsize=8)

# Combined figure
plt.suptitle("Cumulative Incidence Functions by DL-ECG risk tertile\n"
             "(Aalen-Johansen estimator, competing risks accounted)",
             fontsize=14, y=1.01)
plt.tight_layout()
combined_path = OUT_DIR / "cif_combined_components_CVD4.png"
fig_combined.savefig(combined_path, dpi=200, bbox_inches="tight")
fig_combined.savefig(combined_path.with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig_combined)
print(f"\nCombined CIF figure: {combined_path}")

# ── Comparison: KM vs CIF side by side ──────────────────────────────────────
print("\n=== KM vs CIF Comparison ===")
from lifelines import KaplanMeierFitter

fig_comp, axes_comp = plt.subplots(2, 4, figsize=(20, 10))
# Row 0: KM (cause-specific), Row 1: CIF (competing risks)

for idx, (event_code, comp_name) in enumerate(EVENT_LABELS.items()):
    ax_km = axes_comp[0, idx]
    ax_cif = axes_comp[1, idx]

    # Derive cause-specific event column
    cs_event = (merged["event_type"] == event_code).astype(int)

    for g in ["Low", "Intermediate", "High"]:
        gs_mask = merged["risk_group"] == g
        gs = merged[gs_mask]
        n_events = (gs["event_type"] == event_code).sum()
        rate = n_events / len(gs) * 100
        label = f"{g} ({n_events} evt, {rate:.1f}%)"

        # KM (cause-specific: other events = censored)
        kmf = KaplanMeierFitter()
        kmf.fit(gs[TIME_COL], cs_event[gs_mask])
        # Plot as 1 - survival = cumulative incidence (cause-specific)
        times_km = kmf.survival_function_.index.values
        ci_km = 1 - kmf.survival_function_.values.flatten()
        ax_km.plot(times_km, ci_km, color=COLORS[g], linewidth=1.3, label=label)

        # CIF (Aalen-Johansen)
        ajf = AalenJohansenFitter(calculate_variance=True)
        ajf.fit(gs[TIME_COL], gs["event_type"], event_of_interest=event_code)
        ci_aj = ajf.cumulative_density_
        ax_cif.plot(ci_aj.index, ci_aj.values.flatten(), color=COLORS[g],
                    linewidth=1.3, label=label)

    for ax, method in [(ax_km, "KM (cause-specific)"), (ax_cif, "CIF (Aalen-Johansen)")]:
        ax.set_xlim(0, 370)
        ax.set_ylim(0, None)
        ax.set_xlabel("Time (days)", fontsize=9)
        ax.set_ylabel("Cumulative incidence", fontsize=9)
        ax.legend(loc="upper left", fontsize=7)

    ax_km.set_title(f"{comp_name}\nKM (cause-specific)", fontsize=11, fontweight="bold")
    ax_cif.set_title(f"{comp_name}\nCIF (competing risks)", fontsize=11, fontweight="bold")

plt.suptitle("Comparison: Kaplan-Meier (cause-specific) vs Cumulative Incidence Function (competing risks)\n"
             "Top row ignores competing events; Bottom row properly accounts for them",
             fontsize=13, y=1.02)
plt.tight_layout()
comp_path = OUT_DIR / "km_vs_cif_comparison_CVD4.png"
fig_comp.savefig(comp_path, dpi=200, bbox_inches="tight")
fig_comp.savefig(comp_path.with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig_comp)
print(f"Comparison figure: {comp_path}")

print("\nDone.")

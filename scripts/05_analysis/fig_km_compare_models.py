"""
Compare KM curves for top Model 5 variants.
Both tertile (3-split) and quartile (4-split), equal-width on risk range.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

BASE = Path(".")
df04 = pd.read_csv(BASE / "results" / "oof_predictions" / "oof_CVD4.csv")
evt_col = "event_CVD_Composite_4"
time_col = "time_Composite_4"
OUT_DIR = BASE / "results" / "figures" / "km_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Top candidates
CANDIDATES = [
    ("E2E_MultiTask_32_2_phase", "2-phase (C=0.728)"),
    ("E2E_MultiTask_32_aux_3bio_w=0.7", "aux-3bio w=0.7 (C=0.728)"),
    ("E2E_MultiTask_32_aux_5bio_w=0.5", "aux-5bio w=0.5 (C=0.723)"),
    ("E2E_MultiTask_32_aux_3bio_w=1.0", "aux-3bio w=1.0 (C=0.713)"),
    ("E2E_MultiTask_32_aux_5bio_w=1.0", "aux-5bio w=1.0 (C=0.709)"),
]

COLORS_3 = {"Low": "#2ecc71", "Intermediate": "#3498db", "High": "#e74c3c"}
COLORS_4 = {"Low": "#2ecc71", "Intermediate-low": "#3498db",
            "Intermediate-high": "#f39c12", "High": "#e74c3c"}


def equal_width_split(risk, n_groups):
    r_min, r_max = risk.min(), risk.max()
    width = (r_max - r_min) / n_groups
    cuts = [r_min + width * i for i in range(n_groups + 1)]
    cuts[0] = r_min - 0.001  # include_lowest
    if n_groups == 3:
        labels = ["Low", "Intermediate", "High"]
    else:
        labels = ["Low", "Intermediate-low", "Intermediate-high", "High"]
    return pd.cut(risk, bins=cuts, labels=labels, include_lowest=True)


def plot_km(merged, group_col, groups, colors, title, ax):
    for grp in groups:
        sub = merged[merged[group_col] == grp]
        kmf = KaplanMeierFitter()
        kmf.fit(sub[time_col], event_observed=sub[evt_col], label=grp)
        n = len(sub)
        ev = int(sub[evt_col].sum())
        rate = ev / n * 100
        kmf.plot_survival_function(
            ax=ax, color=colors[grp], linewidth=1.8,
            ci_alpha=0.12, ci_show=True,
            label=f"{grp} (n={n}, {ev}ev, {rate:.1f}%)"
        )

    lr = multivariate_logrank_test(merged[time_col], merged[group_col], merged[evt_col])
    p_text = "p < 0.001" if lr.p_value < 0.001 else f"p = {lr.p_value:.3f}"
    ax.text(0.02, 0.03, f"Log-rank {p_text}\n\u03c7\u00b2={lr.test_statistic:.1f}",
            transform=ax.transAxes, fontsize=8, fontstyle="italic")

    ax.set_xlabel("Time (days)", fontsize=9)
    ax.set_ylabel("Survival probability", fontsize=9)
    ax.set_xlim(0, 370)
    ax.set_ylim(0.78, 1.01)
    ax.grid(True, alpha=0.15)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="lower left", frameon=True, edgecolor="#ccc", framealpha=0.95)
    ax.set_title(title, fontsize=9, fontweight="bold")


# ── Main ─────────────────────────────────────────────────────────────────────
print("=" * 120)
print(f"{'Model':<40s} | {'--- Tertile (3-split) ---':^50s} | {'--- Quartile (4-split) ---':^55s}")
print(f"{'':40s} | {'N_L':>6s} {'%L':>5s} {'N_M':>6s} {'%M':>5s} {'N_H':>6s} {'%H':>5s} {'H/L':>5s} {'chi2':>6s}"
      f" | {'N_L':>6s} {'%L':>5s} {'N_ML':>6s} {'%ML':>5s} {'N_MH':>6s} {'%MH':>5s} {'N_H':>6s} {'%H':>5s} {'H/L':>5s} {'chi2':>6s}")
print("-" * 120)

fig, axes = plt.subplots(len(CANDIDATES), 2, figsize=(14, 4 * len(CANDIDATES)))

for row_i, (model_key, model_label) in enumerate(CANDIDATES):
    oof_path = BASE / "model5_cvd_mace4" / f"{model_key}_cvd_mace4_1yr_oof.csv"
    df5 = pd.read_csv(oof_path)
    merged = df04.merge(
        df5[["STUDYID", "oof_log_risk"]].rename(columns={"oof_log_risk": "risk"}),
        on="STUDYID", how="inner"
    )
    risk = merged["risk"].values

    # ── Tertile ──────────────────────────────────────────────────────────
    merged["rg3"] = equal_width_split(risk, 3)
    groups3 = ["Low", "Intermediate", "High"]

    t3_stats = {}
    for g in groups3:
        sub = merged[merged["rg3"] == g]
        t3_stats[g] = {"n": len(sub), "ev": int(sub[evt_col].sum()),
                        "rate": sub[evt_col].sum() / len(sub) * 100}

    lr3 = multivariate_logrank_test(merged[time_col], merged["rg3"], merged[evt_col])
    ratio3 = t3_stats["High"]["rate"] / t3_stats["Low"]["rate"] if t3_stats["Low"]["rate"] > 0 else 0

    # ── Quartile ─────────────────────────────────────────────────────────
    merged["rg4"] = equal_width_split(risk, 4)
    groups4 = ["Low", "Intermediate-low", "Intermediate-high", "High"]

    t4_stats = {}
    for g in groups4:
        sub = merged[merged["rg4"] == g]
        t4_stats[g] = {"n": len(sub), "ev": int(sub[evt_col].sum()),
                        "rate": sub[evt_col].sum() / len(sub) * 100}

    lr4 = multivariate_logrank_test(merged[time_col], merged["rg4"], merged[evt_col])
    ratio4 = t4_stats["High"]["rate"] / t4_stats["Low"]["rate"] if t4_stats["Low"]["rate"] > 0 else 0

    # Print summary
    s3 = t3_stats
    s4 = t4_stats
    print(f"{model_label:<40s} "
          f"| {s3['Low']['n']:6d} {s3['Low']['rate']:4.1f}% {s3['Intermediate']['n']:6d} {s3['Intermediate']['rate']:4.1f}% "
          f"{s3['High']['n']:6d} {s3['High']['rate']:4.1f}% {ratio3:4.1f}x {lr3.test_statistic:6.1f}"
          f" | {s4['Low']['n']:6d} {s4['Low']['rate']:4.1f}% {s4['Intermediate-low']['n']:6d} {s4['Intermediate-low']['rate']:4.1f}% "
          f"{s4['Intermediate-high']['n']:6d} {s4['Intermediate-high']['rate']:4.1f}% "
          f"{s4['High']['n']:6d} {s4['High']['rate']:4.1f}% {ratio4:4.1f}x {lr4.test_statistic:6.1f}")

    # Plot
    plot_km(merged, "rg3", groups3, COLORS_3,
            f"{model_label} — Tertile", axes[row_i, 0])
    plot_km(merged, "rg4", groups4, COLORS_4,
            f"{model_label} — Quartile", axes[row_i, 1])

plt.tight_layout()
fig.savefig(OUT_DIR / "km_all_models_comparison.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT_DIR / "km_all_models_comparison.pdf", dpi=200, bbox_inches="tight")
plt.close()
print(f"\nSaved -> {OUT_DIR / 'km_all_models_comparison.png'}")

# ── Detailed event rate tables ───────────────────────────────────────────────
print("\n\n" + "=" * 80)
print("DETAILED EVENT RATE TABLES")
print("=" * 80)

for model_key, model_label in CANDIDATES:
    oof_path = BASE / "model5_cvd_mace4" / f"{model_key}_cvd_mace4_1yr_oof.csv"
    df5 = pd.read_csv(oof_path)
    merged = df04.merge(
        df5[["STUDYID", "oof_log_risk"]].rename(columns={"oof_log_risk": "risk"}),
        on="STUDYID", how="inner"
    )
    risk = merged["risk"].values

    print(f"\n--- {model_label} ---")
    print(f"  Risk range: [{risk.min():.3f}, {risk.max():.3f}]")

    for n_groups, label in [(3, "Tertile"), (4, "Quartile")]:
        merged["rg"] = equal_width_split(risk, n_groups)
        if n_groups == 3:
            groups = ["Low", "Intermediate", "High"]
        else:
            groups = ["Low", "Intermediate-low", "Intermediate-high", "High"]

        print(f"\n  {label} (equal-width):")
        print(f"  {'Group':<22s} {'N':>6s} {'Events':>7s} {'Rate':>7s} {'% of all events':>16s}")
        total_ev = int(merged[evt_col].sum())
        for g in groups:
            sub = merged[merged["rg"] == g]
            n = len(sub)
            ev = int(sub[evt_col].sum())
            rate = ev / n * 100 if n > 0 else 0
            ev_pct = ev / total_ev * 100 if total_ev > 0 else 0
            print(f"  {g:<22s} {n:6d} {ev:7d} {rate:6.1f}% {ev_pct:15.1f}%")

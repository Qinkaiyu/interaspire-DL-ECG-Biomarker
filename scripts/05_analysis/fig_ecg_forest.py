"""
Fig. 7: ECG biomarker forest plot
All 18 ECG biomarkers — univariate HR + 95% CI for CVD_Composite_4
Horizontal forest plot, sorted by HR magnitude.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from lifelines import CoxPHFitter

BASE = Path(".")
DATA_PATH = BASE / "data" / "INTERASPIRE_analysis_dataset.xlsx"
OOF = BASE / "results" / "oof_predictions" / "oof_CVD4.csv"
OUT_DIR = BASE / "results" / "figures"

evt_col = "event_CVD_Composite_4"
time_col = "time_Composite_4"

# ── Load ─────────────────────────────────────────────────────────────────────
raw = pd.read_excel(DATA_PATH, engine="openpyxl")
df04 = pd.read_csv(OOF)
merged = df04[["STUDYID", evt_col, time_col]].merge(raw[["STUDYID"]], on="STUDYID", how="inner")

# Merge ECG columns
ECG_BINARY = [
    ("Atrial Fibrillation", "Atrial fibrillation"),
    ("LBBB", "Left bundle branch block"),
    ("RBBB", "Right bundle branch block"),
    ("Q Wave", "Q wave"),
    ("ST Elevation", "ST elevation"),
    ("ST Depression", "ST depression"),
    ("T Wave Inversion", "T wave inversion"),
    ("Ischaemic", "Ischaemic changes"),
    ("QT Prolongation", "QT prolongation"),
    ("LVH", "Left ventricular hypertrophy"),
    ("1 AV Block", "First-degree AV block"),
    ("Left Axis Deviation", "Left axis deviation"),
    ("Right Axis Deviation", "Right axis deviation"),
    ("MI (Old)", "Old myocardial infarction"),
    ("MI(Acute)", "Acute myocardial infarction"),
]

ECG_CONT = [
    ("PR Interval (ms)", "PR interval (ms)"),
    ("QRS Duration (ms)", "QRS duration (ms)"),
    ("QT Interval (ms)", "QT interval (ms)"),
]

all_ecg_cols = [c for c, _ in ECG_BINARY] + [c for c, _ in ECG_CONT]
raw_ecg = raw[["STUDYID"] + [c for c in all_ecg_cols if c in raw.columns]].copy()

# Encode binary
for col, _ in ECG_BINARY:
    if col in raw_ecg.columns:
        raw_ecg[col] = raw_ecg[col].map({"Yes": 1, "No": 0})

# Convert continuous to numeric
for col, _ in ECG_CONT:
    if col in raw_ecg.columns:
        raw_ecg[col] = pd.to_numeric(raw_ecg[col], errors="coerce")

merged = merged.merge(raw_ecg, on="STUDYID", how="left")

# ── Univariate Cox PH for each biomarker ─────────────────────────────────────
results = []

for col, display in ECG_BINARY + ECG_CONT:
    if col not in merged.columns:
        continue

    df_cox = merged[["STUDYID", evt_col, time_col, col]].dropna().copy()
    df_cox["_t"] = np.clip(df_cox[time_col], 0.5, None)
    df_cox["_e"] = df_cox[evt_col]

    n = len(df_cox)
    n_pos = int((df_cox[col] > 0).sum()) if col in [c for c, _ in ECG_BINARY] else n
    n_events = int(df_cox["_e"].sum())

    if n < 50 or n_events < 5:
        continue

    # Standardize continuous variables for comparable HR
    is_cont = col in [c for c, _ in ECG_CONT]
    if is_cont:
        mean_val = df_cox[col].mean()
        std_val = df_cox[col].std()
        df_cox[col] = (df_cox[col] - mean_val) / std_val

    try:
        cph = CoxPHFitter()
        cph.fit(df_cox[[col, "_t", "_e"]], duration_col="_t", event_col="_e")
        hr = cph.hazard_ratios_.values[0]
        ci = cph.confidence_intervals_.values[0]
        p = cph.summary["p"].values[0]

        results.append({
            "col": col, "display": display,
            "hr": hr, "hr_lo": np.exp(ci[0]), "hr_hi": np.exp(ci[1]),
            "p": p, "n": n, "n_pos": n_pos, "n_events": n_events,
            "is_cont": is_cont,
        })
    except Exception as e:
        print(f"  {display}: FAILED — {e}")

res = pd.DataFrame(results)

# Print table
print(f"\n{'Biomarker':<32s} {'HR':>6s} {'95% CI':>16s} {'P':>8s} {'N':>6s} {'Prev/SD':>8s}")
print("-" * 80)
for _, r in res.iterrows():
    ci_str = f"({r['hr_lo']:.2f}\u2013{r['hr_hi']:.2f})"
    p_str = "<0.001" if r["p"] < 0.001 else f"{r['p']:.3f}"
    if r["is_cont"]:
        info = "per SD"
    else:
        info = f"{r['n_pos']/r['n']*100:.1f}%"
    print(f"  {r['display']:<30s} {r['hr']:6.2f} {ci_str:>16s} {p_str:>8s} {r['n']:6d} {info:>8s}")

# ── Sort by HR ───────────────────────────────────────────────────────────────
res = res.sort_values("hr", ascending=True).reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE: Horizontal forest plot
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 7))

y_pos = np.arange(len(res))

for i, (_, row) in enumerate(res.iterrows()):
    color = "#e74c3c" if row["hr"] > 1 and row["p"] < 0.05 else \
            "#3498db" if row["hr"] < 1 and row["p"] < 0.05 else "#999"

    ax.errorbar(
        row["hr"], y_pos[i],
        xerr=[[row["hr"] - row["hr_lo"]], [row["hr_hi"] - row["hr"]]],
        fmt="o", color=color, markersize=7, capsize=4, capthick=1.3,
        elinewidth=1.5, markeredgewidth=0, zorder=5,
    )

    # P-value annotation
    p_str = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else ""
    if p_str:
        ax.text(row["hr_hi"] + 0.05, y_pos[i], p_str,
                fontsize=9, va="center", color=color, fontweight="bold")

# Reference line at HR=1
ax.axvline(1.0, color="grey", linestyle="--", linewidth=1, alpha=0.5)

# Labels
labels = []
for _, row in res.iterrows():
    suffix = " (per SD)" if row["is_cont"] else ""
    labels.append(f"{row['display']}{suffix}")

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Hazard Ratio (95% CI)", fontsize=11)
ax.set_title("Univariate association of ECG biomarkers\nwith 1-year CVD composite endpoint",
             fontsize=12, fontweight="bold")
ax.set_xlim(0.3, max(res["hr_hi"].max() + 0.3, 3.5))
ax.grid(True, axis="x", alpha=0.15)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", labelsize=10)

# Significance legend
ax.text(0.98, 0.02, "* p<0.05  ** p<0.01  *** p<0.001\nRed: significant risk factor\nBlue: significant protective",
        transform=ax.transAxes, fontsize=7.5, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", alpha=0.9))

plt.tight_layout()
fig.savefig(OUT_DIR / "fig7_ecg_forest_CVD4.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "fig7_ecg_forest_CVD4.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSaved -> {OUT_DIR / 'fig7_ecg_forest_CVD4.png'}")

# Save data
res.to_csv(OUT_DIR / "fig7_ecg_forest_CVD4_data.csv", index=False)

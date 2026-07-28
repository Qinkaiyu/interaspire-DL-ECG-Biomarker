"""
Fig. 6 v4: Subgroup analysis
- Panels A-D (Sex, Age, ECG status, AF status):
    Model 3/4/5 with 95% CI error bars
- Panel E (Country):
    Model 5 only, each country a different color
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from lifelines.utils import concordance_index

BASE = Path(".")
DATA_PATH = BASE / "data" / "INTERASPIRE_analysis_dataset.xlsx"
OOF_M04 = BASE / "results" / "oof_predictions" / "oof_CVD4.csv"
OOF_M5 = BASE / "model5_cvd_mace4" / "E2E_MultiTask_32_2_phase_cvd_mace4_1yr_oof.csv"
OUT_DIR = BASE / "results" / "figures"

evt_col = "event_CVD_Composite_4"
time_col = "time_Composite_4"

# ── Load & merge ─────────────────────────────────────────────────────────────
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


raw_cols = ["STUDYID", "RQSEX", "AGE", "COUNTRY", "RQINDEX",
            "Sinus Rhythm", "Atrial Fibrillation", "Atrial Flutter",
            "Supraventricular Tachycardia", "RBBB", "LBBB",
            "1 AV Block", "2 AV Block", "3 AV Block",
            "Left Axis Deviation", "Right Axis Deviation", "Q Wave",
            "ST Elevation", "ST Depression", "T Wave Inversion",
            "Ischaemic", "LVH", "RVH", "QT Prolongation", "MI (Old)", "MI(Acute)"]
raw_cols = [c for c in raw_cols if c in raw.columns]
merged = merged.merge(raw[raw_cols], on="STUDYID", how="left")

# ── Derived columns ──────────────────────────────────────────────────────────
merged["age_group"] = np.where(merged["AGE"] < 65, "<65", "\u226565")

merged["idx_type"] = merged["RQINDEX"].map({
    "Acute myocardial infarction STEMI": "STEMI",
    "Acute myocardial infarction Non-STEMI": "NSTEMI",
    "Unstable angina / Acute myocardial ischaemia": "UA",
    "Elective percutaneous transluminal coronary angioplasty": "ePCI",
    "Elective coronary artery by-pass surgery": "eCABG",
})

abnormal_cols = ["Atrial Fibrillation", "Atrial Flutter", "Supraventricular Tachycardia",
                 "RBBB", "LBBB", "1 AV Block", "2 AV Block", "3 AV Block",
                 "Left Axis Deviation", "Right Axis Deviation", "Q Wave",
                 "ST Elevation", "ST Depression", "T Wave Inversion",
                 "Ischaemic", "LVH", "RVH", "QT Prolongation", "MI (Old)", "MI(Acute)"]
has_ecg = merged["Sinus Rhythm"].notna()
is_normal = (merged["Sinus Rhythm"] == "Yes")
for col in abnormal_cols:
    if col in merged.columns:
        is_normal = is_normal & ((merged[col] == "No") | merged[col].isna())
merged["ecg_type"] = "Abnormal"
merged.loc[has_ecg & is_normal, "ecg_type"] = "Normal"

# ── Models ───────────────────────────────────────────────────────────────────
MODEL_DEFS = [
    ("Model 3", "Model_3_cox_risk", "fold"),
    ("Model 4", "Model_4_cox_risk", "fold"),
    ("Model 5", "Model_5_risk", "fold5"),
]

MODEL_COLORS = {"Model 3": "#FF9500", "Model 4": "#AF52DE", "Model 5": "#FF3B30"}
MODEL_MARKERS = {"Model 3": "s", "Model 4": "D", "Model 5": "o"}

# Country colors (14 countries) — high-saturation, distinct hues
COUNTRY_COLORS = {
    "Colombia": "#7B2D8E", "Nigeria": "#1B9E77", "Portugal": "#D95F02",
    "Malaysia": "#2166AC", "UAE": "#E7298A", "Kenia": "#A6761D",
    "Philippines": "#E6AB02", "Tanzania": "#666666", "Poland": "#66A61E",
    "Singapore": "#E41A1C", "Argentina": "#377EB8", "Egypt": "#FF7F00",
    "Indonesia": "#4DAF4A", "China": "#984EA3",
}

# ── C-index helper (mean-fold, consistent with fig2) ────────────────────────
def mean_fold_c(data, risk_col, fold_col):
    """Mean-fold C-index with 95% CI from fold-level SD.
    Matches the approach in fig_cindex_and_roc.py (fig2)."""
    fold_cs = []
    for fid in sorted(data[fold_col].unique()):
        sub = data[data[fold_col] == fid]
        tv = sub[time_col].values.astype(float)
        yv = sub[evt_col].values.astype(float)
        rv = sub[risk_col].values
        v = ~np.isnan(rv)
        if v.sum() == 0 or len(np.unique(yv[v])) < 2:
            continue
        try:
            fold_cs.append(concordance_index(
                np.clip(tv[v], 0.5, None), -rv[v], yv[v]))
        except Exception:
            continue
    if len(fold_cs) < 2:
        return np.nan, np.nan, np.nan
    mean_c = np.mean(fold_cs)
    sd_c = np.std(fold_cs, ddof=1)
    se_c = sd_c / np.sqrt(len(fold_cs))
    return mean_c, mean_c - 1.96 * se_c, mean_c + 1.96 * se_c


# ── Helper: multi-model panel ────────────────────────────────────────────────
def plot_multi_model_panel(ax, col, values, title, show_legend=False):
    """Plot Model 3/4/5 with CI for each subgroup level, with value annotations."""
    # Within each category, keep the three models a bit closer (narrow dodge).
    model_offsets = {"Model 3": -0.14, "Model 4": 0.0, "Model 5": 0.14}
    # Value labels: small per-model vertical stagger above the group's tallest CI cap.
    label_dy = 0.009
    label_stagger = {"Model 3": 0.0, "Model 4": 0.03, "Model 5": 0.06}

    # Pull category centers closer together so panels don't show a wide empty mid-gap
    # (tick positions no longer 0 .. n-1 spanning the full axis width).
    n_cat = len(values)
    x_centers = np.linspace(-0.30, 0.30, n_cat)
    x_margin = 0.26

    for val_i, val in enumerate(values):
        sub = merged[merged[col] == val]
        n = len(sub)
        ev = int(sub[evt_col].sum())
        if ev < 5:
            continue

        x_base = x_centers[val_i]
        series = []
        for mname, cox_col_m, fold_col in MODEL_DEFS:
            c, lo, hi = mean_fold_c(sub, cox_col_m, fold_col)
            if np.isnan(c):
                continue
            x = x_base + model_offsets[mname]
            yerr_lo = max(c - lo, 0)
            yerr_hi = max(hi - c, 0)
            series.append((mname, x, c, hi, yerr_lo, yerr_hi))

        if not series:
            continue

        max_hi = max(t[3] for t in series)

        for mname, x, c, hi, yerr_lo, yerr_hi in series:
            ax.errorbar(
                x, c, yerr=[[yerr_lo], [yerr_hi]],
                fmt=MODEL_MARKERS[mname], color=MODEL_COLORS[mname],
                markersize=8, capsize=4, capthick=1.3,
                elinewidth=1.8, markeredgewidth=0, zorder=5, alpha=1.0,
            )
            y_text = max_hi + label_dy + label_stagger[mname]
            ax.text(
                x,
                y_text,
                f"{c:.2f}",
                fontsize=9,
                fontweight="bold",
                color=MODEL_COLORS[mname],
                ha="center",
                va="bottom",
                zorder=6,
                clip_on=False,
            )

    ax.set_xticks(x_centers)
    xlabels = []
    for val in values:
        sub = merged[merged[col] == val]
        xlabels.append(f"{val}\n(n={len(sub)})")
    ax.set_xticklabels(xlabels, fontsize=12, ha="center")
    ax.set_xlim(x_centers.min() - x_margin, x_centers.max() + x_margin)
    ax.axhline(0.5, color="grey", linestyle="--", alpha=0.3)
    ax.set_ylim(0.35, 0.99)
    ax.set_ylabel("C-index", fontsize=12)
    ax.set_title(title, fontsize=12, fontweight="bold", y=1.20, pad=0)
    ax.tick_params(axis="y", labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ── Helper: country panel (Model 5 only, each country different color) ───────
def bootstrap_pooled_c(data, risk_col, n_boot=500, seed=42):
    """Pooled C-index with bootstrap 95% CI — more stable for small subgroups."""
    from lifelines.utils import concordance_index as ci
    rng = np.random.RandomState(seed)
    t = data[time_col].values.astype(float)
    y = data[evt_col].values.astype(float)
    r = data[risk_col].values
    v = ~np.isnan(r)
    t, y, r = t[v], y[v], r[v]
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    c_pooled = ci(np.clip(t, 0.5, None), -r, y)
    cs = []
    n = len(t)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            cs.append(ci(np.clip(t[idx], 0.5, None), -r[idx], y[idx]))
        except Exception:
            continue
    if cs:
        lo, hi = np.percentile(cs, [2.5, 97.5])
    else:
        lo, hi = np.nan, np.nan
    return c_pooled, lo, hi


def plot_country_panel(ax, title):
    """Model 5 only. Each country = different color dot, CI shown in legend."""
    countries = sorted(merged["COUNTRY"].dropna().unique())

    valid_countries = []
    for c in countries:
        sub = merged[merged["COUNTRY"] == c]
        if sub[evt_col].sum() >= 4:
            valid_countries.append(c)

    country_cs = []
    for c in valid_countries:
        sub = merged[merged["COUNTRY"] == c]
        mc, lo, hi = bootstrap_pooled_c(sub, "Model_5_risk")
        country_cs.append((c, mc, lo, hi, len(sub), int(sub[evt_col].sum())))
    country_cs.sort(key=lambda x: x[1] if not np.isnan(x[1]) else 0, reverse=True)

    for i, (c, mc, lo, hi, n, ev) in enumerate(country_cs):
        if np.isnan(mc):
            continue
        color = COUNTRY_COLORS.get(c, "#333")
        ax.plot(
            i, mc, marker="o", color=color,
            markersize=11, markeredgewidth=0.8, markeredgecolor="white",
            zorder=5, alpha=1.0, linestyle="none",
            label=f"{c} (n={n}, {ev}ev) C={mc:.2f} ({lo:.2f}\u2013{hi:.2f})",
        )

    ax.set_xticks(range(len(country_cs)))
    ax.set_xticklabels([x[0] for x in country_cs], fontsize=12, rotation=45, ha="right")
    ax.axhline(0.5, color="grey", linestyle="--", alpha=0.3)
    ax.set_ylim(0.35, 0.95)
    ax.set_ylabel("C-index", fontsize=12)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=110)
    ax.tick_params(axis="y", labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=12, loc="lower left", bbox_to_anchor=(0.0, 1.01),
              frameon=False, ncol=3)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE: 5 panels, 2 rows
# Row 1: Sex, Age, ECG status, AF status (4 panels, with 3 models + CI)
# Row 2: Country (full width, Model 5 only)
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 9.2))
gs = fig.add_gridspec(2, 4, hspace=0.6, wspace=0.35,
                      height_ratios=[0.85, 1.0])

# Derive AF status from any_af_clean
merged["af_status"] = merged["STUDYID"].map(
    raw.set_index("STUDYID")["any_af_clean"]
).map({0: "Non-AF", 1: "AF"})

# Row 1: 4 panels
ax_sex = fig.add_subplot(gs[0, 0])
ax_age = fig.add_subplot(gs[0, 1])
ax_ecg = fig.add_subplot(gs[0, 2])
ax_af = fig.add_subplot(gs[0, 3])

# Row 2: Country (full width)
ax_country = fig.add_subplot(gs[1, :])

# Plot
plot_multi_model_panel(ax_sex, "RQSEX", ["Male", "Female"], "A  Sex", show_legend=False)
plot_multi_model_panel(ax_age, "age_group", ["<65", "\u226565"], "B  Age", show_legend=False)
plot_multi_model_panel(ax_ecg, "ecg_type", ["Normal", "Abnormal"], "C  ECG status", show_legend=False)
plot_multi_model_panel(ax_af, "af_status", ["Non-AF", "AF"], "D  AF status", show_legend=False)
plot_country_panel(ax_country, "E  Country (Model 5 DL-ECG biomarkers)")

# Push panel E slightly downward and compress its height for cleaner separation
pos = ax_country.get_position()
ax_country.set_position([pos.x0, pos.y0 - 0.03, pos.width, pos.height * 0.88])

# Shared legend for panels A-D
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor=MODEL_COLORS["Model 3"],
           markersize=8, label="Model 3"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=MODEL_COLORS["Model 4"],
           markersize=8, label="Model 4"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=MODEL_COLORS["Model 5"],
           markersize=8, label="Model 5 (DL-ECG biomarkers)"),
]
fig.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.92),
    ncol=3,
    prop={"size": 12, "weight": "bold"},
    frameon=False,
    handletextpad=0.45,
    columnspacing=1.0,
)

fig.savefig(OUT_DIR / "fig5_subgroup_CVD4.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "fig5_subgroup_CVD4.png", dpi=300, bbox_inches="tight")
plt.close()

import shutil
FINAL_DIR = BASE / "results" / "manuscript_final" / "main_figures"
shutil.copy2(OUT_DIR / "fig5_subgroup_CVD4.pdf", FINAL_DIR)
shutil.copy2(OUT_DIR / "fig5_subgroup_CVD4.png", FINAL_DIR)

print(f"Saved -> {OUT_DIR / 'fig5_subgroup_CVD4.png'}")

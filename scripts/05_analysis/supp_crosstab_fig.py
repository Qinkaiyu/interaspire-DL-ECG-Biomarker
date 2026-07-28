"""
Supplementary Figure: Cross-tabulation heatmap (Model 3 × Model 5)
SEER Fig. 2 style — 3×3 grid with N and event rate per cell.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

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

for fid in sorted(merged["fold5"].unique()):
    mask = merged["fold5"] == fid
    vals = merged.loc[mask, "Model_5_risk"]
    merged.loc[mask, "Model_5_risk"] = (vals - vals.mean()) / vals.std()

y = merged[evt_col].values.astype(float)

# ── Tertile groups (percentile-based) ────────────────────────────────────────
m3_cuts = np.percentile(merged["Model_3_cox_prob"], [33.33, 66.67])
m5_cuts = np.percentile(merged["Model_5_risk"], [33.33, 66.67])

def assign_group(vals, cuts):
    return np.where(vals <= cuts[0], "Low", np.where(vals <= cuts[1], "Medium", "High"))

merged["m3_group"] = assign_group(merged["Model_3_cox_prob"].values, m3_cuts)
merged["m5_group"] = assign_group(merged["Model_5_risk"].values, m5_cuts)

groups = ["Low", "Medium", "High"]

# ── Build matrices ───────────────────────────────────────────────────────────
n_matrix = np.zeros((3, 3), dtype=int)
rate_matrix = np.zeros((3, 3))
ev_matrix = np.zeros((3, 3), dtype=int)

for i, g3 in enumerate(groups):
    for j, g5 in enumerate(groups):
        mask = (merged["m3_group"] == g3) & (merged["m5_group"] == g5)
        n = mask.sum()
        ev = int(merged.loc[mask, evt_col].sum())
        rate = ev / n * 100 if n > 0 else 0
        n_matrix[i, j] = n
        ev_matrix[i, j] = ev
        rate_matrix[i, j] = rate

# ── Row and column totals ────────────────────────────────────────────────────
row_n = n_matrix.sum(axis=1)
row_ev = ev_matrix.sum(axis=1)
row_rate = row_ev / row_n * 100

col_n = n_matrix.sum(axis=0)
col_ev = ev_matrix.sum(axis=0)
col_rate = col_ev / col_n * 100

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE: Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6.5))

# Color by event rate
cmap = plt.cm.RdYlGn_r  # red = high risk, green = low risk
norm = mcolors.Normalize(vmin=0, vmax=max(rate_matrix.max(), 12))

for i in range(3):
    for j in range(3):
        color = cmap(norm(rate_matrix[i, j]))
        rect = plt.Rectangle((j, 2 - i), 1, 1, facecolor=color, edgecolor="white", linewidth=3)
        ax.add_patch(rect)

        # Cell text
        n = n_matrix[i, j]
        ev = ev_matrix[i, j]
        rate = rate_matrix[i, j]

        # Choose text color based on background brightness
        brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = "white" if brightness < 0.55 else "black"

        ax.text(j + 0.5, 2 - i + 0.6, f"N = {n}", ha="center", va="center",
                fontsize=11, fontweight="bold", color=text_color)
        ax.text(j + 0.5, 2 - i + 0.38, f"{ev} events", ha="center", va="center",
                fontsize=9, color=text_color)
        ax.text(j + 0.5, 2 - i + 0.18, f"{rate:.1f}%", ha="center", va="center",
                fontsize=13, fontweight="bold", color=text_color)

# Row totals (right side)
for i in range(3):
    ax.text(3.3, 2 - i + 0.5, f"N={row_n[i]}\n{row_rate[i]:.1f}%",
            ha="center", va="center", fontsize=9, fontweight="bold", color="#555")

# Column totals (bottom)
for j in range(3):
    ax.text(j + 0.5, -0.3, f"N={col_n[j]}\n{col_rate[j]:.1f}%",
            ha="center", va="center", fontsize=9, fontweight="bold", color="#555")

# Labels
ax.set_xlim(0, 3)
ax.set_ylim(-0.6, 3.8)
ax.set_xticks([0.5, 1.5, 2.5])
ax.set_xticklabels(["Low", "Medium", "High"], fontsize=11, fontweight="bold")
ax.set_yticks([0.5, 1.5, 2.5])
ax.set_yticklabels(["High", "Medium", "Low"], fontsize=11, fontweight="bold")

ax.set_xlabel("Model 5 (DL-ECG) Risk Group", fontsize=12, fontweight="bold", labelpad=25)
ax.set_ylabel("Model 3 (Clinical) Risk Group", fontsize=12, fontweight="bold")

ax.set_title("Cross-tabulation of risk stratification\nModel 3 (Clinical) × Model 5 (DL-ECG)\n"
             f"Total N={len(merged)}, events={int(y.sum())} ({y.mean()*100:.1f}%)",
             fontsize=12, fontweight="bold")

# Highlight the key cell: M3=Low, M5=High
rect_highlight = plt.Rectangle((2, 2), 1, 1, fill=False, edgecolor="#FF3B30",
                                 linewidth=3, linestyle="--", zorder=10)
ax.add_patch(rect_highlight)
ax.annotate("Missed by\nclinical model",
            xy=(2.5, 3), xytext=(2.5, 3.45),
            ha="center", va="center", fontsize=8, fontweight="bold", color="#FF3B30",
            arrowprops=dict(arrowstyle="->", color="#FF3B30", lw=1.5))

# Highlight M3=High, M5=Low
rect_highlight2 = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#3498db",
                                  linewidth=3, linestyle="--", zorder=10)
ax.add_patch(rect_highlight2)
ax.annotate("Over-estimated\nby clinical model",
            xy=(0.5, 0), xytext=(0.5, -0.55),
            ha="center", va="center", fontsize=8, fontweight="bold", color="#3498db",
            arrowprops=dict(arrowstyle="->", color="#3498db", lw=1.5))

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label("1-year event rate (%)", fontsize=10)

ax.set_aspect("equal")
ax.tick_params(length=0)

plt.tight_layout()
fig.savefig(OUT_DIR / "supp_crosstab_heatmap_CVD4.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "supp_crosstab_heatmap_CVD4.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved -> {OUT_DIR / 'supp_crosstab_heatmap_CVD4.png'}")

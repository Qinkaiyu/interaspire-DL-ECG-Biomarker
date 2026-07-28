"""
Supplementary Figure: Cross-tabulation heatmap + NRI reclassification diagram
Two panels: A) Heatmap, B) Reclassification flow
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
N = len(merged)
N_events = int(y.sum())
N_nonevents = N - N_events

# ── Tertile groups ───────────────────────────────────────────────────────────
m3_cuts = np.percentile(merged["Model_3_cox_prob"], [33.33, 66.67])
m5_cuts = np.percentile(merged["Model_5_risk"], [33.33, 66.67])

def assign_group(vals, cuts):
    return np.where(vals <= cuts[0], "Low", np.where(vals <= cuts[1], "Medium", "High"))

merged["m3_group"] = assign_group(merged["Model_3_cox_prob"].values, m3_cuts)
merged["m5_group"] = assign_group(merged["Model_5_risk"].values, m5_cuts)

groups = ["Low", "Medium", "High"]
group_map = {"Low": 0, "Medium": 1, "High": 2}
m3_num = np.array([group_map[g] for g in merged["m3_group"]])
m5_num = np.array([group_map[g] for g in merged["m5_group"]])

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

# ── NRI calculations ─────────────────────────────────────────────────────────
evt_mask = y == 1
nonevt_mask = y == 0

evt_up = ((m5_num > m3_num) & evt_mask).sum()
evt_down = ((m5_num < m3_num) & evt_mask).sum()
evt_same = ((m5_num == m3_num) & evt_mask).sum()
nri_events = (evt_up - evt_down) / N_events

nonevt_down = ((m5_num < m3_num) & nonevt_mask).sum()
nonevt_up = ((m5_num > m3_num) & nonevt_mask).sum()
nonevt_same = ((m5_num == m3_num) & nonevt_mask).sum()
nri_nonevents = (nonevt_down - nonevt_up) / N_nonevents

nri = nri_events + nri_nonevents

# Bootstrap
rng = np.random.RandomState(42)
nri_boots = []
for _ in range(1000):
    idx = rng.choice(N, N, replace=True)
    yb, m3b, m5b = y[idx], m3_num[idx], m5_num[idx]
    ne, nne = yb.sum(), len(yb) - yb.sum()
    if ne == 0 or nne == 0: continue
    nri_b = (((m5b > m3b) & (yb == 1)).sum() - ((m5b < m3b) & (yb == 1)).sum()) / ne + \
            (((m5b < m3b) & (yb == 0)).sum() - ((m5b > m3b) & (yb == 0)).sum()) / nne
    nri_boots.append(nri_b)
nri_lo, nri_hi = np.percentile(nri_boots, [2.5, 97.5])
nri_p = (np.array(nri_boots) <= 0).mean()

# IDI
m5_prob_cal = 1 / (1 + np.exp(-merged["Model_5_risk"].values))
# Simple Platt for IDI display
from sklearn.linear_model import LogisticRegression
merged["m5_cal"] = np.nan
for fid in sorted(merged["fold5"].unique()):
    tmask = merged["fold5"] == fid
    trmask = ~tmask
    platt = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    platt.fit(merged.loc[trmask, "Model_5_risk"].values.reshape(-1,1), y[trmask])
    merged.loc[tmask, "m5_cal"] = platt.predict_proba(merged.loc[tmask, "Model_5_risk"].values.reshape(-1,1))[:,1]

m3p = merged["Model_3_cox_prob"].values
m5p = merged["m5_cal"].values
ds_m3 = m3p[evt_mask].mean() - m3p[nonevt_mask].mean()
ds_m5 = m5p[evt_mask].mean() - m5p[nonevt_mask].mean()
idi = (m5p[evt_mask].mean() - m3p[evt_mask].mean()) - (m5p[nonevt_mask].mean() - m3p[nonevt_mask].mean())

idi_boots = []
for _ in range(1000):
    idx = rng.choice(N, N, replace=True)
    yb = y[idx]; m3b = m3p[idx]; m5b = m5p[idx]
    em = yb == 1; nm = yb == 0
    if em.sum() == 0 or nm.sum() == 0: continue
    idi_boots.append((m5b[em].mean() - m3b[em].mean()) - (m5b[nm].mean() - m3b[nm].mean()))
idi_lo, idi_hi = np.percentile(idi_boots, [2.5, 97.5])
idi_p = (np.array(idi_boots) <= 0).mean()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE: 2 panels
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 7))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.35)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# ── Panel A: Heatmap ─────────────────────────────────────────────────────────
cmap = plt.cm.RdYlGn_r
norm = mcolors.Normalize(vmin=0, vmax=max(rate_matrix.max(), 12))

row_n = n_matrix.sum(axis=1)
row_ev = ev_matrix.sum(axis=1)
col_n = n_matrix.sum(axis=0)
col_ev = ev_matrix.sum(axis=0)

for i in range(3):
    for j in range(3):
        color = cmap(norm(rate_matrix[i, j]))
        rect = plt.Rectangle((j, 2 - i), 1, 1, facecolor=color, edgecolor="white", linewidth=3)
        ax1.add_patch(rect)

        n = n_matrix[i, j]
        ev = ev_matrix[i, j]
        rate = rate_matrix[i, j]
        brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        tc = "white" if brightness < 0.55 else "black"

        ax1.text(j + 0.5, 2 - i + 0.62, f"N={n}", ha="center", va="center",
                 fontsize=10, fontweight="bold", color=tc)
        ax1.text(j + 0.5, 2 - i + 0.4, f"{ev} events", ha="center", va="center",
                 fontsize=8, color=tc)
        ax1.text(j + 0.5, 2 - i + 0.2, f"{rate:.1f}%", ha="center", va="center",
                 fontsize=12, fontweight="bold", color=tc)

# Row/col totals
for i in range(3):
    ax1.text(3.25, 2 - i + 0.4, f"N={row_n[i]}\n{row_ev[i]/row_n[i]*100:.1f}%",
             ha="center", va="center", fontsize=8, color="#555")
for j in range(3):
    ax1.text(j + 0.5, -0.25, f"N={col_n[j]}\n{col_ev[j]/col_n[j]*100:.1f}%",
             ha="center", va="center", fontsize=8, color="#555")

# Highlight corners
rect_h = plt.Rectangle((2, 2), 1, 1, fill=False, edgecolor="#FF3B30", linewidth=2.5, linestyle="--", zorder=10)
ax1.add_patch(rect_h)
rect_h2 = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#3498db", linewidth=2.5, linestyle="--", zorder=10)
ax1.add_patch(rect_h2)

ax1.set_xlim(0, 3)
ax1.set_ylim(-0.5, 3.3)
ax1.set_xticks([0.5, 1.5, 2.5])
ax1.set_xticklabels(["Low", "Medium", "High"], fontsize=10, fontweight="bold")
ax1.set_yticks([0.5, 1.5, 2.5])
ax1.set_yticklabels(["High", "Medium", "Low"], fontsize=10, fontweight="bold")
ax1.set_xlabel("Model 5 (DL-ECG)", fontsize=11, fontweight="bold", labelpad=20)
ax1.set_ylabel("Model 3 (Clinical)", fontsize=11, fontweight="bold")
ax1.set_title("A  Cross-tabulation", fontsize=12, fontweight="bold", loc="left")
ax1.set_aspect("equal")
ax1.tick_params(length=0)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax1, shrink=0.5, pad=0.12)
cbar.set_label("Event rate (%)", fontsize=9)

# ── Panel B: NRI/IDI Summary ─────────────────────────────────────────────────
ax2.axis("off")

# Title
ax2.text(0.5, 0.95, "B  Reclassification & Discrimination Improvement",
         transform=ax2.transAxes, fontsize=12, fontweight="bold", ha="center", va="top")
ax2.text(0.5, 0.90, "Model 5 (DL-ECG) vs Model 3 (Clinical)",
         transform=ax2.transAxes, fontsize=10, ha="center", va="top", color="#555")

# Events box
box_y = 0.75
ax2.text(0.05, box_y, "Events", fontsize=11, fontweight="bold",
         transform=ax2.transAxes, va="top")
ax2.text(0.05, box_y - 0.05, f"(N = {N_events})", fontsize=9, color="#666",
         transform=ax2.transAxes, va="top")

ax2.text(0.08, box_y - 0.12, f"\u2191 Correctly reclassified up:", fontsize=10,
         transform=ax2.transAxes, va="top", color="#2ecc71")
ax2.text(0.55, box_y - 0.12, f"{evt_up}  ({evt_up/N_events*100:.1f}%)", fontsize=10,
         transform=ax2.transAxes, va="top", fontweight="bold", color="#2ecc71")

ax2.text(0.08, box_y - 0.19, f"\u2193 Incorrectly reclassified down:", fontsize=10,
         transform=ax2.transAxes, va="top", color="#e74c3c")
ax2.text(0.55, box_y - 0.19, f"{evt_down}  ({evt_down/N_events*100:.1f}%)", fontsize=10,
         transform=ax2.transAxes, va="top", fontweight="bold", color="#e74c3c")

ax2.text(0.08, box_y - 0.26, f"\u2194 Unchanged:", fontsize=10,
         transform=ax2.transAxes, va="top", color="#999")
ax2.text(0.55, box_y - 0.26, f"{evt_same}  ({evt_same/N_events*100:.1f}%)", fontsize=10,
         transform=ax2.transAxes, va="top", color="#999")

# Non-events box
box_y2 = 0.42
ax2.text(0.05, box_y2, "Non-events", fontsize=11, fontweight="bold",
         transform=ax2.transAxes, va="top")
ax2.text(0.05, box_y2 - 0.05, f"(N = {N_nonevents})", fontsize=9, color="#666",
         transform=ax2.transAxes, va="top")

ax2.text(0.08, box_y2 - 0.12, f"\u2193 Correctly reclassified down:", fontsize=10,
         transform=ax2.transAxes, va="top", color="#2ecc71")
ax2.text(0.55, box_y2 - 0.12, f"{nonevt_down}  ({nonevt_down/N_nonevents*100:.1f}%)", fontsize=10,
         transform=ax2.transAxes, va="top", fontweight="bold", color="#2ecc71")

ax2.text(0.08, box_y2 - 0.19, f"\u2191 Incorrectly reclassified up:", fontsize=10,
         transform=ax2.transAxes, va="top", color="#e74c3c")
ax2.text(0.55, box_y2 - 0.19, f"{nonevt_up}  ({nonevt_up/N_nonevents*100:.1f}%)", fontsize=10,
         transform=ax2.transAxes, va="top", fontweight="bold", color="#e74c3c")

ax2.text(0.08, box_y2 - 0.26, f"\u2194 Unchanged:", fontsize=10,
         transform=ax2.transAxes, va="top", color="#999")
ax2.text(0.55, box_y2 - 0.26, f"{nonevt_same}  ({nonevt_same/N_nonevents*100:.1f}%)", fontsize=10,
         transform=ax2.transAxes, va="top", color="#999")

# Summary metrics
box_y3 = 0.10
p_nri = "<0.001" if nri_p < 0.001 else f"{nri_p:.3f}"
p_idi = "<0.001" if idi_p < 0.001 else f"{idi_p:.3f}"

ax2.text(0.05, box_y3, "Summary Metrics", fontsize=11, fontweight="bold",
         transform=ax2.transAxes, va="top")

metrics_text = (
    f"Category NRI:   {nri:+.3f}  (95% CI {nri_lo:+.3f} to {nri_hi:+.3f},  P = {p_nri})\n"
    f"IDI:                    {idi:+.4f}  (95% CI {idi_lo:+.4f} to {idi_hi:+.4f},  P = {p_idi})"
)
ax2.text(0.08, box_y3 - 0.07, metrics_text, fontsize=10,
         transform=ax2.transAxes, va="top", family="monospace",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="#ccc"))

# Separator lines
for yy in [box_y + 0.02, box_y2 + 0.02, box_y3 + 0.02]:
    ax2.plot([0.02, 0.98], [yy, yy], color="#ddd", linewidth=1,
             transform=ax2.transAxes, clip_on=False)

fig.savefig(OUT_DIR / "supp_crosstab_nri_CVD4.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "supp_crosstab_nri_CVD4.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved -> {OUT_DIR / 'supp_crosstab_nri_CVD4.png'}")

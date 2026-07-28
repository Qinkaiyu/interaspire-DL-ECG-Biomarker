"""
Fig. 3 (CORE): C-index incremental plot — AIRE Fig. 4B style
Dot plot showing Models 0–4 with bootstrap 95% CI.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index

BASE = Path(".")
OOF_PATH = BASE / "results" / "oof_predictions" / "oof_CVD4.csv"
OUT_DIR = BASE / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOT = 1000
RANDOM_STATE = 42

# ── Load oof predictions ─────────────────────────────────────────────────────
df = pd.read_csv(OOF_PATH)
evt_col = "event_CVD_Composite_4"
time_col = "time_Composite_4"

LANDMARK_T = 365
df["landmark"] = (df[time_col] >= LANDMARK_T) | (df[evt_col] == 1)
df["y_1yr"] = ((df[evt_col] == 1) & (df[time_col] <= LANDMARK_T)).astype(float)

print(f"N={len(df)}, events={int(df[evt_col].sum())}, N_landmark={int(df['landmark'].sum())}")

# ── Model display names ──────────────────────────────────────────────────────
MODELS = [
    ("Model_0", "Age + Sex"),
    ("Model_1", "+ CV risk factors"),
    ("Model_2", "+ Medical history & Medications"),
    ("Model_3", "+ Index event type"),
    ("Model_4", "+ ECG biomarkers"),
]

# ── Bootstrap CI ─────────────────────────────────────────────────────────────
def bootstrap_auc(y_true, y_score, n_boot=N_BOOT, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    return np.percentile(aucs, [2.5, 97.5])


def bootstrap_cindex(time, risk, event, n_boot=N_BOOT, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    cs = []
    n = len(time)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try:
            c = concordance_index(
                np.clip(time[idx], 0.5, None), -risk[idx], event[idx]
            )
            cs.append(c)
        except Exception:
            continue
    return np.percentile(cs, [2.5, 97.5])


# ── Calculate metrics ────────────────────────────────────────────────────────
results = []

for mname, display in MODELS:
    prob_col = f"{mname}_cox_prob"
    cox_col = f"{mname}_cox_risk"

    lm = df["landmark"].values
    valid_prob = ~df[prob_col].isna() & lm
    valid_cox = ~df[cox_col].isna()

    y_1yr = df["y_1yr"].values.astype(float)
    y = df[evt_col].values.astype(float)
    t = df[time_col].values.astype(float)

    # AUC on landmark complete-case subset
    auc = roc_auc_score(y_1yr[valid_prob], df.loc[valid_prob, prob_col])
    auc_ci = bootstrap_auc(y_1yr[valid_prob], df.loc[valid_prob, prob_col].values)

    # C-index
    c_idx = concordance_index(
        np.clip(t[valid_cox], 0.5, None),
        -df.loc[valid_cox, cox_col].values,
        y[valid_cox]
    )
    c_ci = bootstrap_cindex(
        t[valid_cox], df.loc[valid_cox, cox_col].values, y[valid_cox]
    )

    results.append({
        "model": mname,
        "display": display,
        "auc": auc, "auc_lo": auc_ci[0], "auc_hi": auc_ci[1],
        "c_index": c_idx, "c_lo": c_ci[0], "c_hi": c_ci[1],
    })

    print(f"  {display:40s}  AUC={auc:.3f} [{auc_ci[0]:.3f}–{auc_ci[1]:.3f}]  "
          f"C={c_idx:.3f} [{c_ci[0]:.3f}–{c_ci[1]:.3f}]")

res = pd.DataFrame(results)

# ── Plot: AIRE Fig. 4B style ─────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)

# Color palette
colors = ["#7f8c8d", "#3498db", "#2ecc71", "#e67e22", "#e74c3c"]
y_pos = np.arange(len(res))[::-1]  # top to bottom: Model 0 at top

# ── Panel A: AUC (Logistic Regression) ───────────────────────────────────────
for i, (_, row) in enumerate(res.iterrows()):
    xerr_lo = row["auc"] - row["auc_lo"]
    xerr_hi = row["auc_hi"] - row["auc"]
    ax1.errorbar(
        row["auc"], y_pos[i],
        xerr=[[xerr_lo], [xerr_hi]],
        fmt="o", color=colors[i], markersize=10, capsize=5,
        elinewidth=2, markeredgewidth=0, zorder=3,
    )
    # Value label
    ax1.text(
        row["auc_hi"] + 0.005, y_pos[i],
        f'{row["auc"]:.3f}',
        va="center", ha="left", fontsize=9, color=colors[i], fontweight="bold",
    )

ax1.set_yticks(y_pos)
ax1.set_yticklabels([r["display"] for _, r in res.iterrows()], fontsize=10)
ax1.set_xlabel("AUC@1yr (Cox PH)", fontsize=11, fontweight="bold")
ax1.set_title("A  Discrimination — AUC@1yr", fontsize=12, fontweight="bold", loc="left")
ax1.axvline(0.5, color="grey", linestyle="--", alpha=0.4, zorder=1)
ax1.set_xlim(0.50, 0.78)
ax1.grid(axis="x", alpha=0.3)
ax1.xaxis.set_major_locator(mticker.MultipleLocator(0.05))

# ── Panel B: C-index (Cox PH) ────────────────────────────────────────────────
for i, (_, row) in enumerate(res.iterrows()):
    xerr_lo = row["c_index"] - row["c_lo"]
    xerr_hi = row["c_hi"] - row["c_index"]
    ax2.errorbar(
        row["c_index"], y_pos[i],
        xerr=[[xerr_lo], [xerr_hi]],
        fmt="s", color=colors[i], markersize=10, capsize=5,
        elinewidth=2, markeredgewidth=0, zorder=3,
    )
    ax2.text(
        row["c_hi"] + 0.005, y_pos[i],
        f'{row["c_index"]:.3f}',
        va="center", ha="left", fontsize=9, color=colors[i], fontweight="bold",
    )

ax2.set_xlabel("C-index (Cox Proportional Hazards)", fontsize=11, fontweight="bold")
ax2.set_title("B  Discrimination — C-index", fontsize=12, fontweight="bold", loc="left")
ax2.axvline(0.5, color="grey", linestyle="--", alpha=0.4, zorder=1)
ax2.set_xlim(0.50, 0.78)
ax2.grid(axis="x", alpha=0.3)
ax2.xaxis.set_major_locator(mticker.MultipleLocator(0.05))

# ── Annotations: delta ───────────────────────────────────────────────────────
# Add delta AUC annotations between consecutive models
for i in range(1, len(res)):
    delta_auc = res.iloc[i]["auc"] - res.iloc[i-1]["auc"]
    delta_c = res.iloc[i]["c_index"] - res.iloc[i-1]["c_index"]

    mid_y = (y_pos[i] + y_pos[i-1]) / 2
    if abs(delta_auc) >= 0.001:
        ax1.annotate(
            f"Δ={delta_auc:+.3f}",
            xy=(0.505, mid_y), fontsize=7.5, color="#555",
            va="center",
        )
    if abs(delta_c) >= 0.001:
        ax2.annotate(
            f"Δ={delta_c:+.3f}",
            xy=(0.505, mid_y), fontsize=7.5, color="#555",
            va="center",
        )

plt.tight_layout()
fig.savefig(OUT_DIR / "fig3_cindex_incremental_CVD4.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "fig3_cindex_incremental_CVD4.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSaved → {OUT_DIR / 'fig3_cindex_incremental_CVD4.pdf'}")
print(f"Saved → {OUT_DIR / 'fig3_cindex_incremental_CVD4.png'}")

# ── Save numeric results ─────────────────────────────────────────────────────
res_out = res.copy()
res_out["AUC (95% CI)"] = res_out.apply(
    lambda r: f'{r["auc"]:.3f} ({r["auc_lo"]:.3f}–{r["auc_hi"]:.3f})', axis=1
)
res_out["C-index (95% CI)"] = res_out.apply(
    lambda r: f'{r["c_index"]:.3f} ({r["c_lo"]:.3f}–{r["c_hi"]:.3f})', axis=1
)
res_out[["display", "AUC (95% CI)", "C-index (95% CI)"]].to_csv(
    OUT_DIR / "fig3_cindex_incremental_CVD4_data.csv", index=False
)
print("\nResults table:")
for _, r in res_out.iterrows():
    print(f'  {r["display"]:40s}  AUC={r["AUC (95% CI)"]}  C={r["C-index (95% CI)"]}')

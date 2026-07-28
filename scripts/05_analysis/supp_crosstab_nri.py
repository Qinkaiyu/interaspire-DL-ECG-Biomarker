"""
Supplementary: Cross-tabulation (Model 3 × Model 5) + NRI / IDI
SEER Fig. 2 style — risk group cross-table with event rates.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats

BASE = Path(".")
OOF_M04 = BASE / "results" / "oof_predictions" / "oof_CVD4.csv"
OOF_M5 = BASE / "model5_cvd_mace4" / "E2E_MultiTask_32_2_phase_cvd_mace4_1yr_oof.csv"
OUT_DIR = BASE / "results" / "supplementary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

evt_col = "event_CVD_Composite_4"
time_col = "time_Composite_4"

# ── Load & merge ─────────────────────────────────────────────────────────────
df04 = pd.read_csv(OOF_M04)
df5 = pd.read_csv(OOF_M5)
df5 = df5.rename(columns={"oof_log_risk": "Model_5_risk", "test_fold": "fold5"})
merged = df04.merge(df5[["STUDYID", "Model_5_risk", "fold5"]], on="STUDYID", how="inner")

# Normalize Model 5 per fold
for fid in sorted(merged["fold5"].unique()):
    mask = merged["fold5"] == fid
    vals = merged.loc[mask, "Model_5_risk"]
    merged.loc[mask, "Model_5_risk"] = (vals - vals.mean()) / vals.std()

y = merged[evt_col].values.astype(float)
N = len(merged)
N_events = int(y.sum())
N_nonevents = N - N_events

print(f"N={N}, events={N_events}, non-events={N_nonevents}")

# ── Tertile risk groups (percentile-based, equal N) ──────────────────────────
# Model 3: use LR predicted probability
m3_prob = merged["Model_3_cox_prob"].values
m5_risk = merged["Model_5_risk"].values

# Tertile cutpoints
m3_cuts = np.percentile(m3_prob, [33.33, 66.67])
m5_cuts = np.percentile(m5_risk, [33.33, 66.67])

def assign_group(vals, cuts):
    return np.where(vals <= cuts[0], "Low",
           np.where(vals <= cuts[1], "Medium", "High"))

merged["m3_group"] = assign_group(m3_prob, m3_cuts)
merged["m5_group"] = assign_group(m5_risk, m5_cuts)

# ══════════════════════════════════════════════════════════════════════════════
# CROSS-TABULATION TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CROSS-TABULATION: Model 3 (Clinical) × Model 5 (DL-ECG)")
print("=" * 70)

groups = ["Low", "Medium", "High"]
print(f"\n{'':20s}", end="")
for g5 in groups:
    print(f"{'Model 5 ' + g5:>18s}", end="")
print(f"{'Row Total':>14s}")
print("-" * 72)

cross_data = []
for g3 in groups:
    print(f"Model 3 {g3:<10s}", end="")
    row_n = 0
    row_ev = 0
    for g5 in groups:
        mask = (merged["m3_group"] == g3) & (merged["m5_group"] == g5)
        n = mask.sum()
        ev = int(merged.loc[mask, evt_col].sum())
        rate = ev / n * 100 if n > 0 else 0
        row_n += n
        row_ev += ev
        print(f"  {n:4d} ({rate:4.1f}%)", end="    ")
        cross_data.append({
            "Model_3": g3, "Model_5": g5,
            "N": n, "Events": ev, "Rate": round(rate, 1),
        })
    row_rate = row_ev / row_n * 100 if row_n > 0 else 0
    print(f"  {row_n:4d} ({row_rate:4.1f}%)")

# Column totals
print(f"{'Col Total':<20s}", end="")
for g5 in groups:
    mask = merged["m5_group"] == g5
    n = mask.sum()
    ev = int(merged.loc[mask, evt_col].sum())
    rate = ev / n * 100 if n > 0 else 0
    print(f"  {n:4d} ({rate:4.1f}%)", end="    ")
print()

cross_df = pd.DataFrame(cross_data)
cross_df.to_csv(OUT_DIR / "crosstab_m3_m5_CVD4.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# NRI (Category-based)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("NRI: Model 5 vs Model 3 (category-based, tertile)")
print("=" * 70)

# Map groups to numeric: Low=0, Medium=1, High=2
group_map = {"Low": 0, "Medium": 1, "High": 2}
m3_num = np.array([group_map[g] for g in merged["m3_group"]])
m5_num = np.array([group_map[g] for g in merged["m5_group"]])

# Events
evt_mask = y == 1
evt_up = ((m5_num > m3_num) & evt_mask).sum()     # correctly reclassified up
evt_down = ((m5_num < m3_num) & evt_mask).sum()   # incorrectly reclassified down
evt_same = ((m5_num == m3_num) & evt_mask).sum()

nri_events = (evt_up - evt_down) / N_events

# Non-events
nonevt_mask = y == 0
nonevt_down = ((m5_num < m3_num) & nonevt_mask).sum()   # correctly down
nonevt_up = ((m5_num > m3_num) & nonevt_mask).sum()     # incorrectly up
nonevt_same = ((m5_num == m3_num) & nonevt_mask).sum()

nri_nonevents = (nonevt_down - nonevt_up) / N_nonevents

nri = nri_events + nri_nonevents

print(f"\nEvents (N={N_events}):")
print(f"  Reclassified up (correct):   {evt_up:4d} ({evt_up/N_events*100:.1f}%)")
print(f"  Reclassified down (incorrect): {evt_down:4d} ({evt_down/N_events*100:.1f}%)")
print(f"  Unchanged:                   {evt_same:4d} ({evt_same/N_events*100:.1f}%)")
print(f"  NRI(events) = {nri_events:+.4f}")

print(f"\nNon-events (N={N_nonevents}):")
print(f"  Reclassified down (correct): {nonevt_down:4d} ({nonevt_down/N_nonevents*100:.1f}%)")
print(f"  Reclassified up (incorrect): {nonevt_up:4d} ({nonevt_up/N_nonevents*100:.1f}%)")
print(f"  Unchanged:                   {nonevt_same:4d} ({nonevt_same/N_nonevents*100:.1f}%)")
print(f"  NRI(non-events) = {nri_nonevents:+.4f}")

print(f"\n  Category NRI = {nri:+.4f}")

# Bootstrap 95% CI for NRI
rng = np.random.RandomState(42)
nri_boots = []
for _ in range(1000):
    idx = rng.choice(N, N, replace=True)
    yb = y[idx]
    m3b = m3_num[idx]
    m5b = m5_num[idx]
    ne = yb.sum()
    nne = len(yb) - ne
    if ne == 0 or nne == 0:
        continue
    up_e = ((m5b > m3b) & (yb == 1)).sum()
    dn_e = ((m5b < m3b) & (yb == 1)).sum()
    dn_ne = ((m5b < m3b) & (yb == 0)).sum()
    up_ne = ((m5b > m3b) & (yb == 0)).sum()
    nri_b = (up_e - dn_e) / ne + (dn_ne - up_ne) / nne
    nri_boots.append(nri_b)

nri_lo, nri_hi = np.percentile(nri_boots, [2.5, 97.5])
nri_p = (np.array(nri_boots) <= 0).mean()  # one-sided p
print(f"  95% CI: ({nri_lo:+.4f}, {nri_hi:+.4f})")
print(f"  P-value (NRI>0): {nri_p:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# Category-free NRI (continuous)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Category-free NRI (continuous)")
print("=" * 70)

# For events: proportion where Model 5 assigns higher risk than Model 3
# Use rank-based: compare whether predicted risk increased
# Model 3 prob vs Model 5 risk (both higher = more risk)
m3_prob_all = merged["Model_3_cox_prob"].values
m5_risk_all = merged["Model_5_risk"].values

# Convert to ranks for fair comparison
from scipy.stats import rankdata
m3_rank = rankdata(m3_prob_all)
m5_rank = rankdata(m5_risk_all)

# Events: proportion moving up
evt_up_cont = ((m5_rank > m3_rank) & evt_mask).sum()
evt_down_cont = ((m5_rank < m3_rank) & evt_mask).sum()
nri_events_cont = (evt_up_cont - evt_down_cont) / N_events

# Non-events: proportion moving down
nonevt_down_cont = ((m5_rank < m3_rank) & nonevt_mask).sum()
nonevt_up_cont = ((m5_rank > m3_rank) & nonevt_mask).sum()
nri_nonevents_cont = (nonevt_down_cont - nonevt_up_cont) / N_nonevents

nri_cont = nri_events_cont + nri_nonevents_cont

print(f"  NRI(events) = {nri_events_cont:+.4f}")
print(f"  NRI(non-events) = {nri_nonevents_cont:+.4f}")
print(f"  Category-free NRI = {nri_cont:+.4f}")

# Bootstrap
nri_cont_boots = []
for _ in range(1000):
    idx = rng.choice(N, N, replace=True)
    yb = y[idx]
    m3r = m3_rank[idx]
    m5r = m5_rank[idx]
    ne = yb.sum()
    nne = len(yb) - ne
    if ne == 0 or nne == 0:
        continue
    up_e = ((m5r > m3r) & (yb == 1)).sum()
    dn_e = ((m5r < m3r) & (yb == 1)).sum()
    dn_ne = ((m5r < m3r) & (yb == 0)).sum()
    up_ne = ((m5r > m3r) & (yb == 0)).sum()
    nri_cont_boots.append((up_e - dn_e) / ne + (dn_ne - up_ne) / nne)

nri_cont_lo, nri_cont_hi = np.percentile(nri_cont_boots, [2.5, 97.5])
nri_cont_p = (np.array(nri_cont_boots) <= 0).mean()
print(f"  95% CI: ({nri_cont_lo:+.4f}, {nri_cont_hi:+.4f})")
print(f"  P-value: {nri_cont_p:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# IDI
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("IDI: Model 5 vs Model 3")
print("=" * 70)

# Need predicted probabilities on same scale
# Model 3: LR prob (already probability)
# Model 5: sigmoid of normalized risk
m5_prob = 1 / (1 + np.exp(-m5_risk_all))

# IDI = (mean prob increase in events) - (mean prob increase in non-events)
# = [mean_M5(events) - mean_M3(events)] - [mean_M5(nonevents) - mean_M3(nonevents)]

m3_mean_evt = m3_prob_all[evt_mask].mean()
m5_mean_evt = m5_prob[evt_mask].mean()
m3_mean_nonevt = m3_prob_all[nonevt_mask].mean()
m5_mean_nonevt = m5_prob[nonevt_mask].mean()

idi = (m5_mean_evt - m3_mean_evt) - (m5_mean_nonevt - m3_mean_nonevt)

# Discrimination slope
ds_m3 = m3_mean_evt - m3_mean_nonevt
ds_m5 = m5_mean_evt - m5_mean_nonevt

print(f"  Model 3: mean prob (events)={m3_mean_evt:.4f}, (non-events)={m3_mean_nonevt:.4f}, DS={ds_m3:.4f}")
print(f"  Model 5: mean prob (events)={m5_mean_evt:.4f}, (non-events)={m5_mean_nonevt:.4f}, DS={ds_m5:.4f}")
print(f"  IDI = {idi:+.4f}")

# Bootstrap
idi_boots = []
for _ in range(1000):
    idx = rng.choice(N, N, replace=True)
    yb = y[idx]
    m3b = m3_prob_all[idx]
    m5b = m5_prob[idx]
    em = yb == 1
    nm = yb == 0
    if em.sum() == 0 or nm.sum() == 0:
        continue
    idi_b = (m5b[em].mean() - m3b[em].mean()) - (m5b[nm].mean() - m3b[nm].mean())
    idi_boots.append(idi_b)

idi_lo, idi_hi = np.percentile(idi_boots, [2.5, 97.5])
idi_p = (np.array(idi_boots) <= 0).mean()
print(f"  95% CI: ({idi_lo:+.4f}, {idi_hi:+.4f})")
print(f"  P-value: {idi_p:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
summary = pd.DataFrame([
    {"Metric": "Category NRI (tertile)", "Value": f"{nri:+.4f}", "95% CI": f"({nri_lo:+.4f}, {nri_hi:+.4f})", "P": f"{nri_p:.4f}"},
    {"Metric": "Category-free NRI", "Value": f"{nri_cont:+.4f}", "95% CI": f"({nri_cont_lo:+.4f}, {nri_cont_hi:+.4f})", "P": f"{nri_cont_p:.4f}"},
    {"Metric": "IDI", "Value": f"{idi:+.4f}", "95% CI": f"({idi_lo:+.4f}, {idi_hi:+.4f})", "P": f"{idi_p:.4f}"},
])
summary.to_csv(OUT_DIR / "nri_idi_CVD4.csv", index=False)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(summary.to_string(index=False))
print(f"\nSaved -> {OUT_DIR}")

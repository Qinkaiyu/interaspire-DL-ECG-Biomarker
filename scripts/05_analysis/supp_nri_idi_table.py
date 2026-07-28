"""
Supplementary Table: NRI/IDI
Model 5 vs Model 3, Model 5 vs Model 4
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression

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

# Platt-calibrated Model 5 probability
merged["m5_cal"] = np.nan
for fid in sorted(merged["fold5"].unique()):
    tmask = merged["fold5"] == fid
    trmask = ~tmask
    platt = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    platt.fit(merged.loc[trmask, "Model_5_risk"].values.reshape(-1, 1), y[trmask])
    merged.loc[tmask, "m5_cal"] = platt.predict_proba(
        merged.loc[tmask, "Model_5_risk"].values.reshape(-1, 1))[:, 1]

rng = np.random.RandomState(42)


def compute_nri_idi(old_prob, new_prob, old_label, new_label):
    """Compute category NRI, category-free NRI, and IDI with bootstrap CI."""
    from scipy.stats import rankdata

    # ── Category NRI (tertile) ───────────────────────────────────────────
    old_cuts = np.percentile(old_prob, [33.33, 66.67])
    new_cuts = np.percentile(new_prob, [33.33, 66.67])

    def to_num(vals, cuts):
        return np.where(vals <= cuts[0], 0, np.where(vals <= cuts[1], 1, 2))

    old_num = to_num(old_prob, old_cuts)
    new_num = to_num(new_prob, new_cuts)

    evt_mask = y == 1
    nonevt_mask = y == 0

    evt_up = ((new_num > old_num) & evt_mask).sum()
    evt_down = ((new_num < old_num) & evt_mask).sum()
    evt_same = ((new_num == old_num) & evt_mask).sum()
    nri_events = (evt_up - evt_down) / N_events

    nonevt_down = ((new_num < old_num) & nonevt_mask).sum()
    nonevt_up = ((new_num > old_num) & nonevt_mask).sum()
    nonevt_same = ((new_num == old_num) & nonevt_mask).sum()
    nri_nonevents = (nonevt_down - nonevt_up) / N_nonevents

    cat_nri = nri_events + nri_nonevents

    # Bootstrap
    cat_nri_boots = []
    for _ in range(1000):
        idx = rng.choice(N, N, replace=True)
        yb, ob, nb = y[idx], old_num[idx], new_num[idx]
        ne, nne = yb.sum(), len(yb) - yb.sum()
        if ne == 0 or nne == 0: continue
        b = (((nb > ob) & (yb == 1)).sum() - ((nb < ob) & (yb == 1)).sum()) / ne + \
            (((nb < ob) & (yb == 0)).sum() - ((nb > ob) & (yb == 0)).sum()) / nne
        cat_nri_boots.append(b)
    cat_nri_lo, cat_nri_hi = np.percentile(cat_nri_boots, [2.5, 97.5])
    cat_nri_p = (np.array(cat_nri_boots) <= 0).mean()

    # ── Category-free NRI ────────────────────────────────────────────────
    old_rank = rankdata(old_prob)
    new_rank = rankdata(new_prob)

    cf_evt_up = ((new_rank > old_rank) & evt_mask).sum()
    cf_evt_down = ((new_rank < old_rank) & evt_mask).sum()
    cf_nri_events = (cf_evt_up - cf_evt_down) / N_events

    cf_nonevt_down = ((new_rank < old_rank) & nonevt_mask).sum()
    cf_nonevt_up = ((new_rank > old_rank) & nonevt_mask).sum()
    cf_nri_nonevents = (cf_nonevt_down - cf_nonevt_up) / N_nonevents

    cf_nri = cf_nri_events + cf_nri_nonevents

    cf_nri_boots = []
    for _ in range(1000):
        idx = rng.choice(N, N, replace=True)
        yb = y[idx]; or_ = old_rank[idx]; nr_ = new_rank[idx]
        ne, nne = yb.sum(), len(yb) - yb.sum()
        if ne == 0 or nne == 0: continue
        b = (((nr_ > or_) & (yb == 1)).sum() - ((nr_ < or_) & (yb == 1)).sum()) / ne + \
            (((nr_ < or_) & (yb == 0)).sum() - ((nr_ > or_) & (yb == 0)).sum()) / nne
        cf_nri_boots.append(b)
    cf_nri_lo, cf_nri_hi = np.percentile(cf_nri_boots, [2.5, 97.5])
    cf_nri_p = (np.array(cf_nri_boots) <= 0).mean()

    # ── IDI ──────────────────────────────────────────────────────────────
    idi = (new_prob[evt_mask].mean() - old_prob[evt_mask].mean()) - \
          (new_prob[nonevt_mask].mean() - old_prob[nonevt_mask].mean())

    idi_boots = []
    for _ in range(1000):
        idx = rng.choice(N, N, replace=True)
        yb = y[idx]; ob = old_prob[idx]; nb = new_prob[idx]
        em = yb == 1; nm = yb == 0
        if em.sum() == 0 or nm.sum() == 0: continue
        idi_boots.append((nb[em].mean() - ob[em].mean()) - (nb[nm].mean() - ob[nm].mean()))
    idi_lo, idi_hi = np.percentile(idi_boots, [2.5, 97.5])
    idi_p = (np.array(idi_boots) <= 0).mean()

    return {
        "comparison": f"{new_label} vs {old_label}",
        # Category NRI
        "evt_up": evt_up, "evt_down": evt_down, "evt_same": evt_same,
        "nonevt_down": nonevt_down, "nonevt_up": nonevt_up, "nonevt_same": nonevt_same,
        "nri_events": nri_events, "nri_nonevents": nri_nonevents,
        "cat_nri": cat_nri, "cat_nri_ci": f"({cat_nri_lo:+.3f}, {cat_nri_hi:+.3f})", "cat_nri_p": cat_nri_p,
        # Category-free NRI
        "cf_nri": cf_nri, "cf_nri_ci": f"({cf_nri_lo:+.3f}, {cf_nri_hi:+.3f})", "cf_nri_p": cf_nri_p,
        # IDI
        "idi": idi, "idi_ci": f"({idi_lo:+.4f}, {idi_hi:+.4f})", "idi_p": idi_p,
    }


# ── Compute for both comparisons ─────────────────────────────────────────────
m3_prob = merged["Model_3_cox_prob"].values
m4_prob = merged["Model_4_cox_prob"].values
m5_prob = merged["m5_cal"].values

res_5v3 = compute_nri_idi(m3_prob, m5_prob, "Model 3", "Model 5")
res_5v4 = compute_nri_idi(m4_prob, m5_prob, "Model 4", "Model 5")

# ── Print tables ─────────────────────────────────────────────────────────────
for res in [res_5v3, res_5v4]:
    print(f"\n{'='*70}")
    print(f"  {res['comparison']}")
    print(f"{'='*70}")

    print(f"\n  Events (N={N_events}):")
    print(f"    Reclassified up (correct):     {res['evt_up']:4d} ({res['evt_up']/N_events*100:.1f}%)")
    print(f"    Reclassified down (incorrect): {res['evt_down']:4d} ({res['evt_down']/N_events*100:.1f}%)")
    print(f"    Unchanged:                     {res['evt_same']:4d} ({res['evt_same']/N_events*100:.1f}%)")

    print(f"\n  Non-events (N={N_nonevents}):")
    print(f"    Reclassified down (correct):   {res['nonevt_down']:4d} ({res['nonevt_down']/N_nonevents*100:.1f}%)")
    print(f"    Reclassified up (incorrect):   {res['nonevt_up']:4d} ({res['nonevt_up']/N_nonevents*100:.1f}%)")
    print(f"    Unchanged:                     {res['nonevt_same']:4d} ({res['nonevt_same']/N_nonevents*100:.1f}%)")

    p_cat = "<0.001" if res["cat_nri_p"] < 0.001 else f"{res['cat_nri_p']:.3f}"
    p_cf = "<0.001" if res["cf_nri_p"] < 0.001 else f"{res['cf_nri_p']:.3f}"
    p_idi = "<0.001" if res["idi_p"] < 0.001 else f"{res['idi_p']:.3f}"

    print(f"\n  Category NRI:      {res['cat_nri']:+.3f}  95% CI {res['cat_nri_ci']}  P={p_cat}")
    print(f"  Category-free NRI: {res['cf_nri']:+.3f}  95% CI {res['cf_nri_ci']}  P={p_cf}")
    print(f"  IDI:               {res['idi']:+.4f}  95% CI {res['idi_ci']}  P={p_idi}")

# ── Save CSV ─────────────────────────────────────────────────────────────────
summary_rows = []
for res in [res_5v3, res_5v4]:
    p_cat = "<0.001" if res["cat_nri_p"] < 0.001 else f"{res['cat_nri_p']:.3f}"
    p_cf = "<0.001" if res["cf_nri_p"] < 0.001 else f"{res['cf_nri_p']:.3f}"
    p_idi = "<0.001" if res["idi_p"] < 0.001 else f"{res['idi_p']:.3f}"
    summary_rows.extend([
        {"Comparison": res["comparison"], "Metric": "Category NRI",
         "Value": f"{res['cat_nri']:+.3f}", "95% CI": res["cat_nri_ci"], "P": p_cat},
        {"Comparison": res["comparison"], "Metric": "Category-free NRI",
         "Value": f"{res['cf_nri']:+.3f}", "95% CI": res["cf_nri_ci"], "P": p_cf},
        {"Comparison": res["comparison"], "Metric": "IDI",
         "Value": f"{res['idi']:+.4f}", "95% CI": res["idi_ci"], "P": p_idi},
    ])

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / "nri_idi_table_CVD4.csv", index=False)
print(f"\nSaved -> {OUT_DIR / 'nri_idi_table_CVD4.csv'}")
print("\n" + summary_df.to_string(index=False))

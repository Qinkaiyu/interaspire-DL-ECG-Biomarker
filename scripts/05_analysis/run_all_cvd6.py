"""
Run all analyses for CVD_Composite_6 sensitivity analysis.
Mirrors CVD4 analyses: Fig 3-7 + Supp.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from scipy.stats import spearmanr, pointbiserialr
import time

BASE = Path(".")
DATA_PATH = BASE / "data" / "INTERASPIRE_analysis_dataset.xlsx"
OOF_M04 = BASE / "results" / "oof_predictions" / "oof_CVD6.csv"
OOF_M5 = BASE / "model5_cvd_mace6" / "E2E_MultiTask_32_2_phase_cvd_mace6_1yr_oof.csv"
OUT_DIR = BASE / "results" / "sensitivity_CVD6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

evt_col = "event_CVD_Composite_6"
time_col = "time_Composite_6"
N_BOOT = 1000
RANDOM_STATE = 42

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# ── Load & merge ─────────────────────────────────────────────────────────────
log("Loading data...")
raw = pd.read_excel(DATA_PATH, engine="openpyxl")
df04 = pd.read_csv(OOF_M04)
df5 = pd.read_csv(OOF_M5)
df5 = df5.rename(columns={"oof_log_risk": "Model_5_risk", "test_fold": "fold5"})
merged = df04.merge(df5[["STUDYID", "Model_5_risk", "fold5"]], on="STUDYID", how="inner")

# Normalize Model 5 per fold
for fid in sorted(merged["fold5"].unique()):
    mask = merged["fold5"] == fid
    vals = merged.loc[mask, "Model_5_risk"]
    merged.loc[mask, "Model_5_risk"] = (vals - vals.mean()) / vals.std()

merged["Model_5_prob"] = 1 / (1 + np.exp(-merged["Model_5_risk"]))

# Merge raw data
raw_cols = ["STUDYID", "RQSEX", "AGE", "COUNTRY", "RQINDEX",
            "Sinus Rhythm", "Atrial Fibrillation", "Atrial Flutter",
            "Supraventricular Tachycardia", "RBBB", "LBBB",
            "1 AV Block", "2 AV Block", "3 AV Block",
            "Left Axis Deviation", "Right Axis Deviation", "Q Wave",
            "ST Elevation", "ST Depression", "T Wave Inversion",
            "Ischaemic", "LVH", "RVH", "QT Prolongation", "MI (Old)", "MI(Acute)",
            "any_af_clean"]
raw_cols = [c for c in raw_cols if c in raw.columns]
merged = merged.merge(raw[raw_cols], on="STUDYID", how="left")

y = merged[evt_col].values.astype(float)
t = merged[time_col].values.astype(float)
N = len(merged)
log(f"N={N}, events={int(y.sum())} ({y.mean()*100:.1f}%)")

# ── Derived columns ──────────────────────────────────────────────────────────
merged["age_group"] = np.where(merged["AGE"] < 65, "<65", "\u226565")
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
merged["ecg_type"] = "No ECG"
merged.loc[has_ecg & is_normal, "ecg_type"] = "Normal"
merged.loc[has_ecg & ~is_normal, "ecg_type"] = "Abnormal"
merged["af_status"] = merged["any_af_clean"].map({0: "Non-AF", 1: "AF"})

MODELS = [
    ("Model_0", "Model_0_cox_prob", "Model_0_cox_risk", "fold", "Age + Sex"),
    ("Model_1", "Model_1_cox_prob", "Model_1_cox_risk", "fold", "Model 1"),
    ("Model_2", "Model_2_cox_prob", "Model_2_cox_risk", "fold", "Model 2"),
    ("Model_3", "Model_3_cox_prob", "Model_3_cox_risk", "fold", "Model 3"),
    ("Model_4", "Model_4_cox_prob", "Model_4_cox_risk", "fold", "Model 4"),
    ("Model_5", "Model_5_prob", "Model_5_risk", "fold5", "Model 5 (DL-ECG)"),
]

COLORS = {"Model_0":"#8E8E93","Model_1":"#007AFF","Model_2":"#34C759",
          "Model_3":"#FF9500","Model_4":"#AF52DE","Model_5":"#FF3B30"}

# ══════════════════════════════════════════════════════════════════════════════
# Fig 3: ROC + C-index
# ══════════════════════════════════════════════════════════════════════════════
log("Fig 3: ROC + C-index...")
results = []
for mkey, lr_col, cox_col, fold_col, display in MODELS:
    fold_aucs, fold_cs = [], []
    for fid in sorted(merged[fold_col].unique()):
        sub = merged[merged[fold_col] == fid]
        yf = sub[evt_col].values.astype(float)
        v_lr = ~sub[lr_col].isna()
        v_cox = ~sub[cox_col].isna()
        if len(np.unique(yf[v_lr])) > 1:
            fold_aucs.append(roc_auc_score(yf[v_lr], sub.loc[v_lr.values, lr_col].values))
        if v_cox.sum() > 0:
            try: fold_cs.append(concordance_index(np.clip(sub.loc[v_cox.values, time_col].values, 0.5, None),
                                                   -sub.loc[v_cox.values, cox_col].values, yf[v_cox]))
            except: pass
    ma, sa = np.mean(fold_aucs), np.std(fold_aucs, ddof=1)
    mc, sc = np.mean(fold_cs), np.std(fold_cs, ddof=1)
    nf = len(fold_aucs)
    results.append({"key":mkey,"display":display,
                    "mean_auc":ma,"auc_lo":ma-1.96*sa/np.sqrt(nf),"auc_hi":ma+1.96*sa/np.sqrt(nf),
                    "mean_c":mc,"c_lo":mc-1.96*sc/np.sqrt(len(fold_cs)),"c_hi":mc+1.96*sc/np.sqrt(len(fold_cs))})
    log(f"  {display:22s} AUC={ma:.3f}±{sa:.3f}  C={mc:.3f}±{sc:.3f}")

res = pd.DataFrame(results)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
for _, row in res.iterrows():
    lr_col = [m[1] for m in MODELS if m[0]==row["key"]][0]
    valid = ~merged[lr_col].isna()
    fpr, tpr, _ = roc_curve(y[valid], merged.loc[valid, lr_col].values)
    label = f'{row["display"]}  AUC {row["mean_auc"]:.3f} (95% CI {row["auc_lo"]:.3f}\u2013{row["auc_hi"]:.3f})'
    ax1.plot(fpr, tpr, color=COLORS[row["key"]], linewidth=1.3, label=label)
ax1.plot([0,1],[0,1],"k--",alpha=0.3,linewidth=0.8)
ax1.set_xlabel("1 \u2013 Specificity",fontsize=11); ax1.set_ylabel("Sensitivity",fontsize=11)
ax1.set_title("A",fontsize=14,fontweight="bold",loc="left")
ax1.set_xlim(-0.02,1.02); ax1.set_ylim(-0.02,1.02); ax1.set_aspect("equal")
ax1.legend(fontsize=7.5,loc="lower right",frameon=True,edgecolor="#ccc")
ax1.grid(True,alpha=0.15)

x_pos = np.arange(len(res))
for i, (_, row) in enumerate(res.iterrows()):
    ax2.errorbar(x_pos[i], row["mean_c"], yerr=[[row["mean_c"]-row["c_lo"]],[row["c_hi"]-row["mean_c"]]],
                 fmt="o",color=COLORS[row["key"]],markersize=9,capsize=5,capthick=1.8,elinewidth=2,markeredgewidth=0,zorder=5,
                 label=f'{row["display"]}  {row["mean_c"]:.3f} ({row["c_lo"]:.3f}\u2013{row["c_hi"]:.3f})')
ax2.plot(x_pos, res["mean_c"].values, color="#ccc",linewidth=1,zorder=2)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(["Age+\nSex","+ CV risk\nfactors","+ Medical\nhistory","+ Index\nevent","+ ECG\nbiomarkers","+ DL-ECG\n(E2E)"],fontsize=8.5)
ax2.set_ylabel("Cox model C-index",fontsize=11); ax2.set_title("B",fontsize=14,fontweight="bold",loc="left")
ax2.set_xlim(-0.5,len(res)-0.5)
ax2.set_ylim(res["c_lo"].min()-0.03, res["c_hi"].max()+0.03)
ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.02)); ax2.grid(True,axis="y",alpha=0.2)
ax2.legend(fontsize=7.5,loc="lower right",frameon=True,edgecolor="#ccc")
plt.tight_layout(w_pad=3)
fig.savefig(OUT_DIR/"fig3_roc_cindex_CVD6.png",dpi=300,bbox_inches="tight"); plt.close()
log(f"  Saved fig3")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 4: KM tertile
# ══════════════════════════════════════════════════════════════════════════════
log("Fig 4: KM tertile...")
risk = merged["Model_5_risk"].values
r_min, r_max = risk.min(), risk.max()
width = (r_max - r_min) / 3
cuts = [r_min, r_min+width, r_min+2*width, r_max]
merged["rg"] = pd.cut(risk, bins=cuts, labels=["Low","Intermediate","High"], include_lowest=True)

KM_COLORS = {"Low":"#2ecc71","Intermediate":"#3498db","High":"#e74c3c"}
fig, ax = plt.subplots(figsize=(8,6))
for grp in ["Low","Intermediate","High"]:
    sub = merged[merged["rg"]==grp]
    kmf = KaplanMeierFitter()
    kmf.fit(sub[time_col], event_observed=sub[evt_col], label=grp)
    n, ev = len(sub), int(sub[evt_col].sum())
    kmf.plot_survival_function(ax=ax, color=KM_COLORS[grp], linewidth=2, ci_alpha=0.15,
                                label=f"{grp} (n={n}, {ev} events, {ev/n*100:.1f}%)")
lr = multivariate_logrank_test(merged[time_col], merged["rg"], merged[evt_col])
ax.text(0.02,0.03,f"Log-rank p={'<0.001' if lr.p_value<0.001 else f'{lr.p_value:.3f}'}\n\u03c7\u00b2={lr.test_statistic:.1f}",
        transform=ax.transAxes,fontsize=9,fontstyle="italic")
ax.set_xlabel("Time (days)",fontsize=11); ax.set_ylabel("Survival probability",fontsize=11)
ax.set_title("KM by DL-ECG risk tertile (CVD Composite-6)",fontsize=12,fontweight="bold")
ax.set_xlim(0,370); ax.set_ylim(0.75,1.01); ax.grid(True,alpha=0.15)
ax.legend(fontsize=8.5,loc="lower left",frameon=True,edgecolor="#ccc")
plt.tight_layout()
fig.savefig(OUT_DIR/"fig4_km_CVD6.png",dpi=300,bbox_inches="tight"); plt.close()

for grp in ["Low","Intermediate","High"]:
    sub = merged[merged["rg"]==grp]
    log(f"  {grp}: N={len(sub)}, events={int(sub[evt_col].sum())}, rate={sub[evt_col].mean()*100:.1f}%")
log(f"  Log-rank chi2={lr.test_statistic:.1f}, p={lr.p_value:.2e}")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 5: Normal ECG KM
# ══════════════════════════════════════════════════════════════════════════════
log("Fig 5: Normal ECG KM...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, subset, title in [(ax1, "Normal", "Normal ECG"), (ax2, "Abnormal", "Abnormal ECG")]:
    df_sub = merged[merged["ecg_type"] == subset].copy()
    r = df_sub["Model_5_risk"].values
    w = (r.max()-r.min())/3
    c = [r.min(), r.min()+w, r.min()+2*w, r.max()]
    df_sub["rg_sub"] = pd.cut(r, bins=c, labels=["Low","Intermediate","High"], include_lowest=True)

    for grp in ["Low","Intermediate","High"]:
        s = df_sub[df_sub["rg_sub"]==grp]
        kmf = KaplanMeierFitter()
        kmf.fit(s[time_col], event_observed=s[evt_col], label=grp)
        n, ev = len(s), int(s[evt_col].sum())
        kmf.plot_survival_function(ax=ax, color=KM_COLORS[grp], linewidth=2, ci_alpha=0.12,
                                    label=f"{grp} (n={n}, {ev} events, {ev/n*100:.1f}%)")
    lr_s = multivariate_logrank_test(df_sub[time_col], df_sub["rg_sub"], df_sub[evt_col])
    ax.text(0.02,0.03,f"Log-rank p={'<0.001' if lr_s.p_value<0.001 else f'{lr_s.p_value:.3f}'}",
            transform=ax.transAxes,fontsize=9,fontstyle="italic")
    n_t, ev_t = len(df_sub), int(df_sub[evt_col].sum())
    ax.set_title(f"{title} (N={n_t}, {ev_t} events)",fontsize=11,fontweight="bold")
    ax.set_xlabel("Time (days)"); ax.set_ylabel("Survival probability")
    ax.set_xlim(0,370); ax.set_ylim(0.75,1.01); ax.grid(True,alpha=0.15)
    ax.legend(fontsize=8,loc="lower left",frameon=True,edgecolor="#ccc")
    log(f"  {title}: N={n_t}, events={ev_t}, log-rank p={lr_s.p_value:.2e}")

plt.suptitle("DL-ECG risk stratification: Normal vs Abnormal ECG (CVD6)",fontsize=13,fontweight="bold",y=1.02)
plt.tight_layout()
fig.savefig(OUT_DIR/"fig5_km_normal_CVD6.png",dpi=300,bbox_inches="tight"); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# Supp: Cross-tab + NRI/IDI + Calibration + DCA
# ══════════════════════════════════════════════════════════════════════════════
log("Supp: Cross-tab...")
m3_cuts = np.percentile(merged["Model_3_cox_prob"], [33.33, 66.67])
m5_cuts = np.percentile(merged["Model_5_risk"], [33.33, 66.67])
merged["m3g"] = np.where(merged["Model_3_cox_prob"]<=m3_cuts[0],"Low",np.where(merged["Model_3_cox_prob"]<=m3_cuts[1],"Medium","High"))
merged["m5g"] = np.where(merged["Model_5_risk"]<=m5_cuts[0],"Low",np.where(merged["Model_5_risk"]<=m5_cuts[1],"Medium","High"))

# Cross-tab heatmap
groups = ["Low","Medium","High"]
n_mat, ev_mat, rate_mat = np.zeros((3,3),int), np.zeros((3,3),int), np.zeros((3,3))
for i,g3 in enumerate(groups):
    for j,g5 in enumerate(groups):
        mask = (merged["m3g"]==g3)&(merged["m5g"]==g5)
        n_mat[i,j] = mask.sum()
        ev_mat[i,j] = int(merged.loc[mask,evt_col].sum())
        rate_mat[i,j] = ev_mat[i,j]/n_mat[i,j]*100 if n_mat[i,j]>0 else 0

fig, ax = plt.subplots(figsize=(7,6))
cmap = plt.cm.RdYlGn_r
norm = mcolors.Normalize(vmin=0, vmax=max(rate_mat.max(),15))
for i in range(3):
    for j in range(3):
        color = cmap(norm(rate_mat[i,j]))
        ax.add_patch(plt.Rectangle((j,2-i),1,1,facecolor=color,edgecolor="white",linewidth=3))
        br = 0.299*color[0]+0.587*color[1]+0.114*color[2]
        tc = "white" if br<0.55 else "black"
        ax.text(j+0.5,2-i+0.6,f"N={n_mat[i,j]}",ha="center",va="center",fontsize=10,fontweight="bold",color=tc)
        ax.text(j+0.5,2-i+0.38,f"{ev_mat[i,j]} events",ha="center",va="center",fontsize=8,color=tc)
        ax.text(j+0.5,2-i+0.18,f"{rate_mat[i,j]:.1f}%",ha="center",va="center",fontsize=12,fontweight="bold",color=tc)
ax.set_xlim(0,3); ax.set_ylim(-0.2,3.2)
ax.set_xticks([0.5,1.5,2.5]); ax.set_xticklabels(["Low","Medium","High"],fontsize=10,fontweight="bold")
ax.set_yticks([0.5,1.5,2.5]); ax.set_yticklabels(["High","Medium","Low"],fontsize=10,fontweight="bold")
ax.set_xlabel("Model 5 (DL-ECG)",fontsize=11,fontweight="bold")
ax.set_ylabel("Model 3 (Clinical)",fontsize=11,fontweight="bold")
ax.set_title(f"Cross-tabulation (CVD6)\nN={N}, events={int(y.sum())}",fontsize=12,fontweight="bold")
ax.set_aspect("equal"); ax.tick_params(length=0)
sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm); sm.set_array([])
plt.colorbar(sm,ax=ax,shrink=0.5,pad=0.1).set_label("Event rate (%)")
plt.tight_layout()
fig.savefig(OUT_DIR/"supp_crosstab_CVD6.png",dpi=300,bbox_inches="tight"); plt.close()
log("  Saved crosstab")

# NRI/IDI
log("Supp: NRI/IDI...")
gmap = {"Low":0,"Medium":1,"High":2}
m3n = np.array([gmap[g] for g in merged["m3g"]])
m5n = np.array([gmap[g] for g in merged["m5g"]])
em, nm = y==1, y==0
Ne, Nne = int(em.sum()), int(nm.sum())

nri_e = (((m5n>m3n)&em).sum()-((m5n<m3n)&em).sum())/Ne
nri_ne = (((m5n<m3n)&nm).sum()-((m5n>m3n)&nm).sum())/Nne
cat_nri = nri_e + nri_ne

rng = np.random.RandomState(42)
boots = []
for _ in range(1000):
    idx = rng.choice(N,N,replace=True)
    yb,a,b = y[idx],m3n[idx],m5n[idx]
    ne,nne = yb.sum(),len(yb)-yb.sum()
    if ne==0 or nne==0: continue
    boots.append((((b>a)&(yb==1)).sum()-((b<a)&(yb==1)).sum())/ne +
                 (((b<a)&(yb==0)).sum()-((b>a)&(yb==0)).sum())/nne)
nri_lo,nri_hi = np.percentile(boots,[2.5,97.5])
nri_p = (np.array(boots)<=0).mean()

# Platt calibration for IDI
merged["m5_cal"] = np.nan
for fid in sorted(merged["fold5"].unique()):
    tm = merged["fold5"]==fid; trm = ~tm
    pl = LogisticRegression(penalty=None,solver="lbfgs",max_iter=1000)
    pl.fit(merged.loc[trm,"Model_5_risk"].values.reshape(-1,1),y[trm])
    merged.loc[tm,"m5_cal"] = pl.predict_proba(merged.loc[tm,"Model_5_risk"].values.reshape(-1,1))[:,1]

m3p, m5p = merged["Model_3_cox_prob"].values, merged["m5_cal"].values
idi = (m5p[em].mean()-m3p[em].mean())-(m5p[nm].mean()-m3p[nm].mean())
idi_boots = []
for _ in range(1000):
    idx = rng.choice(N,N,replace=True)
    yb=y[idx];a=m3p[idx];b=m5p[idx]
    e=yb==1;n=yb==0
    if e.sum()==0 or n.sum()==0: continue
    idi_boots.append((b[e].mean()-a[e].mean())-(b[n].mean()-a[n].mean()))
idi_lo,idi_hi = np.percentile(idi_boots,[2.5,97.5])
idi_p = (np.array(idi_boots)<=0).mean()

p_nri = "<0.001" if nri_p<0.001 else f"{nri_p:.3f}"
p_idi = "<0.001" if idi_p<0.001 else f"{idi_p:.3f}"
log(f"  Cat NRI: {cat_nri:+.3f} ({nri_lo:+.3f},{nri_hi:+.3f}) P={p_nri}")
log(f"  IDI: {idi:+.4f} ({idi_lo:+.4f},{idi_hi:+.4f}) P={p_idi}")

nri_df = pd.DataFrame([
    {"Metric":"Category NRI","Value":f"{cat_nri:+.3f}","95% CI":f"({nri_lo:+.3f},{nri_hi:+.3f})","P":p_nri},
    {"Metric":"IDI","Value":f"{idi:+.4f}","95% CI":f"({idi_lo:+.4f},{idi_hi:+.4f})","P":p_idi},
])
nri_df.to_csv(OUT_DIR/"nri_idi_CVD6.csv",index=False)

# Calibration + DCA
log("Supp: Calibration + DCA...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
for mname, prob_col, color in [("Model 3","Model_3_cox_prob","#FF9500"),
                                 ("Model 4","Model_4_cox_prob","#AF52DE"),
                                 ("Model 5","m5_cal","#FF3B30")]:
    prob = merged[prob_col].values; valid = ~np.isnan(prob)
    frac, mean_p = calibration_curve(y[valid],prob[valid],n_bins=10,strategy="quantile")
    ax1.plot(mean_p,frac,"o-",color=color,label=mname,markersize=6,linewidth=1.5)
ax1.plot([0,0.25],[0,0.25],"k--",alpha=0.4,linewidth=1,label="Perfect")
ax1.set_xlabel("Predicted probability"); ax1.set_ylabel("Observed event rate")
ax1.set_title("A  Calibration",fontsize=12,fontweight="bold",loc="left")
ax1.legend(fontsize=9); ax1.grid(True,alpha=0.15)
ax1.set_xlim(0,0.25); ax1.set_ylim(0,0.25); ax1.set_aspect("equal")

thresholds = np.arange(0.005,0.25,0.002)
prev = y.mean()
ax2.plot(thresholds*100,prev-(1-prev)*thresholds/(1-thresholds),"k-",linewidth=1,alpha=0.5,label="Treat all")
ax2.axhline(0,color="grey",linewidth=0.8,alpha=0.4,label="Treat none")
for mname,prob_col,color in [("Model 3","Model_3_cox_prob","#FF9500"),
                               ("Model 4","Model_4_cox_prob","#AF52DE"),
                               ("Model 5","m5_cal","#FF3B30")]:
    prob = merged[prob_col].values; valid = ~np.isnan(prob)
    yv,pv = y[valid],prob[valid]; nv = len(yv)
    nb = [((pv>=th)&(yv==1)).sum()/nv - ((pv>=th)&(yv==0)).sum()/nv*th/(1-th) for th in thresholds]
    ax2.plot(thresholds*100,nb,color=color,linewidth=2,label=mname)
ax2.set_xlabel("Threshold probability (%)"); ax2.set_ylabel("Net benefit")
ax2.set_title("B  Decision Curve Analysis",fontsize=12,fontweight="bold",loc="left")
ax2.legend(fontsize=9,loc="upper right"); ax2.grid(True,alpha=0.15)
ax2.set_xlim(0.5,25)
plt.tight_layout(w_pad=3)
fig.savefig(OUT_DIR/"supp_calibration_dca_CVD6.png",dpi=300,bbox_inches="tight"); plt.close()
log("  Saved calibration + DCA")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 6: Subgroup analysis
# ══════════════════════════════════════════════════════════════════════════════
log("Fig 6: Subgroup analysis...")
from matplotlib.lines import Line2D

MODEL_DEFS_SUB = [
    ("Model 3", "Model_3_cox_risk", "fold"),
    ("Model 4", "Model_4_cox_risk", "fold"),
    ("Model 5", "Model_5_risk", "fold5"),
]
MODEL_COLORS_SUB = {"Model 3":"#FF9500","Model 4":"#AF52DE","Model 5":"#FF3B30"}
MODEL_MARKERS_SUB = {"Model 3":"s","Model 4":"D","Model 5":"o"}

asia = ["Singapore","China","Malaysia","Philippines","Indonesia","UAE"]
europe = ["Poland","Portugal"]; latam = ["Colombia","Argentina"]; africa = ["Egypt","Nigeria","Tanzania","Kenia"]
merged["region"] = merged["COUNTRY"].apply(
    lambda x: "Asia" if x in asia else "Europe" if x in europe
    else "Latin America" if x in latam else "Africa" if x in africa else "Other")

def bootstrap_pooled_c_sub(data, risk_col, n_boot=500, seed=42):
    """Pooled C-index with bootstrap 95% CI."""
    rng_b = np.random.RandomState(seed)
    tv = data[time_col].values.astype(float)
    yv = data[evt_col].values.astype(float)
    rv = data[risk_col].values
    v = ~np.isnan(rv)
    tv, yv, rv = tv[v], yv[v], rv[v]
    if len(np.unique(yv)) < 2: return np.nan, np.nan, np.nan
    c = concordance_index(np.clip(tv,0.5,None),-rv,yv)
    bs = []
    n = len(tv)
    for _ in range(n_boot):
        idx = rng_b.choice(n,n,replace=True)
        if len(np.unique(yv[idx]))<2: continue
        try: bs.append(concordance_index(np.clip(tv[idx],0.5,None),-rv[idx],yv[idx]))
        except: continue
    lo, hi = (np.percentile(bs,[2.5,97.5]) if bs else (np.nan, np.nan))
    return c, lo, hi

def plot_multi_model_panel_sub(ax, col, values, title):
    offsets = {"Model 3":-0.22,"Model 4":0.0,"Model 5":0.22}
    for vi, val in enumerate(values):
        sub = merged[merged[col]==val]
        if sub[evt_col].sum()<5: continue
        for mname, cox_col_m, fold_col in MODEL_DEFS_SUB:
            c,lo,hi = bootstrap_pooled_c_sub(sub, cox_col_m)
            if np.isnan(c): continue
            ax.errorbar(vi+offsets[mname],c,yerr=[[max(c-lo,0)],[max(hi-c,0)]],
                        fmt=MODEL_MARKERS_SUB[mname],color=MODEL_COLORS_SUB[mname],
                        markersize=8,capsize=4,capthick=1.3,elinewidth=1.8,markeredgewidth=0,zorder=5,alpha=0.9)
    ax.set_xticks(range(len(values)))
    xlabels = [f"{v}\n(n={len(merged[merged[col]==v])})" for v in values]
    ax.set_xticklabels(xlabels,fontsize=8); ax.axhline(0.5,color="grey",linestyle="--",alpha=0.3)
    ax.set_ylim(0.35,0.90); ax.set_ylabel("C-index",fontsize=10); ax.set_title(title,fontsize=10,fontweight="bold")
    ax.grid(True,axis="y",alpha=0.15)
    legend_elements = [Line2D([0],[0],marker="s",color="w",markerfacecolor="#FF9500",markersize=7,label="Model 3"),
                       Line2D([0],[0],marker="D",color="w",markerfacecolor="#AF52DE",markersize=7,label="Model 4"),
                       Line2D([0],[0],marker="o",color="w",markerfacecolor="#FF3B30",markersize=7,label="Model 5")]
    ax.legend(handles=legend_elements,loc="lower right",fontsize=7,frameon=True,edgecolor="#ccc")

fig_sub = plt.figure(figsize=(16,10))
gs = fig_sub.add_gridspec(2,6,hspace=0.4,wspace=0.4)
ax_sex = fig_sub.add_subplot(gs[0,0:2])
ax_age = fig_sub.add_subplot(gs[0,2:4])
ax_ecg = fig_sub.add_subplot(gs[0,4:6])
ax_country = fig_sub.add_subplot(gs[1,0:4])
ax_af = fig_sub.add_subplot(gs[1,4:6])

plot_multi_model_panel_sub(ax_sex,"RQSEX",["Male","Female"],"A  Sex")
plot_multi_model_panel_sub(ax_age,"age_group",["<65","\u226565"],"B  Age")
plot_multi_model_panel_sub(ax_ecg,"ecg_type",["Normal","Abnormal"],"C  ECG status")
plot_multi_model_panel_sub(ax_af,"af_status",["Non-AF","AF"],"E  AF status")

# Country panel: Model 5 only, pooled C with bootstrap
COUNTRY_COLORS = {"Singapore":"#e6194B","China":"#3cb44b","Malaysia":"#4363d8","Philippines":"#f58231",
                  "Colombia":"#911eb4","Indonesia":"#42d4f4","UAE":"#f032e6","Poland":"#bfef45",
                  "Egypt":"#fabed4","Portugal":"#469990","Nigeria":"#dcbeff","Tanzania":"#9A6324",
                  "Kenia":"#800000","Argentina":"#aaffc3"}
countries = sorted(merged["COUNTRY"].dropna().unique())
valid_countries = [c for c in countries if merged.loc[merged["COUNTRY"]==c,evt_col].sum()>=4]
country_cs = []
for c in valid_countries:
    sub = merged[merged["COUNTRY"]==c]
    tv,rv,yv = sub[time_col].values,sub["Model_5_risk"].values,sub[evt_col].values
    v = ~np.isnan(rv)
    try: mc = concordance_index(np.clip(tv[v],0.5,None),-rv[v],yv[v])
    except: mc = np.nan
    rng2 = np.random.RandomState(42); bs = []
    for _ in range(500):
        idx = rng2.choice(v.sum(),v.sum(),replace=True)
        try: bs.append(concordance_index(np.clip(tv[v][idx],0.5,None),-rv[v][idx],yv[v][idx]))
        except: continue
    lo,hi = (np.percentile(bs,[2.5,97.5]) if bs else (np.nan,np.nan))
    country_cs.append((c,mc,lo,hi,len(sub),int(sub[evt_col].sum())))
country_cs.sort(key=lambda x: x[1] if not np.isnan(x[1]) else 0, reverse=True)

for i,(c,mc,lo,hi,n,ev) in enumerate(country_cs):
    if np.isnan(mc): continue
    ax_country.plot(i,mc,marker="o",color=COUNTRY_COLORS.get(c,"#333"),markersize=10,
                    markeredgewidth=0.5,markeredgecolor="white",zorder=5,alpha=0.9,linestyle="none",
                    label=f"{c} (n={n},{ev}ev) C={mc:.2f} ({lo:.2f}\u2013{hi:.2f})")
ax_country.set_xticks(range(len(country_cs)))
ax_country.set_xticklabels([x[0] for x in country_cs],fontsize=7,rotation=45,ha="right")
ax_country.axhline(0.5,color="grey",linestyle="--",alpha=0.3)
ax_country.set_ylim(0.35,0.95); ax_country.set_ylabel("C-index",fontsize=10)
ax_country.set_title("D  Country (Model 5)",fontsize=10,fontweight="bold")
ax_country.grid(True,axis="y",alpha=0.15)
ax_country.legend(fontsize=5.5,loc="lower left",frameon=True,edgecolor="#ccc",ncol=2)

fig_sub.savefig(OUT_DIR/"fig6_subgroup_CVD6.png",dpi=300,bbox_inches="tight"); plt.close()
log("  Saved fig6")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 7: ECG biomarker forest plot (HR)
# ══════════════════════════════════════════════════════════════════════════════
log("Fig 7: ECG biomarker forest plot...")
from lifelines import CoxPHFitter

ECG_FEATURES = [
    ("Atrial Fibrillation","Atrial fibrillation"),("LBBB","LBBB"),("RBBB","RBBB"),
    ("Q Wave","Q wave"),("ST Elevation","ST elevation"),("ST Depression","ST depression"),
    ("T Wave Inversion","T wave inversion"),("Ischaemic","Ischaemic changes"),
    ("QT Prolongation","QT prolongation"),("LVH","LVH"),("1 AV Block","1st AV block"),
    ("Left Axis Deviation","Left axis dev"),("Right Axis Deviation","Right axis dev"),
    ("MI (Old)","Old MI"),("MI(Acute)","Acute MI"),
    ("PR Interval (ms)","PR interval (per SD)"),("QRS Duration (ms)","QRS duration (per SD)"),
    ("QT Interval (ms)","QT interval (per SD)"),
]

ecg_res = []
for col, display in ECG_FEATURES:
    if col not in merged.columns: continue
    df_cox = merged[["STUDYID",evt_col,time_col,col]].copy()
    if df_cox[col].dtype == object:
        df_cox[col] = df_cox[col].map({"Yes":1,"No":0})
    else:
        df_cox[col] = pd.to_numeric(df_cox[col], errors="coerce")
    df_cox = df_cox.dropna()
    df_cox["_t"] = np.clip(df_cox[time_col],0.5,None); df_cox["_e"] = df_cox[evt_col]
    if len(df_cox)<50 or df_cox["_e"].sum()<5: continue
    is_cont = col in ["PR Interval (ms)","QRS Duration (ms)","QT Interval (ms)"]
    if is_cont: df_cox[col] = (df_cox[col]-df_cox[col].mean())/df_cox[col].std()
    try:
        cph = CoxPHFitter(); cph.fit(df_cox[[col,"_t","_e"]],duration_col="_t",event_col="_e")
        hr = cph.hazard_ratios_.values[0]; ci = cph.confidence_intervals_.values[0]; p = cph.summary["p"].values[0]
        ecg_res.append({"display":display,"hr":hr,"hr_lo":np.exp(ci[0]),"hr_hi":np.exp(ci[1]),"p":p})
    except: pass

ecg_df = pd.DataFrame(ecg_res).sort_values("hr",ascending=True).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(8,7))
for i,(_, row) in enumerate(ecg_df.iterrows()):
    color = "#e74c3c" if row["hr"]>1 and row["p"]<0.05 else "#3498db" if row["hr"]<1 and row["p"]<0.05 else "#999"
    ax.errorbar(row["hr"],i,xerr=[[row["hr"]-row["hr_lo"]],[row["hr_hi"]-row["hr"]]],
                fmt="o",color=color,markersize=7,capsize=4,capthick=1.3,elinewidth=1.5,markeredgewidth=0,zorder=5)
    p_str = "***" if row["p"]<0.001 else "**" if row["p"]<0.01 else "*" if row["p"]<0.05 else ""
    if p_str: ax.text(row["hr_hi"]+0.05,i,p_str,fontsize=9,va="center",color=color,fontweight="bold")
ax.axvline(1.0,color="grey",linestyle="--",linewidth=1,alpha=0.5)
ax.set_yticks(range(len(ecg_df))); ax.set_yticklabels(ecg_df["display"],fontsize=9)
ax.set_xlabel("Hazard Ratio (95% CI)",fontsize=11)
ax.set_title("ECG biomarkers — univariate HR (CVD6)",fontsize=12,fontweight="bold")
ax.grid(True,axis="x",alpha=0.15); ax.tick_params(axis="y",length=0)
plt.tight_layout()
fig.savefig(OUT_DIR/"fig7_ecg_forest_CVD6.png",dpi=300,bbox_inches="tight"); plt.close()
log("  Saved fig7")

log(f"\n{'='*60}")
log(f"All CVD6 sensitivity analyses saved to {OUT_DIR}")
log(f"{'='*60}")

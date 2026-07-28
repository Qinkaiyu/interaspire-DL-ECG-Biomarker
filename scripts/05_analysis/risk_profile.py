"""
Risk group profiles (ECG + clinical portrait):
  1. Composite endpoint: DL-ECG Low/Intermediate/High group characteristics
  2. Component endpoints: ECG profile of patients with each event type
  3. Heatmap visualization
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

# Derive risk groups and component events
rmin, rmax = merged["risk_norm"].min(), merged["risk_norm"].max()
t1, t2 = rmin + (rmax - rmin) / 3, rmin + 2 * (rmax - rmin) / 3
merged["risk_group"] = pd.cut(merged["risk_norm"],
    bins=[rmin - 0.001, t1, t2, rmax + 0.001],
    labels=["Low", "Intermediate", "High"])

first_comp = merged["first_component_Composite_4"].fillna("")
merged["comp_cv_death"] = ((first_comp == "SURVIVAL") & (merged["cvd_death_flag"] == 1)).astype(int)
merged["comp_mi"] = (first_comp == "HOSPAMI").astype(int)
merged["comp_hf"] = (first_comp == "HOSPHF").astype(int)
merged["comp_stroke"] = (first_comp == "HOSPSTROKE").astype(int)

# ── Feature definitions ─────────────────────────────────────────────────────
ECG_BINARY = [
    ("Sinus Rhythm", "Sinus rhythm"),
    ("Atrial Fibrillation", "AF"),
    ("LBBB", "LBBB"),
    ("RBBB", "RBBB"),
    ("Q Wave", "Q wave"),
    ("ST Elevation", "ST elevation"),
    ("ST Depression", "ST depression"),
    ("T Wave Inversion", "T wave inversion"),
    ("Ischaemic", "Ischaemic changes"),
    ("QT Prolongation", "QT prolongation"),
    ("LVH", "LVH"),
    ("1 AV Block", "1st AV block"),
    ("Left Axis Deviation", "Left axis deviation"),
    ("MI (Old)", "Old MI"),
    ("MI(Acute)", "Acute MI"),
]

ECG_CONTINUOUS = [
    ("PR Interval (ms)", "PR interval"),
    ("QRS Duration (ms)", "QRS duration"),
    ("QT Interval (ms)", "QT interval"),
]

CLINICAL = [
    ("AGE", "Age", "cont"),
    ("RQDCBMI", "BMI", "cont"),
    ("RQDCSYSBP", "Systolic BP", "cont"),
    ("CVAREGFR", "eGFR", "cont"),
    ("CVARLABHBA1CPERCENT", "HbA1c", "cont"),
    ("CVARLABLDLFRIEDEWALD", "LDL-C", "cont"),
    ("CVARLABHDL", "HDL-C", "cont"),
    ("RQHISTOFHEARTFAIL", "HF history", "cat"),
    ("RQHISTOFCORARTDIS", "CAD history", "cat"),
    ("RQHISTOFDIAB", "DM history", "cat"),
    ("RQHISTOFHYPER", "HTN history", "cat"),
    ("any_af", "AF history", "cat_num"),
    ("RQHISTOFCOPD", "COPD history", "cat"),
    ("CVARMEDGLUCOSELOWERING", "Glucose-lowering", "cat"),
    ("RQANTICOAGULANTS", "Anticoagulant", "cat"),
]


def get_prevalence(data, col):
    """Get prevalence of Yes/1 for a binary column."""
    if data[col].dtype == object:
        return (data[col] == "Yes").mean() * 100
    else:
        return data[col].mean() * 100


# ═══════════════════════════════════════════════════════════════════════════
# PART 1: Composite endpoint — Risk group heatmap
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Part 1: Composite endpoint risk group profiles ===")

# ECG binary features prevalence by risk group
ecg_data = {}
for grp in ["Low", "Intermediate", "High"]:
    gs = merged[merged["risk_group"] == grp]
    ecg_data[grp] = {}
    for col, label in ECG_BINARY:
        if col in merged.columns:
            ecg_data[grp][label] = get_prevalence(gs, col)

ecg_df = pd.DataFrame(ecg_data)
# Remove Sinus rhythm (inverse meaning)
if "Sinus rhythm" in ecg_df.index:
    ecg_df.loc["Sinus rhythm"] = 100 - ecg_df.loc["Sinus rhythm"]  # show non-sinus rate
    ecg_df = ecg_df.rename(index={"Sinus rhythm": "Non-sinus rhythm"})

# Sort by High group prevalence
ecg_df = ecg_df.sort_values("High", ascending=True)

# Heatmap
fig, ax = plt.subplots(figsize=(8, 8))
data_matrix = ecg_df.values
im = ax.imshow(data_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=40)

ax.set_xticks(range(3))
ax.set_xticklabels(["Low\n(n=2435)", "Intermediate\n(n=1288)", "High\n(n=65)"], fontsize=10)
ax.set_yticks(range(len(ecg_df)))
ax.set_yticklabels(ecg_df.index, fontsize=9)

# Annotate cells
for i in range(len(ecg_df)):
    for j in range(3):
        val = data_matrix[i, j]
        color = "white" if val > 20 else "black"
        ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=8, color=color)

ax.set_title("ECG abnormality prevalence (%) by DL-ECG risk tertile\n(CVD_Composite_4)",
             fontsize=12, fontweight="bold")
plt.colorbar(im, ax=ax, label="Prevalence (%)", shrink=0.8)
plt.tight_layout()
fig.savefig(OUT_DIR / "profile_ecg_heatmap_risk_groups.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT_DIR / "profile_ecg_heatmap_risk_groups.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved: profile_ecg_heatmap_risk_groups.png")

# Clinical features by risk group
print("\nClinical features by risk group:")
clin_rows = []
for col, label, ctype in CLINICAL:
    if col not in merged.columns:
        continue
    row = {"Feature": label}
    for grp in ["Low", "Intermediate", "High"]:
        gs = merged[merged["risk_group"] == grp]
        if ctype == "cont":
            vals = pd.to_numeric(gs[col], errors="coerce").dropna()
            row[grp] = f"{vals.mean():.1f} ± {vals.std():.1f}"
        elif ctype == "cat":
            row[grp] = f"{get_prevalence(gs, col):.1f}%"
        elif ctype == "cat_num":
            row[grp] = f"{gs[col].mean() * 100:.1f}%"
    clin_rows.append(row)

clin_df = pd.DataFrame(clin_rows)
clin_df.to_csv(OUT_DIR / "profile_clinical_risk_groups.csv", index=False)
print(clin_df.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
# PART 2: Component endpoints — ECG profile of each event type
# ═══════════════════════════════════════════════════════════════════════════
print("\n\n=== Part 2: ECG profile by event type ===")

COMPONENTS = {
    "CV death": "comp_cv_death",
    "MI": "comp_mi",
    "HF": "comp_hf",
    "Stroke": "comp_stroke",
}

# ECG prevalence for each event type vs no event
event_ecg = {"No event": {}}
no_evt = merged[merged["event_CVD_Composite_4"] == 0]
for col, label in ECG_BINARY:
    if col in merged.columns:
        event_ecg["No event"][label] = get_prevalence(no_evt, col)

for comp_name, evt_col in COMPONENTS.items():
    event_ecg[comp_name] = {}
    evt_pts = merged[merged[evt_col] == 1]
    for col, label in ECG_BINARY:
        if col in merged.columns:
            event_ecg[comp_name][label] = get_prevalence(evt_pts, col)

event_df = pd.DataFrame(event_ecg)
# Convert sinus rhythm
if "Sinus rhythm" in event_df.index:
    event_df.loc["Sinus rhythm"] = 100 - event_df.loc["Sinus rhythm"]
    event_df = event_df.rename(index={"Sinus rhythm": "Non-sinus rhythm"})

# Sort by max prevalence across event types
event_df["max_evt"] = event_df[list(COMPONENTS.keys())].max(axis=1)
event_df = event_df.sort_values("max_evt", ascending=True).drop(columns="max_evt")

# Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
cols_order = ["No event", "CV death", "MI", "HF", "Stroke"]
data_matrix = event_df[cols_order].values
im = ax.imshow(data_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=50)

n_counts = {
    "No event": len(no_evt),
    "CV death": int(merged["comp_cv_death"].sum()),
    "MI": int(merged["comp_mi"].sum()),
    "HF": int(merged["comp_hf"].sum()),
    "Stroke": int(merged["comp_stroke"].sum()),
}
x_labels = [f"{c}\n(n={n_counts[c]})" for c in cols_order]
ax.set_xticks(range(len(cols_order)))
ax.set_xticklabels(x_labels, fontsize=10)
ax.set_yticks(range(len(event_df)))
ax.set_yticklabels(event_df.index, fontsize=9)

for i in range(len(event_df)):
    for j in range(len(cols_order)):
        val = data_matrix[i, j]
        color = "white" if val > 25 else "black"
        ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=8, color=color)

ax.set_title("ECG abnormality prevalence (%) by first cardiovascular event type\n"
             "(CVD_Composite_4 components)",
             fontsize=12, fontweight="bold")
plt.colorbar(im, ax=ax, label="Prevalence (%)", shrink=0.8)
plt.tight_layout()
fig.savefig(OUT_DIR / "profile_ecg_heatmap_event_types.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT_DIR / "profile_ecg_heatmap_event_types.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved: profile_ecg_heatmap_event_types.png")

# Clinical features by event type
print("\nClinical features by event type:")
clin_evt_rows = []
for col, label, ctype in CLINICAL:
    if col not in merged.columns:
        continue
    row = {"Feature": label}
    for comp_name in ["No event"] + list(COMPONENTS.keys()):
        if comp_name == "No event":
            gs = merged[merged["event_CVD_Composite_4"] == 0]
        else:
            gs = merged[merged[COMPONENTS[comp_name]] == 1]
        if ctype == "cont":
            vals = pd.to_numeric(gs[col], errors="coerce").dropna()
            row[comp_name] = f"{vals.mean():.1f}"
        elif ctype == "cat":
            row[comp_name] = f"{get_prevalence(gs, col):.1f}%"
        elif ctype == "cat_num":
            row[comp_name] = f"{gs[col].mean() * 100:.1f}%"
    clin_evt_rows.append(row)

clin_evt_df = pd.DataFrame(clin_evt_rows)
clin_evt_df.to_csv(OUT_DIR / "profile_clinical_event_types.csv", index=False)
print(clin_evt_df.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
# PART 3: Combined radar chart — ECG profile per event type
# ═══════════════════════════════════════════════════════════════════════════
print("\n\n=== Part 3: Radar chart ===")

# Select most discriminative ECG features
radar_features = ["AF", "LBBB", "ST depression", "T wave inversion",
                  "QT prolongation", "Ischaemic changes", "LVH", "Old MI",
                  "Q wave", "ST elevation"]
radar_cols = [col for col, label in ECG_BINARY if label in radar_features]
radar_labels = [label for col, label in ECG_BINARY if label in radar_features]

# Get prevalence for each event type
n_features = len(radar_labels)
angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
angles += angles[:1]  # close the polygon

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection="polar"))

event_colors = {
    "No event": "#95a5a6",
    "CV death": "#e74c3c",
    "MI": "#e67e22",
    "HF": "#3498db",
    "Stroke": "#2ecc71",
}

for comp_name in ["No event"] + list(COMPONENTS.keys()):
    if comp_name == "No event":
        gs = merged[merged["event_CVD_Composite_4"] == 0]
    else:
        gs = merged[merged[COMPONENTS[comp_name]] == 1]

    values = []
    for col, label in ECG_BINARY:
        if label in radar_features:
            values.append(get_prevalence(gs, col))
    values += values[:1]

    lw = 1.0 if comp_name == "No event" else 2.0
    ls = "--" if comp_name == "No event" else "-"
    n = len(gs)
    ax.plot(angles, values, color=event_colors[comp_name], linewidth=lw,
            linestyle=ls, label=f"{comp_name} (n={n})")
    if comp_name != "No event":
        ax.fill(angles, values, color=event_colors[comp_name], alpha=0.05)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=9)
ax.set_ylim(0, 45)
ax.set_title("ECG abnormality prevalence by event type\n(radar chart)", fontsize=13,
             fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
plt.tight_layout()
fig.savefig(OUT_DIR / "profile_ecg_radar_event_types.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT_DIR / "profile_ecg_radar_event_types.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved: profile_ecg_radar_event_types.png")

print("\nDone. All profile analyses saved.")

#!/usr/bin/env python3
"""Schoenfeld residual proportional-hazards diagnostics for Models 0-4."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


BASE = Path(".")
DATA_PATH = BASE / "data" / "INTERASPIRE_analysis_dataset.xlsx"
OUT_DIR = BASE / "results" / "schoenfeld_ph"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENDPOINTS = {
    "CVD4_primary": {
        "file": BASE / "fold_splits_cvd_mace4_1yr.csv",
        "time": "time_Composite_4",
        "event": "event_CVD_Composite_4",
    },
    "CVD6_sensitivity": {
        "file": BASE / "fold_splits_cvd_mace6_1yr.csv",
        "time": "time_Composite_6",
        "event": "event_CVD_Composite_6",
    },
}

M0_VARS = ["CVARAGE", "RQSEX"]
M1_VARS = [
    "CVARAGE",
    "RQSEX",
    "CVARCOSMOKING",
    "CVARBMI",
    "CVARSYSTOL",
    "CVARLABHBA1CPERCENT",
    "CVAREGFR",
    "CVARLABHDL",
    "CVARLABLDLFRIEDEWALD",
]
M2_VARS = M1_VARS + [
    "RQHISTOFPRECVD",
    "RQHISTOFCORARTDIS",
    "RQHISTOFHEARTFAIL",
    "any_af",
    "CVARMEDBPLOWERING",
    "CVARMEDLIPIDLOWERING",
    "CVARMEDGLUCOSELOWERING",
    "RQANTICOAGULANTS",
    "RQHISTOFCOPD",
]
M3_VARS = M2_VARS + ["idx_STEMI", "idx_NSTEMI", "idx_UA", "idx_ePCI"]
ECG_M4 = ["QT Prolongation", "ST Depression", "QT Interval (ms)"]
M4_VARS = M3_VARS + ECG_M4

MODELS = {
    "Model_0": M0_VARS,
    "Model_1": M1_VARS,
    "Model_2": M2_VARS,
    "Model_3": M3_VARS,
    "Model_4": M4_VARS,
}

ECG_BIN = [
    "Atrial Fibrillation",
    "LBBB",
    "RBBB",
    "Q Wave",
    "ST Elevation",
    "ST Depression",
    "T Wave Inversion",
    "Ischaemic",
    "QT Prolongation",
    "LVH",
    "1 AV Block",
    "Left Axis Deviation",
    "Right Axis Deviation",
    "MI (Old)",
    "MI(Acute)",
]

BIN_EDGES = {
    "PR Interval (ms)": [0, 120, 200, 2000],
    "QRS Duration (ms)": [0, 120, 1000],
    "QT Interval (ms)": [0, 440, 480, 5000],
    "FHPRINTER": [0, 120, 200, 2000],
    "FHQRSDURA": [0, 120, 1000],
    "FHQTC": [0, 440, 480, 5000],
    "CVARSYSTOL": [0, 120, 130, 140, 180, 250],
    "CVARLABHBA1CPERCENT": [0, 5.7, 6.5, 20],
    "CVARLABLDLFRIEDEWALD": [0, 1.8, 2.6, 3.4, 4.9, 10],
    "CVARLABHDL": [0, 1.0, 1.3, 1.55, 5],
    "CVAREGFR": [0, 15, 30, 45, 60, 90, 200],
    "CVARBMI": [0, 18.5, 25, 30, 35, 100],
    "CVARAGE": [0, 50, 60, 70, 80, 150],
}


def encode_yes_no(series: pd.Series) -> pd.Series:
    return series.map({"Yes": 1, "No": 0, "yes": 1, "no": 0})


def bin_series(series: pd.Series, edges: list[float]) -> pd.Series:
    return pd.cut(series, bins=edges, labels=range(len(edges) - 1), include_lowest=True).astype(float)


def prepare_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, engine="openpyxl")
    df["RQSEX"] = df["RQSEX"].map({"Male": 1, "Female": 0})

    yes_no_vars = [
        "CVARCOSMOKING",
        "RQHISTOFPRECVD",
        "RQHISTOFCORARTDIS",
        "RQHISTOFHEARTFAIL",
        "CVARMEDBPLOWERING",
        "CVARMEDLIPIDLOWERING",
        "CVARMEDGLUCOSELOWERING",
        "RQANTICOAGULANTS",
        "RQHISTOFCOPD",
    ]
    for var in yes_no_vars:
        df[var] = encode_yes_no(df[var])

    for var in ECG_BIN:
        if var in df.columns:
            df[var] = encode_yes_no(df[var])

    df["idx_STEMI"] = (df["RQINDEX"] == "Acute myocardial infarction STEMI").astype(float)
    df["idx_NSTEMI"] = (df["RQINDEX"] == "Acute myocardial infarction Non-STEMI").astype(float)
    df["idx_UA"] = (df["RQINDEX"] == "Unstable angina / Acute myocardial ischaemia").astype(float)
    df["idx_ePCI"] = (
        df["RQINDEX"] == "Elective percutaneous transluminal coronary angioplasty"
    ).astype(float)

    for ai_col, fh_col in [
        ("PR Interval (ms)", "FHPRINTER"),
        ("QRS Duration (ms)", "FHQRSDURA"),
        ("QT Interval (ms)", "FHQTC"),
    ]:
        mask = df[ai_col].isna() & df[fh_col].notna()
        df.loc[mask, ai_col] = df.loc[mask, fh_col]

    mask = df["RBBB"].isna()
    df.loc[mask & (df["FHBUNBRBLO"] == "Right bundle branch block"), "RBBB"] = 1
    df.loc[
        mask & df["FHBUNBRBLO"].notna() & (df["FHBUNBRBLO"] != "Right bundle branch block"),
        "RBBB",
    ] = 0

    mask = df["LBBB"].isna()
    df.loc[mask & (df["FHBUNBRBLO"] == "Left bundle branch block"), "LBBB"] = 1
    df.loc[
        mask & df["FHBUNBRBLO"].notna() & (df["FHBUNBRBLO"] != "Left bundle branch block"),
        "LBBB",
    ] = 0

    mask = df["1 AV Block"].isna()
    df.loc[mask & (df["FHAVNOBL"] == "I°"), "1 AV Block"] = 1
    df.loc[mask & df["FHAVNOBL"].notna() & (df["FHAVNOBL"] != "I°"), "1 AV Block"] = 0

    mask = df["LVH"].isna()
    fh_lvh = encode_yes_no(df["FHLEFTVENHYP"])
    df.loc[mask & fh_lvh.notna(), "LVH"] = fh_lvh[mask & fh_lvh.notna()]

    for col, edges in BIN_EDGES.items():
        if col in df.columns:
            df[col] = bin_series(df[col], edges)

    return df


def fdr_bh(p_values: pd.Series) -> pd.Series:
    p = p_values.astype(float).to_numpy()
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        value = min(prev, ranked[i] * n / (i + 1))
        q[order[i]] = value
        prev = value
    return pd.Series(q, index=p_values.index)


def fit_and_test(data: pd.DataFrame, variables: list[str], time_col: str, event_col: str):
    available = [var for var in variables if var in data.columns]
    x = data[available].copy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(imputer.fit_transform(x))

    fit_df = pd.DataFrame(x_scaled, columns=available, index=data.index)
    fit_df["_t"] = np.clip(data[time_col].astype(float).to_numpy(), 0.5, None)
    fit_df["_e"] = data[event_col].astype(float).to_numpy()

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(fit_df, duration_col="_t", event_col="_e", show_progress=False)

    result = proportional_hazard_test(cph, fit_df, time_transform="rank")
    summary = result.summary.reset_index().rename(columns={"index": "variable"})
    summary["q_value_bh"] = fdr_bh(summary["p"])
    summary = summary.rename(columns={"test_statistic": "chisq"})

    residuals = cph.compute_residuals(fit_df, kind="scaled_schoenfeld")
    return summary, residuals


def main() -> None:
    df = prepare_data()
    all_rows = []
    summary_lines = []

    for endpoint_name, endpoint_cfg in ENDPOINTS.items():
        split_df = pd.read_csv(endpoint_cfg["file"])
        time_col = endpoint_cfg["time"]
        event_col = endpoint_cfg["event"]
        merged = df.merge(
            split_df[["STUDYID", "fold", event_col, time_col]],
            on="STUDYID",
            how="inner",
            suffixes=("_orig", ""),
        )
        n_events = int(merged[event_col].sum())
        summary_lines.append(f"{endpoint_name}: N={len(merged)}, events={n_events}")

        for model_name, variables in MODELS.items():
            ph_summary, residuals = fit_and_test(merged, variables, time_col, event_col)
            ph_summary.insert(0, "n_events", n_events)
            ph_summary.insert(0, "n", len(merged))
            ph_summary.insert(0, "model", model_name)
            ph_summary.insert(0, "endpoint", endpoint_name)
            all_rows.append(ph_summary)

            residual_path = OUT_DIR / f"scaled_schoenfeld_{endpoint_name}_{model_name}.csv"
            residuals.to_csv(residual_path, index=True)

            flagged = ph_summary.loc[ph_summary["p"] < 0.05, ["variable", "chisq", "p", "q_value_bh"]]
            if flagged.empty:
                summary_lines.append(f"  {model_name}: no raw p<0.05 PH violations")
            else:
                items = [
                    f"{row.variable} p={row.p:.4g}, q={row.q_value_bh:.4g}"
                    for row in flagged.itertuples(index=False)
                ]
                summary_lines.append(f"  {model_name}: raw p<0.05 -> " + "; ".join(items))

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(OUT_DIR / "schoenfeld_ph_test_models0_4.csv", index=False)
    (OUT_DIR / "schoenfeld_ph_summary.txt").write_text("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))
    print(f"\nSaved: {OUT_DIR / 'schoenfeld_ph_test_models0_4.csv'}")
    print(f"Saved: {OUT_DIR / 'schoenfeld_ph_summary.txt'}")


if __name__ == "__main__":
    main()

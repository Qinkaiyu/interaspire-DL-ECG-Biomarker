# Manuscript Code Map

This file maps the retained scripts to the analysis sections they support.
Data and generated outputs are intentionally excluded from this public copy.

## DL-ECG Model 5 Training

- `training/model5/train_model5.py`: public command-line entry point for the
  two-phase, five-fold Model 5 training pipeline.
- `training/model5/model5_training/model.py`: ECGFounder plus structured-data
  multimodal survival architecture with a 32-dimensional ECG bottleneck.
- `training/model5/model5_training/training.py`: fold-specific preprocessing,
  tabular warm-up, multimodal Cox training, auxiliary ECG-biomarker loss,
  early stopping, checkpoint writing, and OOF prediction.
- `training/model5/configs/model5.example.json`: feature-name and data-column
  template; it contains no participant-level values.

The training package expects private cohort metadata, preprocessed ECG arrays,
fold assignments, and the ECGFounder checkpoint to be supplied locally. None
of these inputs or any generated weights are included.

## Core Model Development

- `scripts/run_final_models.py`: final nested clinical models with logistic
  prediction and Cox proportional-hazards summaries.
- `scripts/run_final_models_rf.py`: random-forest sensitivity model pipeline.
- `scripts/01_feature_selection/`: LASSO, elastic net, random forest, and
  stability selection for conventional ECG biomarker selection.

## Sensitivity Analyses

- `scripts/02_cvd_sensitivity/`: CVD composite endpoint model comparisons and
  feature-selection sensitivity analyses.
- `scripts/03_composite6/`: broader composite endpoint analyses.
- `scripts/05_analysis/schoenfeld_ph_models.py`: proportional-hazards
  diagnostics.

## Manuscript Tables

- `scripts/05_analysis/make_table1.py`: baseline table for the primary endpoint.
- `scripts/05_analysis/make_table1_extra.py`: additional baseline and DL-ECG
  risk-stratum tables.
- `scripts/05_analysis/make_table2.py`: model performance table.
- `scripts/05_analysis/make_table4_compact.py`: compact component-endpoint
  performance table.
- `scripts/05_analysis/supp_nri_idi_table.py`: reclassification summary table.

## Manuscript and Supplementary Figures

- `scripts/05_analysis/fig_cindex_and_roc.py`: ROC and C-index figure.
- `scripts/05_analysis/fig_cindex_incremental.py`: incremental C-index plot.
- `scripts/05_analysis/fig_km_curves.py`: Kaplan-Meier curves by DL-ECG risk
  group.
- `scripts/05_analysis/fig_km_normal_ecg.py`: risk stratification among normal
  and abnormal ECG subgroups.
- `scripts/05_analysis/fig_subgroup*.py`: subgroup performance figures.
- `scripts/05_analysis/fig_ecg_association.py` and
  `scripts/05_analysis/fig_ecg_forest.py`: ECG biomarker association figures.
- `scripts/05_analysis/fig_phewas*.py`: phenome-wide association figures.
- `scripts/05_analysis/supp_calibration_dca.py`: calibration and decision-curve
  analysis.
- `scripts/05_analysis/supp_crosstab*.py`: risk-group cross-tabulation and NRI
  visuals.
- `scripts/05_analysis/component_*.py` and
  `scripts/05_analysis/fig_cif_composite.py`: component endpoint KM/CIF
  analyses.

## ECG Matrix Utilities

- `scripts/04_ecg_preprocessing/`: preprocessing and visualization utilities for
  local ECG matrix files.
- `scripts/05_analysis/ecg_waveform_profile*.py`: waveform profile summaries by
  DL-ECG risk group.

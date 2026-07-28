# INTERASPIRE DL-ECG Risk Stratification Code

This repository contains the Model 5 training pipeline, statistical analyses,
and manuscript-generation code used for the INTERASPIRE DL-ECG risk
stratification project.

No participant-level data, ECG matrices, model weights, out-of-fold
predictions, generated figures, tables, or local machine paths are included.
The scripts expect private study data to be supplied locally by an approved
user.

## Repository Layout

- `scripts/run_final_models.py` and `scripts/run_final_models_rf.py`: final
  clinical model pipelines.
- `training/model5/`: two-phase training code for the final DL-ECG Model 5
  (`E2E MultiTask-32`), including five-fold cross-validation and OOF risk
  prediction.
- `scripts/01_feature_selection/`: ECG biomarker feature selection.
- `scripts/02_cvd_sensitivity/`: CVD composite sensitivity analyses.
- `scripts/03_composite6/`: broader composite endpoint analyses.
- `scripts/04_ecg_preprocessing/`: ECG matrix preprocessing utilities.
- `scripts/05_analysis/`: manuscript figures, tables, diagnostics, and
  supplementary analyses.

## Expected Local Inputs

The code uses relative paths so it can be run from the repository root after
placing approved local inputs in the following locations:

```text
data/INTERASPIRE_analysis_dataset.xlsx
data/qc_zero_lead_samples_with_outcomes.csv
data/ecg_matrices/preprocessed/
data/ecg_matrices/preprocessed_tiled/
data/ecg_matrices/preprocessed_3x4/
data/ecg_matrices/zero_filled/
checkpoints/12_lead_ECGFounder.pth
model5_cvd_mace4/
model5_cvd_mace6/
model5_mace4/
model5_mace6/
results/oof_predictions/
```

These directories are ignored by `.gitignore` and should not be committed.

## Setup

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run scripts from the repository root, for example:

```bash
python scripts/run_final_models.py
python scripts/05_analysis/make_table2.py
```

Train Model 5 from the repository root with approved local inputs:

```bash
python training/model5/train_model5.py \
  --metadata data/model5_metadata.parquet \
  --config training/model5/configs/model5.example.json \
  --backbone-checkpoint checkpoints/12_lead_ECGFounder.pth \
  --ecg-root data/ecg_matrices/preprocessed_3x4 \
  --output-dir outputs/model5 \
  --device cuda
```

See `training/model5/README.md` for the training input contract and output
structure.

## Privacy Notes

This public copy is intentionally code-only. Before pushing changes, run:

```bash
find . -type f \( -name '*.csv' -o -name '*.tsv' -o -name '*.xlsx' -o -name '*.parquet' -o -name '*.mat' -o -name '*.npy' -o -name '*.pt' -o -name '*.pth' -o -name '*.pkl' \)
rg -n '/Users''/|/home''/|/mnt''/|STUDY''ID[, ]|PAT''ID[, ]'
```

Both checks should return no private data files, model weights, or local
absolute paths. Pretrained ECGFounder weights and trained Model 5 checkpoints
must remain outside the public repository.

# Model 5 DL-ECG training code

This folder contains a code-only implementation of the two-phase training
pipeline used for the manuscript's final Model 5 (`E2E MultiTask-32`). The
model combines a pretrained ECGFounder waveform encoder with the Model 4
structured feature block and is trained with a Cox survival objective plus an
auxiliary ECG-biomarker objective.

## What is included

- Five-fold cross-validation and out-of-fold risk prediction.
- Phase 1: tabular branch warm-up.
- Phase 2: frozen ECGFounder backbone, 32-dimensional ECG bottleneck,
  multimodal fusion, Cox survival head, and auxiliary biomarker head.
- Fold-specific median imputation and standardization fitted on training data.
- Saving of fold checkpoints, preprocessing parameters, histories, OOF risk
  scores, and a compact metrics summary.

The package intentionally excludes cohort construction, ECG digitization,
participant-level data, fold assignments, pretrained weights, trained
checkpoints, OOF predictions, and manuscript results.

## Repository layout

```text
training/model5/
├── configs/model5.example.json
├── model5_training/
│   ├── backbone.py
│   ├── config.py
│   ├── data.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   └── training.py
└── train_model5.py
```

## Input contract

Supply one CSV or Parquet metadata table with one row per participant. It must
contain:

- an ECG path column; paths may be absolute or relative to `--ecg-root`;
- survival time and event columns;
- all tabular and auxiliary biomarker columns listed in the JSON config;
- optionally, an ID column and a fixed fold column.

Each ECG file must be a preprocessed `.mat` file containing a 2-D array under
the configured key (default: `feats`). Arrays may be `(12, T)` or `(T, 12)` and
must share a common length. This release does not perform ECG digitization,
filtering, resampling, or layout reconstruction.

The example configuration uses the original INTERASPIRE variable names. For a
new dataset, either map columns to these names or edit the JSON configuration.
Do not commit the resulting metadata or ECG files.

## Setup

Use Python 3.10 or newer and install the unified repository requirements from
the repository root with `pip install -r requirements.txt`.

Obtain the pretrained ECGFounder checkpoint separately and keep it outside the
repository, for example under an ignored `checkpoints/` directory.

## Training

```bash
python training/model5/train_model5.py \
  --metadata data/model5_metadata.parquet \
  --config training/model5/configs/model5.example.json \
  --backbone-checkpoint checkpoints/12_lead_ECGFounder.pth \
  --ecg-root data/ecg_matrices \
  --output-dir outputs/model5 \
  --device cuda
```

If the configured fold column is absent, the script creates a reproducible
stratified five-fold split. To reproduce a prespecified analysis, include the
fold column in the private metadata table.

## Expected outputs

The output directory contains:

- `fold_0.pt` through `fold_4.pt`: model states and preprocessing parameters;
- `fold_*_history.json`: phase-specific training histories;
- `oof_predictions.csv`: participant IDs, fold labels, outcomes, and log-risk;
- `metrics.json`: pooled Harrell C-index and cohort counts;
- `run_config.json`: non-sensitive run settings and feature names.

## Privacy check before release

From the repository root, both commands should return no study files,
model weights, or machine-local absolute paths:

```bash
find . -type f \( -name '*.csv' -o -name '*.parquet' -o -name '*.mat' -o -name '*.npy' -o -name '*.pt' -o -name '*.pth' \)
rg -n '/Users''/|/home''/|/mnt''/|STUDY''ID[, ]|PAT''ID[, ]'
```

Review checkpoint licensing and attribution requirements before distributing
any third-party pretrained weights. No pretrained weights are included here.

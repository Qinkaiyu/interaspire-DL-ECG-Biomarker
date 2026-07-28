# Release checklist

- [x] Model 5 training and manuscript-analysis code are combined in one
  code-only public directory.
- [x] No participant-level tables, ECG matrices, fold assignments, predictions,
  results, or model weights are included.
- [x] Machine-local paths have been replaced by CLI arguments and relative
  example paths.
- [x] The example configuration contains variable names only.
- [x] Private and derived file types are covered by `.gitignore`.
- [ ] Select and add the repository license.
- [ ] Add the manuscript citation, DOI, and contact details when available.
- [ ] Confirm the attribution and redistribution conditions for ECGFounder code
  and pretrained weights.
- [ ] Run the privacy commands in `README.md` immediately before the public
  commit.
- [ ] Review the staged Git diff before pushing.

Pretrained ECGFounder weights and trained Model 5 checkpoints should be
distributed only through a separately approved channel, if permitted.

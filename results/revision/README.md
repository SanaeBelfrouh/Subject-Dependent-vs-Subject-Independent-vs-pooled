# Output of notebook 05 (control analyses added in revision)

`revision_results_E1-E9_E11.zip` is the merged output of the Kaggle sessions of
`notebooks/05_revision_experiments.ipynb`. Unzipped, it gives a `revision/` folder with:

- `runs.csv`: one row per trained model and target subject (experiment, regime, variant, model,
  seed, balanced accuracy, kappa, weighted F1), for experiments E1, E3, E4, E6, E7, E8, E9 and E11;
- `preds*.pkl`: the per-trial predictions of every model;
- one summary CSV per experiment (`E1_regimes_over_runs.csv`, `E9_slow_lda.csv`,
  `E11_fast_dynamics_lda.csv`, ...), written by the summary cell of the notebook;
- files ending in `_in1`/`_in2`: checkpoints of earlier sessions merged by the resume logic.

| Experiment | Content | Manuscript |
|---|---|---|
| E1 | seeded repeats of SD, pooled and SI | Tables 3 and 8, Table A.10 |
| E3 | equal-size pooled model | Tables 3 and 8 |
| E4 | SD under group five-fold CV (storage runs) | Table 4 |
| E6 | permutation nulls | Table 8 |
| E7 | central electrodes, 1 Hz high-pass | Table 8, Fig. 10 |
| E8 | EEGNet-8,2 under the three regimes | Table 8 |
| E9 | pre-onset, baseline-corrected, low-pass, high-pass, detrended maps; slow-feature LDA | Table 8, Fig. 10 |
| E11 | faster dynamics (1–40 Hz): band power, covariance, EEGNet | Table 9 |

`python scripts/revision_tables.py <folder containing revision/> <out_dir>` and
`python scripts/fig10_controls.py revision/runs.csv <out_prefix>` read this folder.

Not in this archive yet: E2 (wider encoders and Euclidean alignment, Table 8) and E10 (recording-block
diagnostics and reconstructed-block cross-validation, Table 8), which ran in separate sessions.

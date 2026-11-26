# Predictions of the original three-regime runs

- `predictions.npz`: keys `sd`, `pooled`, `si`, each a dictionary {subject index 0–14: predicted class of
  the 50 validation-partition trials}, written by `notebooks/01_paper_results.ipynb`. These are the exact
  runs reported in Table 3 (49.73 / 41.07 / 22.53%).
- `test_labels.npz`: key `labels`, an array (15 subjects × 50 trials) of the true classes of the same
  trials, in the same order. It was taken from the output of notebook 05, which loads the data with the
  code of notebook 01; with these labels, `scripts/reproduce_tables.py` returns Table 3 exactly.

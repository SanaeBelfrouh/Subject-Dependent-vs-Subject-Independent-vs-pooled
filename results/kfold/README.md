# 5-fold cross-validation of the original analysis (Tables 4–5)

`per_subject_kfold.csv` holds the per-subject balanced accuracies (%) of the notebook-02 run reported in
the paper, under shuffled and blocked stratified 5-fold cross-validation over the 350 trials of each
subject, and their difference. They are copied from the saved output of
`notebooks/02_sd_kfold_ci.ipynb` (cell "SD via 5-fold CV over 350 trials"), which also prints the means,
confidence intervals, kappa and weighted F1. The per-trial predictions of that run were not kept.
`python scripts/kfold_tables_from_summary.py` recomputes the tables and Wilcoxon tests from this file.

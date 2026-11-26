"""Tables 4-5 (shuffled and blocked 5-fold CV of the original analysis) from per-subject accuracies.

The per-trial predictions of the notebook-02 run reported in the paper were not kept. Its per-subject
balanced accuracies (one decimal) are printed in the saved output of notebooks/02_sd_kfold_ci.ipynb and
are copied to results/kfold/per_subject_kfold.csv. This script recomputes from them the means, the 95%
t-intervals over subjects and the Wilcoxon tests, and adds the held-out row of Table 4 and the Wilcoxon
tests of Table 3 (SI vs chance, SD vs SI) from results/paper, which scripts/reproduce_tables.py prints
only after its k-fold section. Because the inputs are rounded to 0.1 pp, the last
digit can differ slightly from the values printed by notebook 02.

Usage: python scripts/kfold_tables_from_summary.py   (reads $RESULTS_DIR/kfold or ./results/kfold)
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.environ.get('RESULTS_DIR') or os.path.join(ROOT, 'results')
P = pd.read_csv(os.path.join(RES, 'kfold', 'per_subject_kfold.csv'))
z = np.load(os.path.join(RES, 'paper', 'predictions.npz'), allow_pickle=True)
Y = np.load(os.path.join(RES, 'paper', 'test_labels.npz'))['labels']
HELD = {r: np.array([balanced_accuracy_score(Y[s], np.asarray(z[r].item()[s])) * 100 for s in range(15)])
        for r in ['sd', 'pooled', 'si']}
q = lambda v: np.round(np.asarray(v, float), 6)   # exact zeros and ties treated as such


def t_ci(v):
    v = np.asarray(v, float)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), v.mean() - h, v.mean() + h


print('Table 4 - SD under 5-fold CV over 350 trials (mean over 15 subjects, 95% t-CI over subjects)')
m, lo, hi = t_ci(HELD['sd'])
print(f'  {"held-out":9s} {m:6.2f}%  [{lo:.1f}, {hi:.1f}]  SD over subjects {HELD["sd"].std(ddof=1):.2f}')
for scheme in ['shuffled', 'blocked']:
    m, lo, hi = t_ci(P[scheme])
    print(f'  {scheme:9s} {m:6.2f}%  [{lo:.1f}, {hi:.1f}]  SD over subjects {P[scheme].std(ddof=1):.2f}')

d = P.shuffled_minus_blocked   # as printed by notebook 02 (from unrounded accuracies)
print('\nTable 5 - shuffled minus blocked, per subject (pp)')
print(P.to_string(index=False))
print(f'  mean {d.mean():+.2f} pp, range {d.min():+.1f} to {d.max():+.1f}')
print(f'  Wilcoxon signed-rank, shuffled vs blocked: p = {stats.wilcoxon(P.shuffled, P.blocked).pvalue:.2f}')
for scheme in ['shuffled', 'blocked']:
    p = stats.wilcoxon(P[scheme] - 20, alternative='greater').pvalue
    print(f'  Wilcoxon, {scheme} > 20% chance: p = {p:.2g}')

print('\nWilcoxon tests of Table 3 (two-sided, n = 15)')
print(f'  SI vs chance (20%):  p = {stats.wilcoxon(q(HELD["si"] - 20)).pvalue:.2f}')
print(f'  SD vs SI, paired:    p = {stats.wilcoxon(q(HELD["sd"]), q(HELD["si"])).pvalue:.2g}')

"""Reproduce every table and statistic in the paper from the saved predictions (no TensorFlow).

Inputs (relative to the repository root, or to $RESULTS_DIR if set):
  results/paper/predictions.npz   keys sd / si / pooled -> {subject_id: y_pred (50,)}   (written by 01)
  results/paper/test_labels.npz   true labels of the 50 validation trials per subject
  results/kfold/kfold_preds.pkl   {'shuffled'|'blocked': {subject_id: y_pred (350,)}}    (written by 02)
  results/kfold/kfold_labels.npz  true labels of the 350 trials per subject

Usage:  python scripts/reproduce_tables.py
"""
import os, pickle
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.environ.get('RESULTS_DIR') or os.path.join(ROOT, 'results')
N_SUBJ = 15
CHANCE = 20.0
REGIMES = ['sd', 'pooled', 'si']


def _as_dict(obj):
    """Normalise a per-subject container to {int subject_id: 1-D int array}."""
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.ndim == 0:
        obj = obj.item()
    if isinstance(obj, dict):
        return {int(k): np.asarray(v).astype(int).ravel() for k, v in obj.items()}
    arr = np.asarray(obj)
    if arr.dtype == object:                       # array of per-subject arrays
        return {i: np.asarray(v).astype(int).ravel() for i, v in enumerate(arr)}
    if arr.ndim == 2 and arr.shape[0] == N_SUBJ:  # (subjects, trials)
        return {i: arr[i].astype(int) for i in range(N_SUBJ)}
    raise ValueError(f'unrecognised per-subject layout: {type(obj)} {getattr(arr, "shape", None)}')


def load_labels(path, prefer=None):
    """Labels file: either one key per subject, or a single key holding all subjects."""
    z = np.load(path, allow_pickle=True)
    keys = list(z.keys())
    if prefer and prefer in keys:
        return _as_dict(z[prefer])
    if len(keys) == 1:
        return _as_dict(z[keys[0]])
    if all(k.lstrip('S').lstrip('s').isdigit() for k in keys):
        return {int(k.lstrip('Ss')): np.asarray(z[k]).astype(int).ravel() for k in keys}
    raise ValueError(f'{path}: unrecognised keys {keys}')


def metrics(yt, yp):
    return dict(bal_acc=balanced_accuracy_score(yt, yp) * 100,
                kappa=cohen_kappa_score(yt, yp),
                wf1=f1_score(yt, yp, average='weighted') * 100)


def t_ci(v):
    v = np.asarray(v, float)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), v.mean() - h, v.mean() + h


def check(pred, true, n_trials, what):
    assert sorted(pred) == sorted(true) == list(range(N_SUBJ)), f'{what}: subject ids {sorted(pred)}'
    for s in pred:
        assert len(pred[s]) == len(true[s]) == n_trials, f'{what}: subject {s} has {len(pred[s])} trials'
        assert (pred[s] >= 0).all(), f'{what}: subject {s} has unfilled folds'


def main():
    pd.set_option('display.width', 120)
    name = lambda s: f'S{s + 1:02d}'

    # ------------------------------------------------------------ cross-session regimes
    z = np.load(os.path.join(RES, 'paper', 'predictions.npz'), allow_pickle=True)
    preds = {r: _as_dict(z[r]) for r in REGIMES}
    ytrue = load_labels(os.path.join(RES, 'paper', 'test_labels.npz'))
    for r in REGIMES:
        check(preds[r], ytrue, 50, r)
    R = pd.DataFrame([dict(subject=name(s), regime=r, **metrics(ytrue[s], preds[r][s]))
                      for s in range(N_SUBJ) for r in REGIMES])
    T1 = R.groupby('regime')[['bal_acc', 'kappa', 'wf1']].mean().loc[REGIMES]
    T1.loc['SD - SI'] = T1.loc['sd'] - T1.loc['si']
    T1.loc['Pooled - SI'] = T1.loc['pooled'] - T1.loc['si']
    T1 = T1.rename(index={'sd': 'SD', 'pooled': 'Pooled', 'si': 'SI'})
    T1.columns = ['Balanced acc (%)', "Cohen's kappa", 'Weighted F1 (%)']
    print('Table 1 - cross-session regimes (mean over 15 subjects)')
    print(T1.to_string(formatters={'Balanced acc (%)': '{:.2f}'.format,
                                   "Cohen's kappa": '{:.3f}'.format,
                                   'Weighted F1 (%)': '{:.2f}'.format}))

    # ------------------------------------------------------------ 5-fold CV
    with open(os.path.join(RES, 'kfold', 'kfold_preds.pkl'), 'rb') as f:
        kp = pickle.load(f)
    kpred = {sch: _as_dict(kp[sch]) for sch in ['shuffled', 'blocked']}
    ky = load_labels(os.path.join(RES, 'kfold', 'kfold_labels.npz'))
    for sch in kpred:
        check(kpred[sch], ky, 350, sch)
    K = pd.DataFrame({sch: [balanced_accuracy_score(ky[s], kpred[sch][s]) * 100 for s in range(N_SUBJ)]
                      for sch in ['shuffled', 'blocked']}, index=[name(s) for s in range(N_SUBJ)])
    sd_cs = R[R.regime == 'sd'].set_index('subject')['bal_acc'].loc[K.index]

    print('\nTable 2 - subject-dependent estimates (mean, t-based 95% CI over 15 subjects)')
    for label, v in [('Cross-session (50 trials)', sd_cs), ('Shuffled 5-fold (350 trials)', K.shuffled),
                     ('Blocked 5-fold (350 trials)', K.blocked)]:
        m, lo, hi = t_ci(v)
        print(f'  {label:30s} {m:6.2f}  [{lo:.1f}, {hi:.1f}]   between-subject std {np.std(v, ddof=1):.2f}')

    K['difference'] = K.shuffled - K.blocked
    print('\nTable 3 - per-subject 5-fold CV balanced accuracy (%)')
    print(K.round(1).to_string())
    print(f"  {'mean':10s}{K.shuffled.mean():>9.1f}{K.blocked.mean():>9.1f}{K.difference.mean():>+11.1f}")

    # ------------------------------------------------------------ statistics
    # Round away float noise (e.g. 20.000000000000004) so exact zeros and ties are treated as such.
    q = lambda v: np.round(np.asarray(v, float), 6)
    si = q(R[R.regime == 'si'].set_index('subject')['bal_acc'].loc[K.index])
    print('\nStatistics (Wilcoxon signed-rank, two-sided, n = 15)')
    for label, res in [('SI vs chance (20%), one-sample', stats.wilcoxon(si - CHANCE)),
                       ('SD vs SI, paired', stats.wilcoxon(q(sd_cs), si)),
                       ('Shuffled vs blocked, paired', stats.wilcoxon(q(K.shuffled), q(K.blocked)))]:
        print(f'  {label:34s} W = {res.statistic:5.1f}   p = {res.pvalue:.4f}')


if __name__ == '__main__':
    main()

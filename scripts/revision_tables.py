"""Read revision_results (notebook 05 output) and print every number the revision needs, plus LaTeX
tables. Usage: python scripts/revision_tables.py <dir containing revision/>  <out_dir>"""
import sys, os, json
import numpy as np, pandas as pd
from scipy import stats

RES, OUTD = sys.argv[1], sys.argv[2]
D = os.path.join(RES, 'revision'); os.makedirs(OUTD, exist_ok=True)
R = pd.read_csv(f'{D}/runs.csv')
NUM = {}

def q(exp, regime, variant=None, model=None, seeds=None):
    x = R[(R.exp == exp) & (R.regime == regime)]
    if variant is not None: x = x[x.variant == variant]
    if model is not None: x = x[x.model == model]
    if seeds is not None: x = x[x.seed.isin(seeds)]
    return x

def summ(x):
    pr = x.groupby('seed').bal_acc.mean(); ps = x.groupby('subject').bal_acc.mean()
    v = ps.values; n = len(v)
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return dict(runs=len(pr), n_subj=n, mean=pr.mean(), sd_runs=pr.std(ddof=1) if len(pr) > 1 else np.nan,
                min_run=pr.min(), max_run=pr.max(), ci_lo=v.mean() - h, ci_hi=v.mean() + h,
                kappa=x.kappa.mean(), wf1=x.wf1.mean())

def subj(x): return x.groupby('subject').bal_acc.mean()
def n_gt(a, b):
    a, b = subj(a), subj(b); j = a.index.intersection(b.index); return int((a[j] > b[j]).sum())
def wil(a, b, alt='two-sided'):
    a, b = subj(a), subj(b); j = a.index.intersection(b.index)
    return stats.wilcoxon(a[j], b[j], alternative=alt).pvalue if len(j) >= 5 else np.nan

def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1, run)
    return adj

lines = []
def put(k, v): NUM[k] = v; lines.append(f'{k:45s} {v}')

# ---- E1
for reg, seeds in [('sd', None), ('pooled', None), ('si', None)]:
    x = q('E1', reg)
    if len(x):
        s = summ(x)
        for k, v in s.items(): put(f'E1.{reg}.{k}', round(v, 3) if isinstance(v, float) else v)
if len(q('E1', 'pooled')) and len(q('E1', 'si')):
    put('E1.gap_pooled_si', round(subj(q('E1', 'pooled')).mean() - subj(q('E1', 'si')).mean(), 2))
    put('E1.p_pooled_vs_si', wil(q('E1', 'pooled'), q('E1', 'si')))
    put('E1.n_pooled_gt_si', n_gt(q('E1', 'pooled'), q('E1', 'si')))
if len(q('E1', 'sd')) and len(q('E1', 'pooled')):
    put('E1.p_sd_vs_pooled', wil(q('E1', 'sd'), q('E1', 'pooled')))
    put('E1.n_sd_gt_pooled', n_gt(q('E1', 'sd'), q('E1', 'pooled')))
if len(q('E1', 'si')):
    put('E1.p_si_vs_chance', stats.wilcoxon(subj(q('E1', 'si')) - 20, alternative='greater').pvalue)

# ---- E3
if len(q('E3', 'pooled_eq')):
    eq = q('E3', 'pooled_eq'); si0 = q('E1', 'si', seeds=sorted(eq.seed.unique()))
    put('E3.pooled_eq.mean', round(subj(eq).mean(), 2)); put('E3.si_same_seeds.mean', round(subj(si0).mean(), 2))
    put('E3.gap', round(subj(eq).mean() - subj(si0).mean(), 2)); put('E3.p', wil(eq, si0))
    put('E3.n_eq_gt_si', n_gt(eq, si0))

# ---- E6
for reg in ['sd', 'pooled']:
    x = q('E6', reg)
    if len(x):
        null = x.groupby('seed').bal_acc.mean(); obs = summ(q('E1', reg))['mean']
        put(f'E6.{reg}.n_perm', len(null)); put(f'E6.{reg}.null_mean', round(null.mean(), 2))
        put(f'E6.{reg}.null_max', round(null.max(), 2)); put(f'E6.{reg}.observed', round(obs, 2))
        put(f'E6.{reg}.p_perm', round((1 + (null >= obs).sum()) / (1 + len(null)), 4))

# ---- E4 / E4b
for sch in ['group', 'blocked', 'shuffled']:
    x = q('E4', f'sd_cv_{sch}')
    if len(x):
        for k, v in summ(x).items(): put(f'E4.{sch}.{k}', round(v, 3) if isinstance(v, float) else v)
if len(q('E4', 'sd_cv_group')):
    put('E4.p_group_vs_chance', stats.wilcoxon(subj(q('E4', 'sd_cv_group')) - 20, alternative='greater').pvalue)

# ---- E5
f5 = f'{D}/E5_drift_controls.csv'
if os.path.exists(f5):
    E5 = pd.read_csv(f5)
    for c in E5.columns[1:]: put(f'E5.mean.{c}', round(E5[c].mean(), 2))
    for c in [c for c in E5.columns if c.startswith('slow_')]:
        put(f'E5.p_vs_chance.{c}', stats.wilcoxon(E5[c] - 20, alternative='greater').pvalue)

# ---- E7
for v in ['central', 'hp1', 'hp1_central']:
    for reg in ['sd', 'pooled']:
        x = q('E7', reg, variant=v)
        if len(x):
            s = summ(x); put(f'E7.{v}.{reg}.mean', round(s['mean'], 2)); put(f'E7.{v}.{reg}.sd_runs', round(s['sd_runs'], 2))
            put(f'E7.{v}.{reg}.p_vs_base', wil(x, q('E1', reg, seeds=sorted(x.seed.unique()))))
    x = q('E7', 'sd_cv_group', variant=v)
    if len(x):
        put(f'E7.{v}.sd_cv_group.mean', round(summ(x)['mean'], 2))
        put(f'E7.{v}.sd_cv_group.p_vs_chance', stats.wilcoxon(subj(x) - 20, alternative='greater').pvalue)

# ---- E2
for v, mw in [('base', 'w100k'), ('base', 'w330k'), ('ea', 'w25k'), ('ea', 'w330k')]:
    for reg in ['pooled', 'si']:
        x = q('E2', reg, variant=v, model=mw)
        if len(x):
            s = summ(x); put(f'E2.{v}.{mw}.{reg}.mean', round(s['mean'], 2)); put(f'E2.{v}.{mw}.{reg}.runs', s['runs'])
            base = q('E1', reg, seeds=sorted(x.seed.unique()))
            put(f'E2.{v}.{mw}.{reg}.p_vs_base', wil(x, base))
            if reg == 'pooled': put(f'E2.{v}.{mw}.p_vs_sd', wil(x, q('E1', 'sd', seeds=sorted(x.seed.unique()))))

# ---- E8
for reg in ['sd', 'pooled', 'si']:
    x = q('E8', reg, model='eegnet')
    if len(x):
        s = summ(x); put(f'E8.{reg}.mean', round(s['mean'], 2)); put(f'E8.{reg}.sd_runs', round(s['sd_runs'], 2)); put(f'E8.{reg}.runs', s['runs'])

open(f'{OUTD}/numbers.txt', 'w').write('\n'.join(lines) + '\n')
json.dump(NUM, open(f'{OUTD}/numbers.json', 'w'), indent=1, default=float)
print('\n'.join(lines))

# ---- supplementary table: per-subject means over seeded runs (SD, pooled, SI)
if len(q('E1', 'sd')):
    P = R[R.exp == 'E1'].groupby(['subject', 'regime']).bal_acc.mean().unstack()[['sd', 'pooled', 'si']]
    rows = '\n'.join(f'{s} & {r.sd:.1f} & {r.pooled:.1f} & {r.si:.1f} & {r.pooled - r.si:+.1f} \\\\' for s, r in P.iterrows())
    tab = (r'''\begin{table}[t]
\centering
\caption{Per-subject balanced accuracy (\%) on the held-out split, averaged over the seeded runs
(SD and pooled: five runs; SI: three runs). Gap = pooled $-$ SI.}
\label{tab:persubject}
\begin{tabular}{lcccc}
\toprule
Subject & SD & Pooled & SI & Gap (pp) \\
\midrule
''' + rows + '\n\\midrule\n' + f'\\textbf{{Mean}} & {P.sd.mean():.1f} & {P.pooled.mean():.1f} & {P.si.mean():.1f} & {P.pooled.mean() - P.si.mean():+.1f} \\\\\n'
           + '\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    open(f'{OUTD}/table_persubject.tex', 'w').write(tab)

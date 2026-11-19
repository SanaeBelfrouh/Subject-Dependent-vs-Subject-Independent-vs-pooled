# Subject-dependent, subject-independent, or pooled? — EEG imagined-speech evaluation regimes

Code and saved results for the paper below. We train one small (~25K-parameter) time-distributed CNN on
masked 32×32 scalp topography maps from BCIC2020 Track 3 (5-class imagined speech, 15 subjects) and evaluate
it under three regimes: subject-dependent (SD, one model per subject), pooled (one model on all subjects,
tested on seen subjects), and subject-independent (SI, leave-one-subject-out). SD reaches 49.7% balanced
accuracy, pooled 41.1%, and SI 22.5%, near chance (not significantly above 20% across subjects, Wilcoxon
p = 0.10). A 5-fold CV over all 350 trials per subject gives 45.1% (shuffled) and 43.0% (blocked), with
S12 the only clear leakage outlier.

> Belfrouh, Salmam, Errattahi. *Subject-dependent, subject-independent, or pooled? What each evaluation
> regime really measures in EEG imagined-speech decoding.* Submitted to *Journal of Neural Engineering*.
> See `CITATION.cff`.

## Data

BCIC2020 Track 3 (imagined speech), available from https://doi.org/10.17605/OSF.IO/PQ7VB.
Place the 15 training and 15 validation files like this (or point `BCIC_DATA` at the folder):

```
data/
  Train/Data_Sample01.mat ... Data_Sample15.mat
  valid/Data_Sample01.mat ... Data_Sample15.mat
```

## Quick check (seconds, no GPU, no dataset)

```
pip install numpy scipy scikit-learn pandas
python scripts/reproduce_tables.py
```

This recomputes Tables 1–3 and all reported Wilcoxon tests from the saved predictions in `results/`.

## Full rerun

Run the notebooks in order: `01` → `02` → `03`. `04` and `05` are independent (`05` reuses the map cache
of `01` when it is attached).

- `01_paper_results.ipynb` builds the map cache (`maps.pkl`) and trains the three regimes.
- `02_sd_kfold_ci.ipynb` reads the map cache written by `01`. It needs several GPU hours on a T4, so it
  is checkpointed per (scheme, subject, fold) and resumes where it stopped; it may take several Kaggle sessions.
- `03_make_figures.ipynb` draws all figures from the outputs of `01` and `02`.
- `04_band_probe.ipynb` runs the linear probes used in Fig. 9(b).
- `05_revision_experiments.ipynb` runs the control analyses added in revision (seeded repeats, equal-size
  pooled model, wider encoders and Euclidean alignment, instruction-block cross-validation, drift and
  artefact controls, permutation nulls, EEGNet under the three regimes). It reuses the data loading, maps
  and model of `01`/`02` without changing them, runs each part in a fresh process (one part per GPU),
  checkpoints every model, and can be resumed across Kaggle sessions; it writes `revision_results.zip`.

Paths: on Kaggle the notebooks use `/kaggle/input` and `/kaggle/working`. Elsewhere, set `BCIC_DATA`
(dataset folder) and `RESULTS_DIR` (output root); they default to `./data` and `./results`.
Notebooks skip work whose outputs already exist, so **point `RESULTS_DIR` at an empty folder for a fresh
run**; otherwise `01` reloads the saved predictions and `02` can overwrite `results/kfold/kfold_preds.pkl`.

**Determinism.** Data splits are seeded, but TensorFlow weight initialisation is not, so a full rerun differs
by about 1–2 pp. The files in `results/` are the exact runs reported in the paper.

**Mask.** The effective mask keeps 80.7% of the 32×32 grid: pixels within 1.2× the median inter-electrode
spacing of an electrode, intersected with the convex hull where cubic interpolation is defined.

## What produces what

| Paper item | Produced by |
|---|---|
| Table 1 (SD / pooled / SI: balanced acc, kappa, weighted F1) | `01_paper_results.ipynb`, `scripts/reproduce_tables.py` |
| Table 2 (SD 5-fold means, 95% CIs, cross-session CI) | `02_sd_kfold_ci.ipynb`, `scripts/reproduce_tables.py` |
| Table 3 (per-subject shuffled vs blocked) | `02_sd_kfold_ci.ipynb`, `scripts/reproduce_tables.py` |
| Wilcoxon tests (SI vs chance, SD vs SI, shuffled vs blocked) | `scripts/reproduce_tables.py` |
| Figs. 1–5 (pipeline, mask, filmstrip, class means, architecture) | `03_make_figures.ipynb` (needs the dataset) |
| Fig. 6 (regimes), Fig. 7 (confusion matrices), Fig. 8 (protocols, leakage control) | `03_make_figures.ipynb` |
| Fig. 9 (ablation) | `03_make_figures.ipynb`, with band-probe values from `04_band_probe.ipynb` |

The published figures are in `figs/`.

## Not included

The global-spatial-pooling diagnostic (27.9%, Fig. 9a) and the regularisation sweep are not part of this
release; Fig. 9(a) uses the measured value directly.

## Environment

TensorFlow 2.20, Python 3.11, Kaggle 2× T4. Pinned versions are in `requirements.txt`.

## License

MIT, see `LICENSE`.

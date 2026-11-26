# Subject-dependent, subject-independent, or pooled? — EEG imagined-speech evaluation regimes

Code and saved results for the paper below. We train one small (~25K-parameter) time-distributed CNN on
masked 32×32 scalp topography maps from BCIC2020 Track 3 (5-class imagined speech, 15 subjects) and evaluate
it under three regimes: subject-dependent (SD, one model per subject), pooled (one model on all subjects,
tested on seen subjects), and subject-independent (SI, leave-one-subject-out). SD reaches 49.7% balanced
accuracy, pooled 41.1%, and SI 22.5%, near chance (not significantly above 20% across subjects, Wilcoxon
p = 0.10). A 5-fold CV over all 350 trials per subject gives 45.1% (shuffled) and 43.0% (blocked), with
S12 the only clear leakage outlier.

The control analyses added in revision (notebook `05`) show that the class information measured by every
regime is a slow offset below 1 Hz that is already present before imagery onset: baseline correction
brings every regime to chance, and so does keeping reconstructed recording blocks whole in
cross-validation.

> Belfrouh, Salmam, Errattahi. *Subject-dependent, subject-independent, or pooled? What each evaluation
> regime really measures in EEG imagined-speech decoding.* Submitted to *Array*. See `CITATION.cff`.

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

This recomputes the three-regime, protocol and leakage tables (Tables 3–5) and the Wilcoxon tests of the
original analysis from the saved predictions in `results/`.

## Full rerun

Run the notebooks in order: `01` → `02` → `03`. `05` is independent (it reuses the map cache of `01` when it
is attached).

- `01_paper_results.ipynb` builds the map cache (`maps.pkl`) and trains the three regimes.
- `02_sd_kfold_ci.ipynb` reads the map cache written by `01`. It needs several GPU hours on a T4, so it
  is checkpointed per (scheme, subject, fold) and resumes where it stopped; it may take several Kaggle sessions.
- `03_make_figures.ipynb` draws Figs. 1–9 from the outputs of `01` and `02`. The linear-probe values of
  Fig. 9 (44.5% and 33.5%) and the global-pooling value (27.9%) are entered as measured constants; the
  notebooks that measured them are not part of this release.
- `05_revision_experiments.ipynb` runs the control analyses added in revision (seeded repeats, equal-size
  pooled model, wider encoders and Euclidean alignment, instruction-block cross-validation, drift and
  artefact controls, pre-onset and slow-component controls, recording-block diagnostics, faster-dynamics
  controls, permutation nulls, EEGNet under the three regimes). It reuses the data loading, maps
  and model of `01`/`02` without changing them, runs each part in a fresh process (one part per GPU),
  checkpoints every model, and can be resumed across Kaggle sessions; it writes `revision_results.zip`.

Scripts for the revision (they read the output of `05`):

- `scripts/revision_tables.py <dir containing revision/> <out_dir>` recomputes the numbers of the seeded
  repeats, equal-size pooled model, capacity, alignment, group-fold, permutation, artefact and EEGNet
  analyses from `revision/runs.csv`, and writes the per-subject table (Table A.10). The pre-onset,
  recording-block and faster-dynamics results are summarised by notebook `05` itself, in the CSV files of
  `revision_results.zip`.
- `scripts/fig10_controls.py <revision/runs.csv> <out_prefix>` draws Fig. 10.
- `scripts/fig9_revised.py <out_prefix>` redraws Fig. 9 with the same measured values and clearer
  annotations.

Paths: on Kaggle the notebooks use `/kaggle/input` and `/kaggle/working`. Elsewhere, set `BCIC_DATA`
(dataset folder) and `RESULTS_DIR` (output root); they default to `./data` and `./results`.
Notebooks skip work whose outputs already exist, so **point `RESULTS_DIR` at an empty folder for a fresh
run**; otherwise `01` reloads the saved predictions and `02` can overwrite `results/kfold/kfold_preds.pkl`.

**Determinism.** In `01` and `02`, data splits are seeded but TensorFlow weight initialisation is not, so a
full rerun differs by about 1–2 pp; the files in `results/` are the exact runs reported in the paper.
Notebook `05` also seeds weight initialisation.

**Mask.** The effective mask keeps 80.7% of the 32×32 grid: pixels within 1.2× the median inter-electrode
spacing of an electrode, intersected with the convex hull where cubic interpolation is defined.

## What produces what

Table and figure numbers follow the revised manuscript.

| Paper item | Produced by |
|---|---|
| Table 3 (SD / pooled / SI: balanced acc, kappa, weighted F1) | `01_paper_results.ipynb`, `scripts/reproduce_tables.py`; seeded repeats: `05` |
| Table 4 (SD under the evaluation protocols, 95% CIs) | `02_sd_kfold_ci.ipynb`, `scripts/reproduce_tables.py`; group folds: `05` |
| Table 5 (per-subject shuffled vs blocked) | `02_sd_kfold_ci.ipynb`, `scripts/reproduce_tables.py` |
| Tables 8 and 9 (control analyses, faster dynamics) | `05_revision_experiments.ipynb` (summary CSV files), `scripts/revision_tables.py` |
| Table A.10 (per subject) | `scripts/revision_tables.py` |
| Wilcoxon tests of the original analysis | `scripts/reproduce_tables.py` |
| Figs. 1–4 and 6 (pipeline, mask, filmstrip, architecture, class means) | `03_make_figures.ipynb` (needs the dataset) |
| Fig. 5 (regimes), Fig. 7 (confusion matrices), Fig. 8 (protocols, leakage control) | `03_make_figures.ipynb` |
| Fig. 9 (design choices) | `03_make_figures.ipynb`; revised drawing: `scripts/fig9_revised.py` |
| Fig. 10 (where the class information lies) | `scripts/fig10_controls.py` |

The figures of the original submission are in `figs/` (file names follow the order in which they were
made, not the figure numbers above).

## Not included

The global-spatial-pooling diagnostic (27.9%, Fig. 9a), the band-power linear probes (Fig. 9b) and the
regularisation sweep are not part of this release; Fig. 9 uses the measured values directly.

## Environment

TensorFlow 2.20, Python 3.11, Kaggle 2× T4. Pinned versions are in `requirements.txt`.

## License

MIT, see `LICENSE`.

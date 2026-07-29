# Code map

Two tiers. `src/` is the library that `predict.py` needs at inference time;
`src/experiments/` is everything used to *choose* the model and is not needed to
run it.

## `src/` — inference library

Read in this order:

| File | Lines to look at first |
|---|---|
| `dsp.py` | All signal processing, written from scratch. Framing, energy, autocorrelation F0 (FFT-based, parabolic interpolation, octave-jump protection), adaptive speech/silence mask, mel filterbank, DCT cepstra, spectral statistics. Knows nothing about pauses — it only ever sees an already-truncated waveform. |
| `features_eot.py` | **`causal_slice()` is the causality audit point** — the single place audio is handed to feature code, returning `x[:int(pause_start*sr)]`. Then `base_features()` (the 8 feature families) and `extract_turn()`. |
| `dataset.py` | Builds the pause table for a folder; `add_relative_features()` is the within-turn-relative family (expanding window shifted by one pause — causal). |
| `model.py` | `BlendModel` (HistGradientBoosting ×5 seeds ⊕ L2 logistic, rank-averaged) and `risk_weights()` — the duration-based cost weighting that matches the scorer's asymmetry. |
| `metric.py` | `score.py`'s policy sweep reimplemented in-process, so model selection happens on the graded metric instead of accuracy or AUC. |
| `vap.py`, `vap_train.py` | Voice Activity Projection objective, reimplemented from scratch. Measured, documented, **not used by the shipped model** — see §7 of `SUMMARY.html`. |

## `src/experiments/` — how the model was chosen

Not imported by `predict.py`. Each script adds `src/` to `sys.path` and resolves
paths from the repo root, so run them from anywhere.

| File | What it does |
|---|---|
| `cv.py` | Held-out evaluation: 5-fold `GroupKFold` by `turn_id` + cross-language transfer. Rebuilds the feature cache if absent. |
| `exp.py` | The ablation sweep in `RUNLOG.md` rows 6–7 (hold filtering, risk-weight shapes, feature selection) plus `restricted_auc_rank()`, the EOT-vs-long-hold diagnostic that drove the feature design. |
| `pipeline.py` | VAP vs prosodic vs blend, with the VAP frame model retrained per fold so it never sees a test turn's audio. `RUNLOG.md` rows 8–10. |
| `gen_report.py` | Out-of-fold predictions, the five figures, `artifacts/results.json`. |
| `make_summary.py` | Builds `SUMMARY.html` from `results.json` + `figs.json`. `SUMMARY.html` is generated, never hand-edited, so its numbers cannot drift from the measurements. |

## Reproduce

```bash
python predict.py --data_dir <folder> --out predictions.csv   # inference only
python train_final.py                                          # refit + save model
python src/experiments/cv.py                                   # the honest numbers
python src/experiments/gen_report.py                           # figures + results.json
python src/experiments/make_summary.py                         # rebuild SUMMARY.html
```

`artifacts/*.parquet` and `*.npy` are gitignored caches; every script rebuilds
them from audio when missing.

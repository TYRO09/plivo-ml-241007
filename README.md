# plivo-ml-241007 — End-of-Turn Detection

Plivo AI/ML internship assignment, STT track. Predicts, for every pause in a
user turn, the probability `p_eot` that the turn is over — using only audio
from before the pause.

**Held-out result** (5-fold `GroupKFold` by `turn_id`, 3 shuffles):

| | English | Hindi |
|---|---|---|
| silence-only baseline (given) | 1600 ms | 850 ms |
| **this model** | **1142 ms** | **835 ms** |

Metric: mean response delay at ≤5% interrupted turns, from the official
`score.py`. Lower is better.

> **On the two provided folders the shipped model reports 264 ms / 266 ms.**
> That is **in-sample** — those 200 turns are its training data — and it is not
> a generalisation estimate. The honest numbers are the held-out ones above.
> `predictions_*_heldout.csv` are the out-of-fold predictions that produce them.
> See the warning at the top of `SUMMARY.html`.

## Run it

```bash
pip install numpy scipy scikit-learn pandas soundfile matplotlib pyarrow
python predict.py --data_dir <folder> --out predictions.csv
python score.py   --data_dir <folder> --pred predictions.csv
```

`predict.py` loads the saved model from `artifacts/model.joblib` and never
refits. It needs `<folder>/labels.csv` (`turn_id, audio_file, pause_index,
pause_start, pause_end, label`) and the wavs it points at; the `label` column
is not required and is never read by the model.

## Deliverables

| File | |
|---|---|
| `SUMMARY.html` | full write-up: solution, results, graphs, human-vs-agent split, why it beats a silence timer |
| `predict.py` | the required entry point |
| `predictions_english.csv`, `predictions_hindi.csv` | required, from the shipped model |
| `predictions_english_heldout.csv`, `predictions_hindi_heldout.csv` | the same pauses scored out of fold — the honest predictions |
| `RUNLOG.md` | every scoring run, what changed, why |
| `NOTES.md` | signals used, failure modes, next steps |

## Code

| | |
|---|---|
| `src/dsp.py` | all signal processing, written from scratch: framing, energy, octave-protected autocorrelation F0, mel filterbank, DCT cepstra, spectral statistics |
| `src/features_eot.py` | the 256 causal features. `causal_slice()` is the single point where audio enters — it returns `x[:int(pause_start*sr)]`, so no feature can see the future |
| `src/dataset.py` | feature table + within-turn relative features |
| `src/model.py` | HistGradientBoosting ×5 seeds ⊕ L2 logistic, rank-averaged; duration-risk sample weights |
| `src/metric.py` | `score.py`'s policy sweep in-process, for model selection on the real metric |
| `src/cv.py`, `src/exp.py` | held-out evaluation and the ablations in `RUNLOG.md` |
| `src/vap.py`, `src/vap_train.py` | Voice Activity Projection objective (Ekstedt & Skantze 2022) reimplemented from scratch — measured, documented, **not shipped**; see §7 of `SUMMARY.html` |
| `src/gen_report.py`, `src/make_summary.py` | regenerate every figure and number |
| `train_final.py` | retrain and save `artifacts/model.joblib` |

## Rules compliance

- Laptop CPU only; no GPU, no cloud training.
- Libraries: numpy, scipy, scikit-learn, pandas, soundfile, matplotlib. No
  pretrained models, no downloaded weights, no external data. The VAP work is a
  from-scratch reimplementation of the *objective* described in the paper — no
  code, weights or encoder from that repository is used.
- Causality: features for a pause use only `[0, pause_start)`. The current
  row's `pause_end`, the waveform length, and the turn's total pause count are
  never read. Pause duration and the VAP targets come from future audio and are
  used **only as training weights / labels**, exactly like the `label` column;
  `predict.py` computes neither.

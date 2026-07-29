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

## Layout

```
plivo-ml-241007/
├── SUMMARY.html               # the write-up: solution, results, graphs  [deliverable 1]
├── predict.py                 # --data_dir / --out, loads the saved model [deliverable 2]
├── predictions_english.csv    # scored with the shipped model             [deliverable 3]
├── predictions_hindi.csv      #   "
├── predictions_*_heldout.csv  # the same pauses scored OUT OF FOLD (honest)
├── RUNLOG.md                  # every scoring run, what changed, why      [deliverable 4]
├── NOTES.md                   # signals, failure modes, next steps        [deliverable 5]
├── README.md                  # this file
├── requirements.txt
├── score.py                   # the official scorer, unmodified
├── train_final.py             # refit on all data, save artifacts/model.joblib
├── artifacts/
│   ├── model.joblib           # what predict.py loads
│   ├── results.json           # every number in SUMMARY.html
│   ├── figs.json              # the embedded figures
│   └── train_meta.json
└── src/
    ├── README.md              # code map — start here
    ├── dsp.py                 # signal processing, from scratch
    ├── features_eot.py        # the 256 causal features; causal_slice() is the audit point
    ├── dataset.py             # pause table + within-turn relative features
    ├── model.py               # the blend + duration-risk weights
    ├── metric.py              # score.py's sweep, in-process, for model selection
    ├── vap.py, vap_train.py   # VAP objective, from scratch — measured, NOT shipped
    └── experiments/           # how the model was chosen; not needed to run it
        ├── cv.py              # held-out evaluation + cross-language transfer
        ├── exp.py             # the ablations in RUNLOG rows 6-7
        ├── pipeline.py        # VAP vs prosodic vs blend (RUNLOG rows 8-10)
        ├── gen_report.py      # out-of-fold predictions, figures, results.json
        └── make_summary.py    # builds SUMMARY.html from results.json
```

`src/` is what inference needs; `src/experiments/` is what chose the model. See
[src/README.md](src/README.md) for what each file does and the order to read
them in. `SUMMARY.html` is generated from measured results, never hand-edited,
so its numbers cannot drift.

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

# RUNLOG

Every scoring run, in order, with what changed and why. All numbers are
**mean response delay (ms) at ≤5% interrupted turns** from `score.py`
(lower is better). `AUC` is the scorer's diagnostic AUC.

**Two kinds of number appear below — read the labels.**

- **held-out (CV)** = 5-fold `GroupKFold` grouped by `turn_id`, so no turn is
  ever in both train and test. **This is the honest estimate of what the
  hidden test set will show.**
- **in-sample** = the shipped model scored on the folders it was trained on.
  It reads ~265 ms and it is *meaningless as a generalisation estimate* — the
  model has seen those turns. It is reported only because `predict.py` +
  `score.py` on the provided folders produces it, and I would rather flag that
  than let it look like a real result.

Model selection was done **only** on held-out CV numbers, and on the official
metric (not accuracy, not AUC) — `src/metric.py` re-implements `score.py`'s
policy sweep in-process and is verified to agree with it.

| # | change | English | Hindi | note |
|---|--------|---------|-------|------|
| 0 | `baseline.py` (p_eot = 1 for every pause), official scorer | **1600** | **850** | Reproduces the ~1600 ms reference. Hindi is already 850 ms because 83% of Hindi hold pauses are ≤0.5 s, so a plain silence timer survives there — the English number is the one the handout quotes. |
| 1 | v1: 153 causal features (terminal prosody, F0 slope, energy decay, rhythm, spectral, turn-context, within-turn relative), HGB+logistic blend | 1224 | 850 | AUC 0.658 / 0.734. Beats baseline on English by 376 ms. Hindi unmoved: the scorer's best operating point is still "fire always, wait 850 ms". |
| 2 | v2: **duration-risk sample weights** — up-weight long holds, down-weight <300 ms breath pauses | 1216 | 819 | Motivated by reading the scorer: a hold can only cause a false cutoff if `delay < pause_dur`, so short holds are nearly free and the real task is *EOT vs long hold*. Pause duration is used as a training **weight** only, never as a feature. First movement on Hindi. |
| 3 | v3: + hesitation/elongation features (spectral-flux "frozen vowel" detector), cepstral-mean-normalised MFCCs, pitch-declination residual | 1117 | 850 | English −99 ms: English holds are marked by *elongation* ("annnd…", "soooo…"), not by pitch. CMN was added so the cepstra describe the final phone rather than the handset — needed because the hidden set is a different recording pool. |
| 4 | Diagnostic: per-feature AUC restricted to **EOT vs holds >0.35 s** | — | — | Hindi's best single feature is `rel_p_end_below_ref_dmean` (AUC 0.767) — final pitch relative to *this speaker's earlier pauses in the same turn*. English's best is `m_specflux_400` (elongation). Confirms the two languages end turns differently and validated the within-turn-relative idea. |
| 5 | v4: + 6×100 ms "acoustic shape of the final word" slices (CMN cepstra + speaker-relative pitch + relative energy per slice) | 1180 | 858 | 256 features. Intended as an ASR-free proxy for turn-final words ("thank you", "bas") vs hold-final words ("and", "aur"). Within noise on English, slight lift in Hindi AUC (0.756). Kept: the trees ignore it where it is useless and the hidden set is Hindi-heavy. |
| 6 | Ablation: drop short holds from training entirely (`hold_min_dur` 0.25 / 0.35) | 1144 / 1134 | 850 / 850 | **Rejected.** Weighting them down helps, removing them hurts — 496 rows is too little data to throw any away. |
| 7 | Ablation: feature selection to top-25 / 45 / 70 by restricted AUC | 1090–1200 | 797–858 | **Rejected.** No consistent gain over using all features; the regularised HGB (`max_features=0.55`, 7 leaves) handles the wide matrix better than my hand-picked subsets. |
| 8 | **VAP** (Ekstedt & Skantze 2022, arXiv:2205.09812) reimplemented from scratch: predict the discretised 2 s future voice-activity window (bins 200/400/600/800 ms, 50% threshold, joint/"Discrete" state), self-supervised from VAD. Zero-shot `p_eot = P(no activity in 2 s)`. | 1484 (zero-shot alone) | 858 | Trained with **zero** EOT labels. In-sample it looked excellent (AUC 0.754) but out-of-fold it collapsed to AUC 0.527 — it was memorising turns. |
| 9 | VAP fix: one sample per *silence event* instead of per frame (consecutive frames inside one pause are ~identical), drop in-speech frames (trivial target), 3-class target | 1412 (alone) | 850 | AUC 0.565 / 0.681 — better, still below the supervised model. Diagnosis: on 36 min of **mono** audio the objective yields only ~1,600 distinct silence events, not the ~100k independent examples the frame count suggests. The paper's leverage comes from thousands of hours of *two-channel* dialogue, where the other speaker's activity is most of the signal. Here there is no second channel at all. |
| 10 | VAP blended into the final score by rank-average (w_vap = 0.2–0.75) | 1268 @ w=0.25 | 850 @ w=0.25 | **Not shipped.** It lifts Hindi *AUC* (0.765 → 0.779) but never the graded delay, and it costs English. Shipping a component that loses on held-out CV to look sophisticated would be the wrong call. Kept in the repo as `src/vap.py` + `src/vap_train.py` with this honest negative result. |
| 11 | **Shipped model**: config #5 features, moderate risk weights, HGB(×5 seeds)+logistic rank-blend, trained on English+Hindi pooled. Held-out numbers averaged over 3 CV shuffles (the single-shuffle numbers above are noisier: ±60 ms) | **1142 held-out** | **835 held-out** | See `SUMMARY.html`. Cross-language transfer, re-measured on the final 256-feature set (the earlier figures in this log came from the v1/v2 feature sets): **en→hi 850 ms (AUC 0.668), hi→en 1313 ms (AUC 0.666)**. Training on a single language transfers weakly — Hindi lands exactly on its baseline — so pooling both languages is doing real work here, not just adding rows. |
| 12 | Shipped model, `predict.py` → `score.py` on the two **provided** folders | 264 *(in-sample)* | 266 *(in-sample)* | **Not a real score.** AUC 0.988 because these 200 turns are in the training set. Shipped as `predictions_english.csv` / `predictions_hindi.csv` because that is what the deliverable asks for; the honest equivalents are `predictions_*_heldout.csv` (out-of-fold), which score 1142 / 835. Expect the hidden test set to land near the held-out numbers. |

| 13 | Ablation: **sequence model instead of hand-designed features** — a small causal transformer (d_model=32, 2 layers, ~20k params, positional encoding, causal attention mask) over the last 2 s as a 197×17 frame matrix (energy, F0, voice activity, ZCR, 12 MFCCs), pooled English+Hindi, 30 epochs, BCE | **1411 held-out** | **843 held-out** | **Rejected — and the most instructive run in this log.** Scored the way it is tempting to score it (train on all 200 turns, predict the same 200) it reports **115 ms / 100 ms at AUC 0.996 / 0.999**, which looks like a solved problem. Under 5-fold GroupKFold by `turn_id` the *identical* code gives AUC **0.543 / 0.637** — English is near chance. A 20k-parameter transformer trained 30 epochs on 496 sequences memorises the turns; the in-sample number is measuring recall of the training set, not end-of-turn detection. This is why every number in this log is out-of-fold, and it is the concrete reason the shipped model is gradient-boosted trees on ~256 designed features rather than a sequence model: at this data scale the features are the regularisation. |

| 14 | **Stability check**: repeated the whole held-out evaluation over 8 independent fold shuffles instead of 3 | **1184 ± 49** | **835 ± 16** | Run to run, English moves ±49 ms and Hindi ±16 ms, so single-split numbers in the rows above are worth ±1 sd and several of the small "improvements" (1224 → 1216 → 1180) are inside noise. English beats baseline in **8/8** shuffles (+416 ms ≈ 8σ). Hindi beats it in only **5/8** (+15 ms ≈ 0.9σ) — **not a reliable gain**, and the headline was corrected to say so. Per-shuffle numbers in `artifacts/stability.json`. |

## Verification

- `src/metric.py` reproduces `score.py` exactly (same sweep grid, same
  timeout, same turn-level cutoff accounting); all decisions above were made
  with it and the final numbers were confirmed by running the official
  `score.py` unmodified.
- Causality is enforced at one choke point, `features_eot.causal_slice()`,
  which returns `x[:int(pause_start*sr)]`. No feature function ever receives
  the full waveform, so no feature can read `pause_end`, the file length, or
  any audio at/after the pause. The frame-level VAP extractor uses
  exclusively backward-looking windows and is queried at the last frame that
  *ends* at or before `pause_start`.

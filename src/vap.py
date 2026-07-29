"""Voice Activity Projection (VAP), reimplemented from scratch for this task.

Idea from Ekstedt & Skantze, "Voice Activity Projection: Self-supervised
Learning of Turn-taking Events" (Interspeech 2022), arXiv:2205.09812, and
their reference implementation github.com/ErikEkstedt/VoiceActivityProjection.

NOTHING is downloaded or reused from that repo: no weights, no code, no
pretrained encoder (the assignment forbids it).  What is borrowed is the
TRAINING OBJECTIVE, and that is the whole point:

  Instead of asking "is this pause an end-of-turn?" (496 labelled examples in
  this handout), ask "what will the user's voice activity look like over the
  next 2 s?" (~100k examples, because the label is derived from the audio
  itself by VAD -- self-supervised, no annotation needed).

Following the paper we discretise the 2 s projection window into 4 bins of
200/400/600/800 ms with a 50% activity threshold, and we model the window
JOINTLY (their "Discrete" variant, which they show beats independent per-bin
sigmoids) rather than as independent bins.  This handout has only the user
channel, so the 2-speaker/256-state codebook collapses to the 4-bit window of
one speaker; the realisable states reduce to "which is the first bin in which
the user speaks again", i.e. 5 classes:

    class 0 : resumes within 200 ms
    class 1 : resumes in 200-600 ms
    class 2 : resumes in 600-1200 ms
    class 3 : resumes in 1200-2000 ms
    class 4 : does NOT resume within 2 s      <-- end of turn

Zero-shot end-of-turn (their zero-shot SHIFT/HOLD mapping, mono version):

    p_eot = P(class 4)

CAUSALITY: every feature at frame t is an aggregate over frames <= t only
(see `frame_features`, which uses backward-looking windows exclusively), and
we only ever query the last frame that ends at or before `pause_start`
(`frame_index_for`).  The VAP TARGETS are read from future audio, but a target
is a training label, exactly like `label` in labels.csv -- it is never an
input, and predict.py computes no targets at all.
"""
import numpy as np

from dsp import (dct_mfcc, energy_db, f0_contour, hz_to_semitones,
                 log_mel_spectrogram, runs_of, speech_mask)

HOP_MS = 10.0                  # internal analysis hop
FRAME_MS = 25.0
STEP = 2                       # keep every 2nd frame -> 50 Hz, as in the paper
BIN_TIMES = [0.2, 0.4, 0.6, 0.8]
FRAME_HZ = 50
THRESHOLD_RATIO = 0.5
N_CLASSES = 5


# --------------------------------------------------------------- targets -----
def frame_vad(x, sr):
    """Self-supervised voice-activity labels at 50 Hz over the whole file."""
    db = energy_db(x, sr)
    m = speech_mask(db)
    return m[::STEP].astype(np.float32)


def projection_classes(vad):
    """Per-frame VAP target: index of the first active bin, else 4.

    Bin b covers frames (cumulative) of BIN_TIMES; a bin is active when the
    mean VAD inside it >= THRESHOLD_RATIO, exactly as in the paper.  Frames
    past the end of the file are treated as silence: the turn is over, so the
    user is not speaking.
    """
    bin_frames = [int(round(t * FRAME_HZ)) for t in BIN_TIMES]
    horizon = sum(bin_frames)
    v = np.concatenate([vad, np.zeros(horizon + 1, dtype=vad.dtype)])
    n = len(vad)
    # window starts at t+1 (strictly the future)
    idx = np.arange(n)[:, None] + 1 + np.arange(horizon)[None, :]
    win = v[idx]                                   # (n, horizon)
    bins = []
    s = 0
    for b in bin_frames:
        bins.append(win[:, s:s + b].mean(axis=1) >= THRESHOLD_RATIO)
        s += b
    bins = np.stack(bins, axis=1)                  # (n, 4) the VAP window
    first = np.where(bins.any(axis=1), bins.argmax(axis=1), N_CLASSES - 1)
    return first.astype(np.int64), bins


# -------------------------------------------------------------- features -----
def _causal_windows(a, k):
    """Rows = frames, cols = the k values ending at that frame (front-padded).

    Column k-1 is frame t itself; column 0 is frame t-k+1.  Nothing after t.
    """
    a = np.asarray(a, dtype=np.float64)
    pad = np.concatenate([np.full(k - 1, a[0] if len(a) else 0.0), a])
    return np.lib.stride_tricks.sliding_window_view(pad, k)


def _cmean(a, k):
    return _causal_windows(a, k).mean(axis=1)


def _cslope(a, k):
    w = _causal_windows(a, k)
    t = np.arange(k) * (HOP_MS / 1000.0)
    t = t - t.mean()
    return (w - w.mean(axis=1, keepdims=True)) @ t / (np.sum(t ** 2) + 1e-12)


def _expanding_median_proxy(a, mask):
    """Causal running mean of `a` over frames where mask is True (speech)."""
    a = np.where(mask, a, 0.0)
    cs = np.cumsum(a)
    cn = np.cumsum(mask.astype(np.float64))
    return cs / np.maximum(cn, 1.0)


def _silence_run_length(sp):
    """Frames since the last speech frame (0 while speaking)."""
    out = np.zeros(len(sp), dtype=np.float64)
    c = 0
    for i, v in enumerate(sp):
        c = 0 if v else c + 1
        out[i] = c
    return out


def _speech_run_length(sp):
    out = np.zeros(len(sp), dtype=np.float64)
    c = 0
    for i, v in enumerate(sp):
        c = c + 1 if v else 0
        out[i] = c
    return out


FEATURE_NAMES = None


def frame_features(x, sr):
    """Causal feature matrix at 50 Hz.  Row t uses only frames <= t."""
    global FEATURE_NAMES
    db = energy_db(x, sr)
    f0, strength = f0_contour(x, sr)
    n = min(len(db), len(f0))
    if n < 12:
        return np.zeros((0, 1), dtype=np.float32)
    db, f0, strength = db[:n], f0[:n], strength[:n]
    sp = speech_mask(db)[:n]
    voiced = (f0 > 0)

    lm = log_mel_spectrogram(x, sr)[:n]
    mf = dct_mfcc(lm, n_mfcc=13)
    flux = np.concatenate([[0.0], np.linalg.norm(np.diff(mf[:, 1:9], axis=0), axis=1)])[:n]

    # speaker-relative references, computed causally (expanding over the past)
    e_ref = _expanding_median_proxy(db, sp)
    f0_run = _expanding_median_proxy(f0, voiced)
    f0_run = np.where(f0_run > 40, f0_run, 150.0)
    st = np.where(voiced, hz_to_semitones(f0, 1.0), np.nan) - \
        12.0 * np.log2(np.maximum(f0_run, 1.0))
    st_filled = np.where(np.isfinite(st), st, 0.0)

    sil_len = _silence_run_length(sp) * (HOP_MS / 1000.0)
    spe_len = _speech_run_length(sp) * (HOP_MS / 1000.0)
    flux_ref = _expanding_median_proxy(flux, sp) + 1e-6

    feats, names = [], []

    def add(name, v):
        feats.append(np.asarray(v, dtype=np.float64))
        names.append(name)

    # --- silence hazard (what a plain VAD timer would use) ---------------
    add("sil_len", sil_len)
    add("spe_len", spe_len)
    add("in_speech", sp.astype(float))
    # --- energy ---------------------------------------------------------
    add("e_rel", db - e_ref)
    for k in (10, 30, 60):
        add(f"e_mean_{k}", _cmean(db, k) - e_ref)
    for k in (20, 40):
        add(f"e_slope_{k}", _cslope(db, k))
    add("e_dyn", _causal_windows(db, 100).max(axis=1) - db)
    # --- pitch ----------------------------------------------------------
    add("st_now", st_filled)
    add("st_mean_20", _cmean(st_filled, 20))
    add("st_slope_20", _cslope(st_filled, 20))
    add("st_slope_40", _cslope(st_filled, 40))
    add("st_min_40", _causal_windows(st_filled, 40).min(axis=1))
    add("st_max_40", _causal_windows(st_filled, 40).max(axis=1))
    add("voiced_now", voiced.astype(float))
    add("voiced_frac_50", _cmean(voiced.astype(float), 50))
    add("voiced_frac_150", _cmean(voiced.astype(float), 150))
    add("strength_20", _cmean(strength, 20))
    # --- articulation / hesitation --------------------------------------
    add("flux_15", _cmean(flux, 15) / flux_ref)
    add("flux_40", _cmean(flux, 40) / flux_ref)
    add("flux_slope", _cslope(flux, 40))
    # --- final-phone cepstra (channel-normalised) -----------------------
    mf_ref = np.array([_expanding_median_proxy(mf[:, i], sp) for i in range(9)]).T
    for i in range(1, 9):
        add(f"c{i}", _cmean(mf[:, i], 15) - mf_ref[:, i])
        add(f"dc{i}", _cmean(mf[:, i], 10) - _cmean(mf[:, i], 40))
    # --- turn-so-far context --------------------------------------------
    cum_sp = np.cumsum(sp.astype(float)) * (HOP_MS / 1000.0)
    t_axis = np.arange(n) * (HOP_MS / 1000.0)
    onset = np.flatnonzero(sp)
    onset_t = onset[0] * (HOP_MS / 1000.0) if len(onset) else 0.0
    add("t_since_onset", np.maximum(t_axis - onset_t, 0.0))
    add("speech_total", cum_sp)
    add("speech_frac", cum_sp / np.maximum(t_axis - onset_t, 0.2))
    gaps = np.zeros(n)
    for s, e in runs_of(sp, False):
        if (e - s) >= 10 and e < n:
            gaps[e:] += 1
    add("n_gaps", gaps)

    F = np.stack(feats, axis=1)[::STEP]
    if FEATURE_NAMES is None:
        FEATURE_NAMES = names
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def feature_names():
    """Names of the columns of `frame_features`, in order.

    Recomputed locally from a synthetic signal so the names are available in
    the parent process even when extraction ran inside worker processes.
    """
    global FEATURE_NAMES
    if FEATURE_NAMES is None:
        sr = 16000
        t = np.arange(sr * 3) / sr
        x = (0.2 * np.sin(2 * np.pi * 140 * t) *
             (np.sin(2 * np.pi * 3 * t) > 0)).astype(np.float32)
        frame_features(x, sr)
    return FEATURE_NAMES


def frame_index_for(pause_start):
    """Last 50 Hz frame that ends at or before `pause_start` (never after)."""
    t = int(np.floor((float(pause_start) * 1000.0 - FRAME_MS) / (HOP_MS * STEP)))
    return max(t, 0)


# ------------------------------------------------------- zero-shot readout ---
def zero_shot(proba, n_classes=3):
    """Map the projection distribution to interpretable turn-taking scalars.

    proba: (n, 5) over the classes documented at the top of this file.
    """
    proba = np.asarray(proba, dtype=float)
    if proba.shape[1] < n_classes:                 # class missing in training
        pad = np.zeros((len(proba), n_classes - proba.shape[1]))
        proba = np.concatenate([proba, pad], axis=1)
    p_eot = proba[:, -1]
    return {
        "vap_p_eot": p_eot,                        # no activity within 2 s
        "vap_p_quiet_600": proba[:, 1:].sum(axis=1),
        "vap_p_quiet_1200": proba[:, -1],
        "vap_p_resume_fast": proba[:, 0],          # resumes soon
        "vap_expected_bin": proba @ np.arange(proba.shape[1]),
        "vap_logit_eot": np.log(np.clip(p_eot, 1e-6, 1 - 1e-6) /
                                np.clip(1 - p_eot, 1e-6, 1 - 1e-6)),
    }

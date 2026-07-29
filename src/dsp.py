"""Low-level DSP utilities (vectorised).

Nothing in this file knows about pauses, labels or files: it only ever sees a
1-D waveform that has ALREADY been truncated by `features_eot.causal_slice`.
That is the single choke point that enforces the causality rule.
"""
import numpy as np

FRAME_MS = 25.0
HOP_MS = 10.0
F0_FRAME_MS = 40.0
EPS = 1e-12


# ----------------------------------------------------------------- framing ---
def frame_signal(x, sr, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    """(n_frames, frame_len) view-like matrix of overlapping frames."""
    fl = int(round(sr * frame_ms / 1000.0))
    hp = int(round(sr * hop_ms / 1000.0))
    if len(x) < fl:
        return np.empty((0, fl), dtype=np.float32), fl, hp
    n = 1 + (len(x) - fl) // hp
    idx = np.arange(fl)[None, :] + hp * np.arange(n)[:, None]
    return x[idx], fl, hp


def frame_times(n_frames, sr, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    """Centre time (s) of each frame."""
    return (np.arange(n_frames) * hop_ms + frame_ms / 2.0) / 1000.0


# ----------------------------------------------------------------- energy ----
def energy_db(x, sr, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    fr, _, _ = frame_signal(x, sr, frame_ms, hop_ms)
    if len(fr) == 0:
        return np.zeros(0, dtype=np.float32)
    rms = np.sqrt(np.mean(fr.astype(np.float64) ** 2, axis=1) + EPS)
    return (20.0 * np.log10(rms + EPS)).astype(np.float32)


def speech_mask(db, min_speech_frames=5, min_sil_frames=10):
    """Adaptive speech/silence decision on a dB energy contour.

    Threshold sits between the noise floor and the speech peak of THIS
    segment, so it adapts to per-call gain and channel noise (important:
    the hidden set is a different recording pool).
    """
    if len(db) == 0:
        return np.zeros(0, dtype=bool)
    floor = np.percentile(db, 10.0)
    peak = np.percentile(db, 95.0)
    if peak - floor < 6.0:                     # essentially flat -> all speech
        return np.ones(len(db), dtype=bool)
    thr = floor + 0.42 * (peak - floor)
    m = db > thr
    m = _despeckle(m, min_speech_frames, min_sil_frames)
    return m


def _despeckle(m, min_true, min_false):
    """Drop speech blips, then bridge silence blips."""
    m = m.copy()
    for target, min_len in ((True, min_true), (False, min_false)):
        for s, e in runs_of(m, target):
            if e - s < min_len:
                m[s:e] = not target
    return m


def runs_of(mask, value=True):
    """[(start, end)) index pairs of maximal runs equal to `value`."""
    if len(mask) == 0:
        return []
    m = (mask == value).astype(np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


# -------------------------------------------------------------------- F0 -----
def f0_contour(x, sr, frame_ms=F0_FRAME_MS, hop_ms=HOP_MS,
               fmin=60.0, fmax=400.0, voicing_thresh=0.32):
    """Autocorrelation F0, fully vectorised, with octave-jump protection.

    Returns (f0_hz, strength) per frame; f0 = 0 where unvoiced.
    Improvements over the starter's per-frame tracker: FFT autocorrelation
    (100x faster), parabolic peak interpolation, sub-harmonic preference
    (take the LOWEST lag within 88% of the best peak -> kills the octave
    halving that makes creaky turn-final voice look like a pitch reset),
    and a median filter to remove isolated jumps.
    """
    fr, fl, _ = frame_signal(x, sr, frame_ms, hop_ms)
    if len(fr) == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    fr = fr.astype(np.float64)
    fr = fr - fr.mean(axis=1, keepdims=True)
    win = np.hanning(fl)
    fr = fr * win

    nfft = 1 << int(np.ceil(np.log2(2 * fl)))
    spec = np.fft.rfft(fr, n=nfft, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, n=nfft, axis=1)[:, :fl]
    zero = ac[:, :1].copy()
    zero[zero <= 0] = EPS
    ac = ac / zero

    lo = max(2, int(sr / fmax))
    hi = min(int(sr / fmin), fl - 2)
    if hi <= lo + 1:
        n = len(fr)
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)

    band = ac[:, lo:hi]
    best = band.max(axis=1)
    # lowest lag whose peak is within 12% of the best -> avoids octave halving
    ok = band >= (0.88 * best[:, None])
    lag = lo + np.argmax(ok, axis=1)

    # parabolic interpolation around the chosen lag
    l0 = np.clip(lag, 1, fl - 2)
    y0 = ac[np.arange(len(ac)), l0 - 1]
    y1 = ac[np.arange(len(ac)), l0]
    y2 = ac[np.arange(len(ac)), l0 + 1]
    denom = (y0 - 2 * y1 + y2)
    shift = np.where(np.abs(denom) > EPS, 0.5 * (y0 - y2) / (denom + EPS), 0.0)
    shift = np.clip(shift, -0.5, 0.5)
    f0 = sr / (l0 + shift)

    strength = y1
    amp = np.max(np.abs(fr), axis=1)
    voiced = (strength >= voicing_thresh) & (amp > 1e-4) & (f0 >= fmin) & (f0 <= fmax)
    f0 = np.where(voiced, f0, 0.0)
    f0 = _median_filter_voiced(f0, k=5)
    return f0.astype(np.float32), (strength * voiced).astype(np.float32)


def _median_filter_voiced(f0, k=5):
    """Median filter that ignores unvoiced (0) neighbours."""
    n = len(f0)
    if n < k:
        return f0
    out = f0.copy()
    h = k // 2
    for i in range(n):
        if f0[i] <= 0:
            continue
        w = f0[max(0, i - h):min(n, i + h + 1)]
        w = w[w > 0]
        if len(w):
            out[i] = np.median(w)
    return out


def hz_to_semitones(f0, ref):
    """Semitone distance from a reference Hz (speaker-relative pitch)."""
    ref = max(ref, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 12.0 * np.log2(np.maximum(f0, EPS) / ref)


# --------------------------------------------------------------- spectral ----
def log_mel_spectrogram(x, sr, n_mels=26, frame_ms=25.0, hop_ms=10.0,
                        fmin=50.0, fmax=7600.0):
    """Log-mel with a locally built filterbank (no external weights)."""
    fr, fl, _ = frame_signal(x, sr, frame_ms, hop_ms)
    if len(fr) == 0:
        return np.zeros((0, n_mels), dtype=np.float32)
    fr = fr.astype(np.float64) * np.hanning(fl)
    nfft = 1 << int(np.ceil(np.log2(fl)))
    power = np.abs(np.fft.rfft(fr, n=nfft, axis=1)) ** 2
    fb = _mel_filterbank(sr, nfft, n_mels, fmin, fmax)
    mel = power @ fb.T
    return np.log(mel + 1e-10).astype(np.float32)


_FB_CACHE = {}


def _mel_filterbank(sr, nfft, n_mels, fmin, fmax):
    key = (sr, nfft, n_mels, fmin, fmax)
    if key in _FB_CACHE:
        return _FB_CACHE[key]

    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(min(fmax, sr / 2)), n_mels + 2))
    bins = np.floor((nfft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, nfft // 2 + 1))
    for i in range(n_mels):
        a, b, c = bins[i], bins[i + 1], bins[i + 2]
        if b == a:
            b = a + 1
        if c == b:
            c = b + 1
        c = min(c, fb.shape[1] - 1)
        b = min(b, c)
        if b > a:
            fb[i, a:b] = (np.arange(a, b) - a) / (b - a)
        if c > b:
            fb[i, b:c] = (c - np.arange(b, c)) / (c - b)
    _FB_CACHE[key] = fb
    return fb


def dct_mfcc(logmel, n_mfcc=13):
    """DCT-II of the log-mel frames -> MFCCs (own implementation)."""
    if len(logmel) == 0:
        return np.zeros((0, n_mfcc), dtype=np.float32)
    n = logmel.shape[1]
    k = np.arange(n_mfcc)[:, None]
    m = np.arange(n)[None, :]
    basis = np.cos(np.pi * k * (2 * m + 1) / (2 * n))
    basis[0] *= 1 / np.sqrt(2)
    return (logmel @ basis.T * np.sqrt(2.0 / n)).astype(np.float32)


def spectral_stats(x, sr, frame_ms=25.0, hop_ms=10.0):
    """Per-frame centroid (Hz), rolloff85 (Hz), flatness, tilt, hf/lf ratio."""
    fr, fl, _ = frame_signal(x, sr, frame_ms, hop_ms)
    if len(fr) == 0:
        z = np.zeros((0,), dtype=np.float32)
        return dict(centroid=z, rolloff=z, flatness=z, tilt=z, hflf=z)
    fr = fr.astype(np.float64) * np.hanning(fl)
    nfft = 1 << int(np.ceil(np.log2(fl)))
    mag = np.abs(np.fft.rfft(fr, n=nfft, axis=1)) + EPS
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    tot = mag.sum(axis=1, keepdims=True)
    centroid = (mag * freqs).sum(axis=1) / tot[:, 0]
    csum = np.cumsum(mag, axis=1) / tot
    rolloff = freqs[np.argmax(csum >= 0.85, axis=1)]
    logm = np.log(mag)
    flatness = np.exp(logm.mean(axis=1)) / (mag.mean(axis=1) + EPS)
    # spectral tilt: slope of log-magnitude vs log-frequency (dB/decade-ish)
    lf = np.log(freqs + 1.0)
    lf = lf - lf.mean()
    tilt = (logm * lf).sum(axis=1) / (np.sum(lf ** 2) + EPS)
    lo = (freqs >= 80) & (freqs < 1000)
    hi = (freqs >= 2000) & (freqs < 8000)
    hflf = 10.0 * np.log10((mag[:, hi] ** 2).mean(axis=1) + EPS) - \
        10.0 * np.log10((mag[:, lo] ** 2).mean(axis=1) + EPS)
    return dict(centroid=centroid.astype(np.float32),
                rolloff=rolloff.astype(np.float32),
                flatness=flatness.astype(np.float32),
                tilt=tilt.astype(np.float32),
                hflf=hflf.astype(np.float32))


# ------------------------------------------------------------------ misc -----
def lin_slope(y, dt=HOP_MS / 1000.0):
    """Least-squares slope of y w.r.t. time, in units/second."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 3:
        return 0.0
    t = np.arange(n) * dt
    t = t - t.mean()
    return float((t * (y - y.mean())).sum() / (np.sum(t ** 2) + EPS))


def safe(v, default=0.0):
    v = float(v) if np.isscalar(v) or np.ndim(v) == 0 else default
    if not np.isfinite(v):
        return default
    return v

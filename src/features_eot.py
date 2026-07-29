"""Causal feature extraction for end-of-turn detection.

=============================  CAUSALITY  ===================================
`causal_slice()` is the ONLY place audio is handed to the feature code, and it
returns `x[:int(pause_start*sr)]`.  Every feature below is computed from that
truncated array, so no feature can physically see a single sample at or after
`pause_start`.  `pause_end` of the current pause is never read, the waveform
length is never read, and the number of pauses in the turn is never read.
Context features use only PREVIOUS pause rows, whose `pause_end` is by
construction < the current `pause_start`.
=============================================================================
"""
import numpy as np
import soundfile as sf

from dsp import (HOP_MS, dct_mfcc, energy_db, f0_contour, hz_to_semitones,
                 lin_slope, log_mel_spectrogram, runs_of, safe, speech_mask,
                 spectral_stats)

HOP_S = HOP_MS / 1000.0


def load_wav(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def causal_slice(x, sr, pause_start):
    """THE causality choke point: audio from 0 up to (not incl.) pause_start."""
    n = int(round(float(pause_start) * sr))
    if n < 0:
        n = 0
    return x[:n]


# --------------------------------------------------------------- helpers -----
def _win(arr, anchor, ms):
    """Values of `arr` in the window of `ms` milliseconds ending at `anchor`."""
    k = max(1, int(round(ms / HOP_MS)))
    lo = max(0, anchor - k + 1)
    return arr[lo:anchor + 1]


def _runs_with_gap(mask, max_gap=2):
    """Runs of True, bridging gaps of <= max_gap frames (jitter tolerance)."""
    m = mask.copy()
    for s, e in runs_of(m, False):
        if e - s <= max_gap and s > 0 and e < len(m):
            m[s:e] = True
    return runs_of(m, True)


def _peak_rate(db, lo, hi):
    """Crude syllable rate: energy peaks per second inside [lo, hi) frames."""
    seg = db[lo:hi]
    if len(seg) < 5:
        return 0.0
    seg = np.convolve(seg, np.ones(3) / 3.0, mode="same")
    rng = seg.max() - seg.min()
    if rng < 3.0:
        return 0.0
    prom = max(3.0, 0.25 * rng)
    n = 0
    last = -10
    for i in range(1, len(seg) - 1):
        if seg[i] >= seg[i - 1] and seg[i] > seg[i + 1]:
            local_min = seg[max(0, i - 12):i + 1].min()
            if seg[i] - local_min >= prom and i - last >= 8:
                n += 1
                last = i
    return n / (len(seg) * HOP_S)


# ------------------------------------------------------------- the features --
def base_features(x_causal, sr, ctx):
    """Feature dict for one pause.  `x_causal` is already truncated."""
    f = {}

    # ---------------- context (past pause structure only) ----------------
    t_now = float(ctx["pause_start"])
    prev = ctx["prev_pauses"]                    # list of (start, end), all past
    prev_durs = np.array([e - s for s, e in prev], dtype=np.float64)
    prev_pause_total = float(prev_durs.sum()) if len(prev_durs) else 0.0
    speech_so_far = max(t_now - prev_pause_total, 0.05)

    f["ctx_pause_index"] = float(ctx["pause_index"])
    f["ctx_is_first_pause"] = 1.0 if ctx["pause_index"] == 0 else 0.0
    f["ctx_log_t_now"] = float(np.log1p(t_now))
    f["ctx_log_speech_so_far"] = float(np.log1p(speech_so_far))
    f["ctx_prev_pause_total"] = prev_pause_total
    f["ctx_prev_pause_mean"] = float(prev_durs.mean()) if len(prev_durs) else 0.0
    f["ctx_prev_pause_max"] = float(prev_durs.max()) if len(prev_durs) else 0.0
    f["ctx_prev_pause_last"] = float(prev_durs[-1]) if len(prev_durs) else 0.0
    f["ctx_since_prev_end"] = float(t_now - prev[-1][1]) if len(prev) else t_now
    f["ctx_pause_rate"] = ctx["pause_index"] / max(t_now, 0.5)
    f["ctx_pause_time_frac"] = prev_pause_total / max(t_now, 0.5)

    # ---------------- frame-level contours over the turn so far ----------
    db = energy_db(x_causal, sr)
    f0, strength = f0_contour(x_causal, sr)
    n = min(len(db), len(f0))
    if n < 8:                                    # almost no audio yet
        f["ac_valid"] = 0.0                      # rest stays missing -> NaN
        return f
    db, f0, strength = db[:n], f0[:n], strength[:n]

    sp = speech_mask(db)[:n]
    voiced = f0 > 0

    # Anchor at the LAST SPEECH FRAME, not at the raw cut point: pause_start is
    # annotated on a 100 ms grid, so the true speech offset can sit up to ~50 ms
    # earlier.  Anchoring on the observed offset makes every terminal feature
    # immune to that quantisation.
    sp_idx = np.flatnonzero(sp)
    anchor = int(sp_idx[-1]) if len(sp_idx) else n - 1
    f["ac_valid"] = 1.0
    f["ac_lead_silence"] = (n - 1 - anchor) * HOP_S

    # Many turns start with several seconds of dead air, so wall-clock
    # `pause_start` is a poor proxy for "how long has this person been
    # talking".  Measure from the observed speech onset instead.
    onset = int(sp_idx[0]) if len(sp_idx) else 0
    f["ac_onset_t"] = onset * HOP_S
    f["ac_t_since_onset"] = max((anchor - onset) * HOP_S, 0.0)
    f["ac_log_t_since_onset"] = float(np.log1p(f["ac_t_since_onset"]))
    f["ac_speech_total"] = float(sp.sum()) * HOP_S
    f["ac_speech_frac"] = float(sp[onset:anchor + 1].mean()) if anchor > onset else 1.0
    gaps = [(e - s) * HOP_S for s, e in runs_of(sp[onset:anchor + 1], False)]
    f["ac_n_gaps_100"] = float(sum(1 for g in gaps if g >= 0.10))
    f["ac_n_gaps_300"] = float(sum(1 for g in gaps if g >= 0.30))
    f["ac_gap_total"] = float(sum(gaps))

    # speaker references from the turn so far (self-normalisation)
    sp_db = db[sp] if sp.any() else db
    e_ref = float(np.median(sp_db))
    e_p90 = float(np.percentile(sp_db, 90))
    v_f0 = f0[voiced]
    f0_ref = float(np.median(v_f0)) if len(v_f0) >= 5 else 150.0
    f["ac_f0_ref"] = f0_ref
    f["ac_e_ref"] = e_ref
    f["ac_voiced_frac_all"] = float(voiced.mean())

    # ---------------- energy dynamics into the pause ----------------------
    for ms in (50, 150, 300):
        f[f"en_term_{ms}"] = float(np.mean(_win(db, anchor, ms))) - e_ref
    f["en_slope_200"] = lin_slope(_win(db, anchor, 200))
    f["en_slope_500"] = lin_slope(_win(db, anchor, 500))
    f["en_slope_1000"] = lin_slope(_win(db, anchor, 1000))
    w1 = _win(db, anchor, 1000)
    f["en_fade_depth"] = float(w1.max() - np.mean(_win(db, anchor, 150)))
    f["en_std_500"] = float(np.std(_win(db, anchor, 500)))
    f["en_pctile_term"] = float((sp_db < np.mean(_win(db, anchor, 150))).mean())
    f["en_dyn_range"] = e_p90 - e_ref

    # ---------------- pitch: the core prosodic cue -----------------------
    st = np.where(voiced, hz_to_semitones(f0, f0_ref), np.nan)
    vruns = [(s, e) for s, e in _runs_with_gap(voiced, max_gap=2) if e - s >= 3]
    # the final voiced stretch (must end within 250 ms of the anchor)
    last_vr = None
    for s, e in vruns:
        if e - 1 <= anchor + 3 and (anchor - (e - 1)) * HOP_S <= 0.25:
            last_vr = (s, e)
    f["p_has_final_voiced"] = 1.0 if last_vr else 0.0

    if last_vr:
        s, e = last_vr
        seg = st[s:e]
        seg = seg[np.isfinite(seg)]
        f["p_final_run_dur"] = (e - s) * HOP_S
        if len(seg) >= 3:
            f["p_end_st"] = float(np.mean(seg[-3:]))
            f["p_start_st"] = float(np.mean(seg[:3]))
            f["p_run_slope"] = lin_slope(seg)
            f["p_slope_200"] = lin_slope(seg[-20:]) if len(seg) >= 6 else lin_slope(seg)
            f["p_slope_400"] = lin_slope(seg[-40:]) if len(seg) >= 8 else lin_slope(seg)
            f["p_range"] = float(seg.max() - seg.min())
            f["p_min_st"] = float(seg.min())
            f["p_max_st"] = float(seg.max())
            f["p_end_minus_min"] = float(np.mean(seg[-3:]) - seg.min())
            f["p_end_minus_max"] = float(np.mean(seg[-3:]) - seg.max())
            f["p_argmax_pos"] = float(np.argmax(seg) / max(len(seg) - 1, 1))
            f["p_fall_frac"] = float(np.mean(np.diff(seg) < 0)) if len(seg) > 1 else 0.5
            # phrase-final low target: how deep below the speaker's own median
            f["p_end_below_ref"] = float(-np.mean(seg[-3:]))
        # creak / glottalisation: weak, very low-pitched final voicing
        stg = strength[s:e]
        f["p_strength_end"] = float(np.mean(stg[-5:])) if len(stg) else 0.0
        f["p_creak_frac"] = float(np.mean(seg[-30:] < -5.0)) if len(seg) >= 3 else 0.0
    # pitch of the final run vs the earlier runs of the same turn
    if len(vruns) >= 2 and last_vr:
        earlier = np.concatenate([st[s:e] for s, e in vruns if (s, e) != last_vr])
        earlier = earlier[np.isfinite(earlier)]
        cur = st[last_vr[0]:last_vr[1]]
        cur = cur[np.isfinite(cur)]
        if len(earlier) >= 3 and len(cur) >= 3:
            f["p_final_vs_earlier"] = float(np.mean(cur[-5:]) - np.mean(earlier))
            f["p_final_run_z"] = float((np.mean(cur[-5:]) - np.mean(earlier)) /
                                       (np.std(earlier) + 1e-3))
    f["p_voiced_frac_500"] = float(np.mean(voiced[max(0, anchor - 49):anchor + 1]))
    f["p_voiced_frac_1500"] = float(np.mean(voiced[max(0, anchor - 149):anchor + 1]))

    # ---------------- rhythm: final lengthening & slowing down -----------
    vdurs = np.array([(e - s) * HOP_S for s, e in vruns]) if vruns else np.zeros(0)
    if len(vdurs) >= 2:
        f["r_final_len_ratio"] = float(vdurs[-1] / (np.median(vdurs[:-1]) + 1e-3))
        f["r_vrun_dur_median"] = float(np.median(vdurs[:-1]))
    f["r_n_vruns_1500"] = float(sum(1 for s, e in vruns if (e - 1) >= anchor - 149))

    sruns = _runs_with_gap(sp, max_gap=9)
    last_sr_dur = 0.0
    for s, e in sruns:
        if s <= anchor < e or e - 1 <= anchor:
            last_sr_dur = (min(e, anchor + 1) - s) * HOP_S
    f["r_last_speech_run"] = last_sr_dur
    if len(sruns) >= 2:
        sd = np.array([(e - s) * HOP_S for s, e in sruns])
        f["r_speech_run_ratio"] = float(last_sr_dur / (np.median(sd[:-1]) + 1e-3))

    rate_recent = _peak_rate(db, max(0, anchor - 149), anchor + 1)
    rate_all = _peak_rate(db, 0, anchor + 1)
    f["r_rate_recent"] = rate_recent
    f["r_rate_all"] = rate_all
    f["r_rate_ratio"] = rate_recent / (rate_all + 1e-3)

    # ---------------- pitch declination residual --------------------------
    # Speakers drift downward across a phrase (declination).  Fit that drift on
    # the turn so far and ask how far BELOW its own trend the final pitch is:
    # a true phrase-final fall undershoots the trend, a continuation does not.
    vi = np.flatnonzero(voiced)
    if len(vi) >= 12 and last_vr:
        stv = st[vi]
        ok = np.isfinite(stv)
        if ok.sum() >= 12:
            t = vi[ok] * HOP_S
            a, b = np.polyfit(t, stv[ok], 1)
            f["p_declination"] = float(a)
            tail = np.flatnonzero(voiced[last_vr[0]:last_vr[1]]) + last_vr[0]
            if len(tail) >= 3:
                pred = a * (tail[-3:] * HOP_S) + b
                f["p_decl_resid"] = float(np.mean(st[tail[-3:]] - pred))

    # ---------------- spectral shape of the final phone -------------------
    ss = spectral_stats(x_causal, sr)
    for k, v in ss.items():
        v = v[:n]
        if len(v) == 0:
            continue
        f[f"s_{k}_200"] = float(np.mean(_win(v, anchor, 200)))
        f[f"s_{k}_slope"] = lin_slope(_win(v, anchor, 400))
        sp_v = v[sp] if sp.any() else v
        f[f"s_{k}_rel"] = float(np.mean(_win(v, anchor, 200)) - np.mean(sp_v))

    lm = log_mel_spectrogram(x_causal, sr)[:n]
    if len(lm):
        mf = dct_mfcc(lm, n_mfcc=13)
        term = np.mean(_win(mf, anchor, 200), axis=0)
        prevw = np.mean(mf[max(0, anchor - 44):max(1, anchor - 19)], axis=0)
        # Cepstral mean normalisation against THIS turn's own speech: removes
        # the handset/channel colouring so what is left is the identity of the
        # final phone.  Essential for the unseen (different pool) test set.
        ref_mf = np.mean(mf[sp], axis=0) if sp.any() else np.mean(mf, axis=0)
        for i in range(1, 13):
            f[f"m_mfcc{i}"] = float(term[i] - ref_mf[i])
        for i in range(1, 7):
            f[f"m_dmfcc{i}"] = float(term[i] - prevw[i])
        f["m_c0_delta"] = float(term[0] - prevw[0])

        # --- articulation / elongation ---------------------------------
        # Frame-to-frame cepstral distance = how fast the vocal tract is
        # moving.  A held-out vowel ("aur...", "soooo...", "umm") is
        # spectrally FROZEN, which is the classic hesitation-before-more-
        # speech cue; a completed word keeps moving right up to the offset.
        dmf = np.linalg.norm(np.diff(mf[:, 1:9], axis=0), axis=1)
        if len(dmf) > 5:
            a2 = min(anchor, len(dmf) - 1)
            f["m_specflux_150"] = float(np.mean(_win(dmf, a2, 150)))
            f["m_specflux_400"] = float(np.mean(_win(dmf, a2, 400)))
            sp_flux = dmf[sp[:len(dmf)]] if sp[:len(dmf)].any() else dmf
            ref_flux = float(np.median(sp_flux)) + 1e-6
            f["m_specflux_ratio"] = f["m_specflux_150"] / ref_flux
            f["m_specflux_ratio400"] = f["m_specflux_400"] / ref_flux
            # length of the frozen stretch running back from the offset
            thr = 0.65 * ref_flux
            k = 0
            i = a2
            while i >= 0 and dmf[i] < thr and k < 120:
                k += 1
                i -= 1
            f["m_steady_tail"] = k * HOP_S
            f["m_steady_and_voiced"] = f["m_steady_tail"] * f.get("p_voiced_frac_500", 0.0)
            # hesitation score: long frozen voiced tail with flat pitch
            flat = 1.0 / (1.0 + abs(f.get("p_slope_400", 0.0)))
            f["m_hesitation"] = f["m_steady_tail"] * flat * f.get("p_voiced_frac_500", 0.0)

        # --- "acoustic shape of the final word" ------------------------
        # Six 100 ms slices running back from the speech offset, each
        # described by channel-normalised cepstra + speaker-relative pitch +
        # relative energy.  We cannot run ASR, but turn-final words ("thank
        # you", "that's it", "haan ji", "bas") and hold-final words ("and",
        # "aur", "toh", "umm") have very different trajectories, and the
        # trees can learn those templates straight from this handout.
        for j in range(6):
            hi_i = anchor - j * 10
            lo_i = max(0, hi_i - 9)
            if hi_i < 0:
                break
            sl = mf[lo_i:hi_i + 1]
            if len(sl) == 0:
                continue
            v = sl.mean(axis=0) - ref_mf
            for i in range(1, 9):
                f[f"q{j}_c{i}"] = float(v[i])
            f[f"q{j}_e"] = float(np.mean(db[lo_i:hi_i + 1]) - e_ref)
            sub = st[lo_i:hi_i + 1]
            sub = sub[np.isfinite(sub)]
            if len(sub):                      # else left missing on purpose
                f[f"q{j}_st"] = float(np.mean(sub))
            f[f"q{j}_vf"] = float(np.mean(voiced[lo_i:hi_i + 1]))

    return {k: safe(v) for k, v in f.items()}


# ---------------------------------------------------------- turn assembly ----
# Features that get a second, "relative to the pauses I already held through"
# version (see add_relative_features in dataset.py).  Strictly causal: a pause
# only ever looks at EARLIER pauses of the same turn.
RELATIVE_BASE = [
    "en_term_150", "en_term_300", "en_slope_500", "en_fade_depth",
    "en_pctile_term", "p_end_st", "p_run_slope", "p_slope_400", "p_range",
    "p_end_minus_max", "p_final_run_dur", "p_end_below_ref", "p_strength_end",
    "r_final_len_ratio", "r_last_speech_run", "r_rate_ratio",
    "p_voiced_frac_500", "s_tilt_rel", "s_centroid_rel", "s_hflf_rel",
    "m_mfcc1", "m_mfcc2", "m_mfcc3",
    "p_decl_resid", "m_hesitation", "m_steady_tail", "m_specflux_ratio",
    "m_specflux_ratio400",
]


def extract_turn(x, sr, rows):
    """Features for every pause of one turn.  `rows` = that turn's label rows
    ordered by pause_index.  Returns a list of dicts, one per pause."""
    out = []
    prev = []
    for r in rows:
        ps = float(r["pause_start"])
        ctx = {"pause_start": ps, "pause_index": int(r["pause_index"]),
               "prev_pauses": list(prev)}
        out.append(base_features(causal_slice(x, sr, ps), sr, ctx))
        prev.append((ps, float(r["pause_end"])))   # becomes PAST for later rows
    return out

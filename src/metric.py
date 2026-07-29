"""Exact re-implementation of score.py's policy sweep, callable in-process.

Verified against score.py (see RUNLOG.md).  Used for model selection so that
every decision is made on the metric we are actually graded on, not on AUC.
"""
import numpy as np

TIMEOUT_S = 1.6
THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 3)
DELAYS = np.round(np.arange(0.10, 1.65, 0.05), 3)


def sweep(turn_ids, durs, is_eot, p, budget=0.05):
    turn_ids = np.asarray(turn_ids)
    durs = np.asarray(durs, dtype=float)
    is_eot = np.asarray(is_eot).astype(bool)
    p = np.asarray(p, dtype=float)

    uniq, tidx = np.unique(turn_ids, return_inverse=True)
    n_turns = len(uniq)
    hold = ~is_eot
    best = None
    for t in THRESHOLDS:
        fires = p >= t
        eot_fire_frac = fires[is_eot].mean() if is_eot.any() else 0.0
        hf = fires & hold
        hd = durs[hf]
        ht = tidx[hf]
        for d in DELAYS:
            bad = ht[hd > d]
            cut = len(np.unique(bad)) / max(1, n_turns)
            if cut > budget:
                continue
            lat = d * eot_fire_frac + TIMEOUT_S * (1.0 - eot_fire_frac)
            if best is None or lat < best["latency"]:
                best = {"latency": lat, "cutoff": cut, "threshold": float(t),
                        "delay": float(d), "eot_recall": float(eot_fire_frac)}
    if best is None:
        best = {"latency": TIMEOUT_S, "cutoff": 0.0, "threshold": 1.0,
                "delay": TIMEOUT_S, "eot_recall": 0.0}
    return best


def auc(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    if not n1 or not n0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def report(df, p, budget=0.05):
    r = sweep(df["turn_id"].to_numpy(),
              (df["pause_end"] - df["pause_start"]).to_numpy(),
              (df["label"].to_numpy() == "eot"), p, budget)
    r["auc"] = auc(df["label"].to_numpy() == "eot", p)
    return r


def fmt(r):
    return (f"delay={r['latency']*1000:6.0f}ms  cut={r['cutoff']*100:4.1f}%  "
            f"AUC={r['auc']:.3f}  (thr={r['threshold']:.2f} d={r['delay']*1000:.0f}ms "
            f"recall={r['eot_recall']:.2f})")

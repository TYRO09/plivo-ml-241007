"""Fast experiment harness: cached features -> pooled CV on the real metric."""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(SRC)
sys.path.insert(0, SRC)
import dataset
import metric
from model import BlendModel, risk_weights

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts")


def load():
    en = pd.read_parquet(os.path.join(CACHE, "feats_english.parquet"))
    hi = pd.read_parquet(os.path.join(CACHE, "feats_hindi.parquet"))
    en["lang"], hi["lang"] = "english", "hindi"
    return pd.concat([en, hi], ignore_index=True).reset_index(drop=True)


def run(both, cols, hold_min_dur=0.0, weight_kw=None, model_kw=None,
        n_folds=5, cv_seeds=1, verbose=True, tag=""):
    """Pooled CV.  `hold_min_dur` drops short holds from TRAINING only -- they
    can never cause a false cutoff at any useful action delay, so they are
    label noise for the decision we actually make."""
    X = both[cols].to_numpy(dtype=float)
    y = (both["label"].to_numpy() == "eot").astype(int)
    dur = (both["pause_end"] - both["pause_start"]).to_numpy()
    groups = both["turn_id"].to_numpy()
    w = risk_weights(both, **(weight_kw or {}))
    keep_train = (y == 1) | (dur >= hold_min_dur)

    oof = np.zeros((cv_seeds, len(both)))
    for s in range(cv_seeds):
        rs = np.random.RandomState(s)
        uniq = np.unique(groups)
        remap = {u: i for i, u in enumerate(uniq[rs.permutation(len(uniq))])}
        shuf = np.array([remap[g] for g in groups])
        for tr, te in GroupKFold(n_splits=n_folds).split(X, y, shuf):
            tr = tr[keep_train[tr]]
            m = BlendModel(**({"n_seeds": 2} | (model_kw or {})))
            m.fit(X[tr], y[tr], sample_weight=w[tr])
            oof[s, te] = m.predict_proba(X[te])[:, 1]
    p = oof.mean(axis=0)

    out = {}
    for lang in ["english", "hindi"]:
        d = both[both.lang == lang]
        out[lang] = metric.report(d, p[(both.lang == lang).to_numpy()])
    out["mean"] = 0.5 * (out["english"]["latency"] + out["hindi"]["latency"])
    if verbose:
        print(f"{tag:44s} mean={out['mean']*1000:6.0f}  "
              f"en {metric.fmt(out['english'])}  |  hi {metric.fmt(out['hindi'])}")
    return out, p


def restricted_auc_rank(both, cols, min_dur=0.35):
    """Rank features by |AUC-0.5| on the decision that matters: EOT vs the
    holds that are long enough to actually cost us a false cutoff."""
    dur = (both["pause_end"] - both["pause_start"]).to_numpy()
    is_eot = (both["label"] == "eot").to_numpy()
    sub = both[is_eot | (dur >= min_dur)]
    y = (sub["label"] == "eot").astype(int).to_numpy()
    sc = {}
    for c in cols:
        v = sub[c].to_numpy(float)
        m = np.isfinite(v)
        if m.sum() < 80 or np.nanstd(v[m]) == 0:
            continue
        sc[c] = abs(metric.auc(y[m], v[m]) - 0.5)
    return pd.Series(sc).sort_values(ascending=False)


if __name__ == "__main__":
    both = load()
    cols = [c for c in dataset.feature_columns(both) if c != "lang"]
    print(f"{len(cols)} features, {len(both)} pauses")
    run(both, cols, tag="A all feats, risk weights")
    run(both, cols, hold_min_dur=0.25, tag="B drop holds <250ms from train")
    run(both, cols, hold_min_dur=0.35, tag="C drop holds <350ms from train")
    run(both, cols, weight_kw=dict(floor=0.05, gain=4.0, pivot=0.45, sharp=0.10),
        tag="D sharper risk weights")
    run(both, cols, hold_min_dur=0.25,
        weight_kw=dict(floor=0.05, gain=4.0, pivot=0.45, sharp=0.10),
        tag="E B+D")
    r = restricted_auc_rank(both, cols)
    print("\ntop 30 by EOT-vs-longhold AUC:")
    print(r.head(30).round(3).to_string())
    for k in (25, 45, 70):
        run(both, r.head(k).index.tolist(), hold_min_dur=0.25, tag=f"F top-{k} feats")

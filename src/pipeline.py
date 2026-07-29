"""Full system: VAP (self-supervised) + prosodic model (supervised), blended.

Honest evaluation: for every fold, BOTH models are trained only on that fold's
training TURNS, so the VAP frame model never sees a test turn's audio.

    python pipeline.py            # 5-fold GroupKFold, pooled over languages
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset
import metric
import vap_train
from model import BlendModel, risk_weights

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "eot_handout", "eot_data")
CACHE = os.path.join(HERE, "..", "artifacts")
VAP_KEYS = ["vap_p_eot", "vap_p_quiet_600", "vap_p_quiet_1200",
            "vap_p_resume_fast", "vap_expected_bin", "vap_logit_eot"]


def ranks(v):
    v = np.asarray(v, dtype=float)
    v = np.nan_to_num(v, nan=np.nanmedian(v[np.isfinite(v)]) if np.isfinite(v).any() else 0.5)
    o = np.argsort(v)
    r = np.empty(len(v), dtype=float)
    r[o] = np.arange(len(v))
    return r / max(len(v) - 1, 1)


def load_all(langs=("english", "hindi")):
    pause_tables, frame_tables = [], {}
    for lang in langs:
        p = os.path.join(CACHE, f"feats_{lang}.parquet")
        df = pd.read_parquet(p) if os.path.exists(p) else dataset.build_table(
            os.path.join(DATA, lang))
        df["lang"] = lang
        pause_tables.append(df)
        frame_tables.update(vap_train.build_frames(os.path.join(DATA, lang)))
    both = pd.concat(pause_tables, ignore_index=True).reset_index(drop=True)
    return both, frame_tables


def cv(both, frames, n_folds=5, cv_seeds=2, n_seeds=2, verbose=True):
    cols = [c for c in dataset.feature_columns(both) if c != "lang"]
    X = both[cols].to_numpy(dtype=float)
    y = (both["label"].to_numpy() == "eot").astype(int)
    groups = both["turn_id"].to_numpy()
    w = risk_weights(both, pivot=0.42, sharp=0.13, floor=0.30, gain=2.6)

    oof_pros = np.zeros((cv_seeds, len(both)))
    oof_vap = np.zeros((cv_seeds, len(both)))
    for s in range(cv_seeds):
        rs = np.random.RandomState(s)
        uniq = np.unique(groups)
        remap = {u: i for i, u in enumerate(uniq[rs.permutation(len(uniq))])}
        shuf = np.array([remap[g] for g in groups])
        for tr, te in GroupKFold(n_splits=n_folds).split(X, y, shuf):
            # --- supervised prosodic model -----------------------------
            m = BlendModel(n_seeds=n_seeds)
            m.fit(X[tr], y[tr], sample_weight=w[tr])
            oof_pros[s, te] = m.predict_proba(X[te])[:, 1]
            # --- self-supervised VAP model (audio of TRAIN turns only) --
            tr_turns = np.unique(groups[tr])
            Xf, yf = vap_train.sample_training_frames(frames, tr_turns, seed=s)
            vm = vap_train.fit_vap(Xf, yf, seed=s)
            z = vap_train.score_pauses(vm, frames, both.iloc[te])
            oof_vap[s, te] = z["vap_p_eot"].fillna(0.5).to_numpy()
    return oof_pros.mean(axis=0), oof_vap.mean(axis=0)


def evaluate(both, p_pros, p_vap, weights=(0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0),
             verbose=True):
    en_m = (both.lang == "english").to_numpy()
    hi_m = (both.lang == "hindi").to_numpy()
    rows = []
    for wv in weights:
        p = (1 - wv) * ranks(p_pros) + wv * ranks(p_vap)
        r_en = metric.report(both[en_m], p[en_m])
        r_hi = metric.report(both[hi_m], p[hi_m])
        rows.append(dict(w_vap=wv, en=r_en["latency"] * 1000, hi=r_hi["latency"] * 1000,
                         mean=(r_en["latency"] + r_hi["latency"]) * 500,
                         auc_en=r_en["auc"], auc_hi=r_hi["auc"]))
        if verbose:
            tag = ("prosodic only" if wv == 0 else
                   "VAP zero-shot only" if wv == 1 else f"blend w_vap={wv}")
            print(f"  {tag:22s} en {metric.fmt(r_en)}  |  hi {metric.fmt(r_hi)}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cv_seeds", type=int, default=2)
    args = ap.parse_args()
    both, frames = load_all()
    print(f"{len(both)} pauses, {both.turn_id.nunique()} turns, "
          f"{len(frames)} frame tables")
    p_pros, p_vap = cv(both, frames, n_folds=args.folds, cv_seeds=args.cv_seeds)
    np.save(os.path.join(CACHE, "oof_pros.npy"), p_pros)
    np.save(os.path.join(CACHE, "oof_vap.npy"), p_vap)
    tab = evaluate(both, p_pros, p_vap)
    print(tab.round(1).to_string(index=False))
    best = tab.loc[tab["mean"].idxmin()]
    print(f"\nbest blend weight: w_vap={best.w_vap}  mean={best['mean']:.0f} ms")


if __name__ == "__main__":
    main()

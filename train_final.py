"""Train the shipped model on ALL provided data and save it to artifacts/.

    python train_final.py [--data_root eot_handout/eot_data]

The saved bundle is what predict.py loads; predict.py never refits.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from joblib import dump

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import dataset                                        # noqa: E402
import metric                                         # noqa: E402
from model import BlendModel, risk_weights            # noqa: E402

# Chosen by 5-fold GroupKFold CV on the OFFICIAL metric -- see RUNLOG.md.
# hold_min_dur=0 (keeping short holds in training beat dropping them) and the
# moderate risk-weight curve both won on CV; the sharper variants overfit.
CONFIG = dict(hold_min_dur=0.0, n_seeds=5, w_linear=0.35,
              weight_kw=dict(pivot=0.42, sharp=0.13, floor=0.30, gain=2.6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.path.join(HERE, "..", "eot_handout", "eot_data"))
    ap.add_argument("--langs", nargs="+", default=["english", "hindi"])
    ap.add_argument("--out", default=os.path.join(HERE, "artifacts", "model.joblib"))
    ap.add_argument("--use_cache", action="store_true")
    args = ap.parse_args()

    tables = []
    for lang in args.langs:
        cache = os.path.join(HERE, "artifacts", f"feats_{lang}.parquet")
        if args.use_cache and os.path.exists(cache):
            df = pd.read_parquet(cache)
        else:
            df = dataset.build_table(os.path.join(args.data_root, lang))
            df.to_parquet(cache)
        df["lang"] = lang
        tables.append(df)
    both = pd.concat(tables, ignore_index=True)

    cols = [c for c in dataset.feature_columns(both) if c != "lang"]
    X = both[cols].to_numpy(dtype=float)
    y = (both["label"].to_numpy() == "eot").astype(int)
    dur = (both["pause_end"] - both["pause_start"]).to_numpy()
    w = risk_weights(both, **CONFIG["weight_kw"])

    keep = (y == 1) | (dur >= CONFIG["hold_min_dur"])
    print(f"training on {keep.sum()}/{len(both)} rows "
          f"({y[keep].sum()} eot, {(~y[keep].astype(bool)).sum()} long holds), "
          f"{len(cols)} features")

    m = BlendModel(n_seeds=CONFIG["n_seeds"], w_linear=CONFIG["w_linear"])
    m.fit(X[keep], y[keep], sample_weight=w[keep])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dump({"model": m, "cols": cols, "config": CONFIG,
          "trained_on": args.langs, "n_rows": int(keep.sum())}, args.out)
    print(f"saved -> {args.out}")

    # in-sample sanity only (real numbers live in RUNLOG.md / cv.py)
    p = m.predict_proba(X)[:, 1]
    for lang in args.langs:
        d = both[both.lang == lang]
        print(f"  in-sample {lang:8s}: "
              f"{metric.fmt(metric.report(d, p[(both.lang == lang).to_numpy()]))}")
    with open(os.path.join(HERE, "artifacts", "train_meta.json"), "w") as f:
        json.dump({"config": {k: str(v) for k, v in CONFIG.items()},
                   "n_features": len(cols), "langs": args.langs}, f, indent=2)


if __name__ == "__main__":
    main()

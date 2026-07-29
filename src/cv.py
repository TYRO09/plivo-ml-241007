"""Cross-validated evaluation on the OFFICIAL metric.

  python cv.py                # pooled 5-fold GroupKFold(turn) + cross-language
  python cv.py --rebuild      # force feature re-extraction

Everything is grouped by turn_id, so no turn is ever split across folds.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset
import metric
from model import build_model, fit_predict_cv

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                    "eot_handout", "eot_data")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts")


def get_table(lang, rebuild=False):
    p = os.path.join(CACHE, f"feats_{lang}.parquet")
    if os.path.exists(p) and not rebuild:
        return pd.read_parquet(p)
    df = dataset.build_table(os.path.join(DATA, lang))
    df["lang"] = lang
    os.makedirs(CACHE, exist_ok=True)
    df.to_parquet(p)
    return df


def load_both(rebuild=False):
    en = get_table("english", rebuild)
    hi = get_table("hindi", rebuild)
    en["lang"], hi["lang"] = "english", "hindi"
    return en, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    en, hi = load_both(args.rebuild)
    both = pd.concat([en, hi], ignore_index=True)
    cols = [c for c in dataset.feature_columns(both) if c != "lang"]
    print(f"features: {len(cols)}   pauses: {len(both)}   turns: {both.turn_id.nunique()}")

    # ---- pooled CV (train on both languages, predict out-of-fold) ----------
    oof = fit_predict_cv(both, cols, n_folds=args.folds, n_seeds=args.seeds)
    both["p"] = oof
    lines = []
    for lang in ["english", "hindi"]:
        d = both[both.lang == lang]
        r = metric.report(d, d["p"].to_numpy())
        lines.append(f"  CV pooled  {lang:8s}: {metric.fmt(r)}")
    r_all = metric.report(both, oof)
    lines.append(f"  CV pooled  ALL     : {metric.fmt(r_all)}")

    # ---- cross-language: train on one, test on the other (transfer check) --
    for tr, te in [(en, hi), (hi, en)]:
        m = build_model()
        from model import risk_weights
        m.fit(tr[cols].to_numpy(float), (tr.label == "eot").astype(int),
              sample_weight=risk_weights(tr))
        p = m.predict_proba(te[cols].to_numpy(float))[:, 1]
        r = metric.report(te, p)
        lines.append(f"  XLANG {tr.lang.iloc[0][:2]}->{te.lang.iloc[0][:2]}      : {metric.fmt(r)}")

    print("\n".join(lines))
    if args.tag:
        with open(os.path.join(CACHE, "cv_history.txt"), "a") as f:
            f.write(f"[{args.tag}]\n" + "\n".join(lines) + "\n")
    return both


if __name__ == "__main__":
    main()

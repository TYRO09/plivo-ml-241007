"""End-of-turn detection: score every pause of an unseen folder.

    python predict.py --data_dir <folder> --out predictions.csv

<folder> must contain labels.csv (turn_id, audio_file, pause_index,
pause_start, pause_end, label) and the wavs it points at.  The `label` column
is not required and is never read by the model.  Output columns:
turn_id,pause_index,p_eot

Loads the pre-trained model from artifacts/model.joblib -- it does not refit.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from joblib import load

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import dataset                                       # noqa: E402

DEFAULT_MODEL = os.path.join(HERE, "artifacts", "model.joblib")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args()

    bundle = load(args.model)
    model, cols = bundle["model"], bundle["cols"]

    df = dataset.build_table(args.data_dir, n_jobs=args.n_jobs)

    # align to the training feature order; anything the extractor could not
    # produce for this folder stays NaN (the trees handle that natively)
    missing = [c for c in cols if c not in df.columns]
    for c in missing:
        df[c] = np.nan
    if missing:
        print(f"note: {len(missing)} feature(s) absent for this folder -> NaN")
    X = df[cols].to_numpy(dtype=float)

    p = model.predict_proba(X)[:, 1]
    p = np.clip(np.nan_to_num(p, nan=0.5), 0.0, 1.0)

    out = pd.DataFrame({"turn_id": df["turn_id"],
                        "pause_index": df["pause_index"].astype(int),
                        "p_eot": np.round(p, 6)})
    # emit in the same order as the input labels.csv
    lab = pd.read_csv(os.path.join(args.data_dir, "labels.csv"))
    key = ["turn_id", "pause_index"]
    lab["pause_index"] = lab["pause_index"].astype(int)
    out = lab[key].merge(out, on=key, how="left")
    out["p_eot"] = out["p_eot"].fillna(0.5)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} predictions -> {args.out}")


if __name__ == "__main__":
    main()

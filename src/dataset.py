"""Build the pause-level feature table for a data_dir (english/ or hindi/)."""
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from features_eot import RELATIVE_BASE, extract_turn, load_wav

META = ["turn_id", "pause_index", "pause_start", "pause_end", "label", "audio_file"]


def _one_turn(data_dir, audio_file, rows):
    x, sr = load_wav(os.path.join(data_dir, audio_file))
    return extract_turn(x, sr, rows)


def build_table(data_dir, n_jobs=-1, verbose=0):
    lab = pd.read_csv(os.path.join(data_dir, "labels.csv"))
    lab = lab.sort_values(["turn_id", "pause_index"]).reset_index(drop=True)
    turns = list(lab.groupby("turn_id", sort=True))

    jobs = [delayed(_one_turn)(data_dir, g["audio_file"].iloc[0],
                               g.to_dict("records")) for _, g in turns]
    feats = Parallel(n_jobs=n_jobs, verbose=verbose)(jobs)

    recs, meta = [], []
    for (tid, g), fl in zip(turns, feats):
        for (_, r), fd in zip(g.iterrows(), fl):
            recs.append(fd)
            meta.append({k: r[k] for k in META if k in r})
    X = pd.DataFrame(recs)
    M = pd.DataFrame(meta)
    df = pd.concat([M, X], axis=1)
    df = add_relative_features(df)
    return df


def add_relative_features(df):
    """For each pause, compare a feature to the SAME feature at the earlier
    pauses of the same turn.  This is the 'does this pause sound more final
    than the ones I already held through?' signal, and it cancels speaker,
    channel and language variance.  Causal: expanding window, shifted by 1.
    """
    df = df.sort_values(["turn_id", "pause_index"]).reset_index(drop=True)
    g = df.groupby("turn_id", sort=False)
    new = {}
    for c in RELATIVE_BASE:
        if c not in df.columns:
            continue
        prev_mean = g[c].transform(lambda s: s.shift(1).expanding().mean())
        prev_max = g[c].transform(lambda s: s.shift(1).expanding().max())
        prev_std = g[c].transform(lambda s: s.shift(1).expanding().std())
        new[f"rel_{c}_dmean"] = df[c] - prev_mean
        new[f"rel_{c}_dmax"] = df[c] - prev_max
        new[f"rel_{c}_z"] = (df[c] - prev_mean) / (prev_std + 1e-3)
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def feature_columns(df):
    return [c for c in df.columns if c not in META]


def xy(df):
    cols = feature_columns(df)
    X = df[cols].to_numpy(dtype=np.float64)
    y = (df["label"].to_numpy() == "eot").astype(int) if "label" in df else None
    return X, y, cols

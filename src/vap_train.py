"""Train / apply the VAP projection model and read out zero-shot p_eot."""
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier

import vap
from features_eot import load_wav

MAX_SIL_TRAIN = 0.60          # keep silence frames up to 600 ms into a pause
SPEECH_KEEP = 0.20            # subsample of frames where the user is talking


def _one_file(data_dir, audio_file):
    x, sr = load_wav(os.path.join(data_dir, audio_file))
    F = vap.frame_features(x, sr)
    v = vap.frame_vad(x, sr)
    n = min(len(F), len(v))
    cls, _ = vap.projection_classes(v[:n])
    return F[:n], cls[:n]


def build_frames(data_dir, n_jobs=-1):
    """Frame table for every turn in a folder: {turn_id: (features, targets)}."""
    lab = pd.read_csv(os.path.join(data_dir, "labels.csv"))
    files = lab[["turn_id", "audio_file"]].drop_duplicates().values.tolist()
    res = Parallel(n_jobs=n_jobs)(delayed(_one_file)(data_dir, af) for _, af in files)
    return {tid: r for (tid, _), r in zip(files, res)}


# Frames to keep per detected silence, as multiples of the 20 ms frame:
# 20 / 100 / 200 / 300 ms into the silence.  One decision per silence EVENT
# rather than per frame -- consecutive frames inside one pause are almost
# perfectly correlated, so keeping all of them inflates the apparent sample
# size and makes the model memorise turns instead of generalising.
EVENT_OFFSETS = (1, 5, 10, 15)


def sample_training_frames(frames, turn_ids, seed=0, per_event=True):
    """Frames worth learning from: the decision points right after a speech
    offset.  Frames while the user is still speaking are dropped -- their
    target is trivially "resumes immediately" and they only dilute training."""
    rs = np.random.RandomState(seed)
    Xs, ys = [], []
    names = vap.feature_names()
    i_sil = names.index("sil_len")
    for tid in turn_ids:
        if tid not in frames:
            continue
        F, cls = frames[tid]
        if len(F) == 0:
            continue
        sil_frames = np.round(F[:, i_sil] * vap.FRAME_HZ).astype(int)
        if per_event:
            keep = np.isin(sil_frames, EVENT_OFFSETS)
        else:
            keep = (sil_frames > 0) & (F[:, i_sil] <= MAX_SIL_TRAIN)
        if keep.sum() == 0:
            continue
        Xs.append(F[keep])
        ys.append(cls[keep])
    if not Xs:
        return np.zeros((0, 1)), np.zeros(0, dtype=int)
    X, y = np.concatenate(Xs), np.concatenate(ys)
    # 3-class target: more mass per class at this (small) event count.
    #   0 = resumes within 600 ms, 1 = resumes in 600-2000 ms, 2 = end of turn
    y3 = np.where(y <= 1, 0, np.where(y <= 3, 1, 2))
    return X, y3


def fit_vap(X, y, seed=0):
    m = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.06, max_iter=140, max_leaf_nodes=8,
        min_samples_leaf=25, l2_regularization=2.0, max_bins=64,
        early_stopping=False, random_state=seed)
    m.fit(X, y)
    return m


def score_pauses(model, frames, lab):
    """Zero-shot VAP scalars for every pause row in `lab`."""
    rows, idx = [], []
    for i, r in lab.iterrows():
        tid = r["turn_id"]
        if tid not in frames or len(frames[tid][0]) == 0:
            continue
        F = frames[tid][0]
        t = min(vap.frame_index_for(r["pause_start"]), len(F) - 1)
        rows.append(F[t])
        idx.append(i)
    if not rows:
        return pd.DataFrame(index=lab.index)
    P = model.predict_proba(np.stack(rows))
    n_cls = int(max(model.classes_)) + 1
    full = np.zeros((len(P), n_cls))
    for j, c in enumerate(model.classes_):
        full[:, int(c)] = P[:, j]
    z = vap.zero_shot(full, n_classes=n_cls)
    out = pd.DataFrame(z, index=idx)
    return out.reindex(lab.index)

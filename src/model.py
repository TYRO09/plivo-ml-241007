"""The model: gradient-boosted trees + regularised logistic regression.

Why this and not a neural net: 496 labelled pauses total.  Trees handle the
missing-by-construction features (no final voiced run -> NaN) natively, and
the linear model contributes a smooth, monotone-ish extrapolation that keeps
the ranking sane on the unseen (mostly Hindi) test pool.  Rank-averaging the
two is consistently better than either alone on the official metric.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer


def _hgb(seed=0):
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.045,
        max_iter=400,
        max_leaf_nodes=7,
        min_samples_leaf=14,
        l2_regularization=2.0,
        max_features=0.55,
        max_bins=64,
        early_stopping=False,
        random_state=seed,
    )


def _linear(seed=0):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("qt", QuantileTransformer(output_distribution="normal",
                                   n_quantiles=200, random_state=seed)),
        ("lr", LogisticRegression(C=0.06, max_iter=4000,
                                  class_weight="balanced")),
    ])


class BlendModel:
    """Rank-average of an HGB ensemble and a linear model."""

    def __init__(self, n_seeds=5, w_linear=0.35):
        self.n_seeds = n_seeds
        self.w_linear = w_linear
        self.trees, self.lin = [], None

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.trees = []
        for s in range(self.n_seeds):
            m = _hgb(seed=s)
            m.fit(X, y, sample_weight=sample_weight)
            self.trees.append(m)
        self.lin = _linear()
        self.lin.fit(X, y, lr__sample_weight=sample_weight)
        return self

    def _ranks(self, v):
        order = np.argsort(v)
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(len(v))
        return r / max(len(v) - 1, 1)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        pt = np.mean([m.predict_proba(X)[:, 1] for m in self.trees], axis=0)
        pl = self.lin.predict_proba(X)[:, 1]
        if len(X) < 12:                    # too few rows to rank-average
            p = (1 - self.w_linear) * pt + self.w_linear * pl
        else:
            p = (1 - self.w_linear) * self._ranks(pt) + self.w_linear * self._ranks(pl)
            p = 0.02 + 0.96 * p            # keep strictly inside (0,1)
        return np.column_stack([1 - p, p])


def build_model(**kw):
    return BlendModel(**kw)


def risk_weights(df, pivot=0.42, sharp=0.13, floor=0.30, gain=2.6, eot_w=1.0):
    """Cost-sensitive weights that match the scorer's asymmetry.

    A hold pause can only cause a false cutoff if the agent's action delay is
    SHORTER than that pause.  Short holds are therefore almost free, long
    "thinking" holds are what actually cost us.  So we up-weight long holds and
    down-weight the sub-300 ms breath pauses.  This uses the pause DURATION of
    training labels only -- it never becomes a feature, so causality at
    inference time is unaffected.
    """
    dur = (df["pause_end"] - df["pause_start"]).to_numpy(dtype=float)
    is_eot = (df["label"].to_numpy() == "eot")
    w_hold = floor + gain / (1.0 + np.exp(-(dur - pivot) / sharp))
    w = np.where(is_eot, eot_w, w_hold)
    # keep the two classes' total mass balanced so the threshold stays usable
    w[is_eot] *= (w[~is_eot].sum() / max(w[is_eot].sum(), 1e-9))
    return w


def fit_predict_cv(df, cols, n_folds=5, n_seeds=3, model_factory=None,
                   weights=True):
    """Out-of-fold probabilities, grouped by turn, averaged over seeds.

    Ranks are computed WITHIN each fold's test block and then pooled, which
    mirrors how the scorer sees a whole unseen folder at once.
    """
    X = df[cols].to_numpy(dtype=float)
    y = (df["label"].to_numpy() == "eot").astype(int)
    groups = df["turn_id"].to_numpy()
    w = risk_weights(df) if weights else None
    oof = np.zeros((n_seeds, len(df)))
    for s in range(n_seeds):
        rs = np.random.RandomState(s)
        uniq = np.unique(groups)
        perm = rs.permutation(len(uniq))
        remap = {u: i for u, i in zip(uniq[perm], range(len(uniq)))}
        shuffled = np.array([remap[g] for g in groups])
        gkf = GroupKFold(n_splits=n_folds)
        for tr, te in gkf.split(X, y, shuffled):
            m = (model_factory or build_model)()
            m.fit(X[tr], y[tr], sample_weight=None if w is None else w[tr])
            oof[s, te] = m.predict_proba(X[te])[:, 1]
    return oof.mean(axis=0)

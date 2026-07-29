"""Generate the honest out-of-fold predictions, the figures and results.json."""
import base64
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset
import metric
from model import BlendModel, risk_weights

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
CACHE = os.path.join(ROOT, "artifacts")
FIGS = os.path.join(ROOT, "figs")
C = {"en": "#2b6cb0", "hi": "#dd6b20", "base": "#a0aec0", "eot": "#2f855a",
     "hold": "#c53030"}


def load():
    en = pd.read_parquet(os.path.join(CACHE, "feats_english.parquet"))
    hi = pd.read_parquet(os.path.join(CACHE, "feats_hindi.parquet"))
    en["lang"], hi["lang"] = "english", "hindi"
    return pd.concat([en, hi], ignore_index=True).reset_index(drop=True)


def oof(both, cols, n_folds=5, cv_seeds=3, n_seeds=3):
    X = both[cols].to_numpy(float)
    y = (both.label == "eot").astype(int).to_numpy()
    g = both.turn_id.to_numpy()
    w = risk_weights(both)
    acc = np.zeros((cv_seeds, len(both)))
    for s in range(cv_seeds):
        rs = np.random.RandomState(s)
        u = np.unique(g)
        remap = {v: i for i, v in enumerate(u[rs.permutation(len(u))])}
        shuf = np.array([remap[v] for v in g])
        for tr, te in GroupKFold(n_splits=n_folds).split(X, y, shuf):
            m = BlendModel(n_seeds=n_seeds)
            m.fit(X[tr], y[tr], sample_weight=w[tr])
            acc[s, te] = m.predict_proba(X[te])[:, 1]
    return acc.mean(axis=0)


def b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fig_scores(res):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    labels = ["English", "Hindi"]
    base = [res["baseline"]["english"], res["baseline"]["hindi"]]
    ours = [res["cv"]["english"], res["cv"]["hindi"]]
    xs = np.arange(2)
    ax.bar(xs - 0.2, base, 0.38, label="silence-only baseline", color=C["base"])
    ax.bar(xs + 0.2, ours, 0.38, label="this model (held-out CV)", color=C["en"])
    for x, v in zip(xs - 0.2, base):
        ax.text(x, v + 25, f"{v:.0f}", ha="center", fontsize=9)
    for x, v in zip(xs + 0.2, ours):
        ax.text(x, v + 25, f"{v:.0f}", ha="center", fontsize=9, weight="bold")
    ax.set_xticks(xs, labels)
    ax.set_ylabel("mean response delay @ ≤5% cutoffs (ms)")
    ax.set_title("Lower is better")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return b64(fig)


def fig_dist(both, p):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, lang in zip(axes, ["english", "hindi"]):
        m = (both.lang == lang).to_numpy()
        d = both[m]
        pp = p[m]
        for lab, col in [("hold", C["hold"]), ("eot", C["eot"])]:
            ax.hist(pp[(d.label == lab).to_numpy()], bins=22, alpha=0.65,
                    label=lab, color=col)
        ax.set_title(f"{lang} — out-of-fold p_eot")
        ax.set_xlabel("p_eot")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    return b64(fig)


def fig_risk(both, p):
    """The plot that explains the metric: only holds ABOVE the action delay
    can cause a false cutoff."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    d = both.copy()
    d["dur"] = d.pause_end - d.pause_start
    d["p"] = p
    h = d[d.label == "hold"]
    e = d[d.label == "eot"]
    ax.scatter(h.dur, h.p, s=14, alpha=0.6, color=C["hold"], label="hold")
    ax.scatter(e.dur, e.p, s=14, alpha=0.6, color=C["eot"], label="eot")
    ax.axvline(0.42, ls="--", lw=1, color="#4a5568")
    ax.text(0.45, 1.02, "risk-weight pivot (420 ms)", fontsize=8, color="#4a5568")
    ax.set_xlabel("pause duration (s)  — future info, training weights only")
    ax.set_ylabel("out-of-fold p_eot")
    ax.set_xlim(0, 2.2)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return b64(fig)


def fig_frontier(both, p):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for lang, col in [("english", C["en"]), ("hindi", C["hi"])]:
        m = (both.lang == lang).to_numpy()
        d = both[m]
        budgets = np.arange(0.01, 0.31, 0.01)
        ys = [metric.report(d, p[m], budget=b)["latency"] * 1000 for b in budgets]
        ax.plot(budgets * 100, ys, marker="o", ms=3, color=col, label=lang)
    ax.axvline(5, ls="--", lw=1, color="#4a5568")
    ax.text(5.3, ax.get_ylim()[1] * 0.95, "graded budget", fontsize=8,
            color="#4a5568")
    ax.set_xlabel("allowed interrupted turns (%)")
    ax.set_ylabel("mean response delay (ms)")
    ax.set_title("Latency / interruption trade-off (held-out)")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return b64(fig)


def fig_importance(both, cols):
    """Permutation importance, hand-rolled: BlendModel is deliberately not a
    sklearn estimator, and the metric we care about is ranking (AUC)."""
    from sklearn.model_selection import GroupShuffleSplit
    X = both[cols].to_numpy(float)
    y = (both.label == "eot").astype(int).to_numpy()
    g = both.turn_id.to_numpy()
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=0).split(X, y, g))
    m = BlendModel(n_seeds=2).fit(X[tr], y[tr], sample_weight=risk_weights(both)[tr])
    Xte, yte = X[te].copy(), y[te]
    base = metric.auc(yte, m.predict_proba(Xte)[:, 1])
    rs = np.random.RandomState(0)
    drops = np.zeros(len(cols))
    for j in range(len(cols)):
        keep = Xte[:, j].copy()
        acc = []
        for _ in range(3):
            Xte[:, j] = rs.permutation(keep)
            acc.append(base - metric.auc(yte, m.predict_proba(Xte)[:, 1]))
        Xte[:, j] = keep
        drops[j] = np.mean(acc)
    imp = pd.Series(drops, index=cols).sort_values()[-16:]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.barh(imp.index, imp.values, color=C["en"])
    ax.set_xlabel("drop in held-out AUC when the feature is shuffled")
    ax.set_title("What the model actually uses")
    ax.spines[["top", "right"]].set_visible(False)
    return b64(fig), imp


def main():
    os.makedirs(FIGS, exist_ok=True)
    both = load()
    cols = [c for c in dataset.feature_columns(both) if c != "lang"]
    cache_p = os.path.join(CACHE, "oof_final.npy")
    if os.path.exists(cache_p):
        p = np.load(cache_p)
    else:
        p = oof(both, cols)
        np.save(cache_p, p)

    res = {"cv": {}, "baseline": {}, "insample": {}, "n_features": len(cols)}
    for lang in ["english", "hindi"]:
        m = (both.lang == lang).to_numpy()
        d = both[m]
        r = metric.report(d, p[m])
        res["cv"][lang] = r["latency"] * 1000
        res[f"cv_{lang}_full"] = {k: float(v) for k, v in r.items()}
        rb = metric.report(d, np.ones(m.sum()))
        res["baseline"][lang] = rb["latency"] * 1000
        # honest out-of-fold predictions file
        out = pd.DataFrame({"turn_id": d.turn_id, "pause_index": d.pause_index,
                            "p_eot": np.round(p[m], 6)})
        out.to_csv(os.path.join(ROOT, f"predictions_{lang}_heldout.csv"), index=False)

    figs = {"scores": fig_scores(res), "dist": fig_dist(both, p),
            "risk": fig_risk(both, p), "frontier": fig_frontier(both, p)}
    figs["importance"], imp = fig_importance(both, cols)
    res["top_features"] = imp.iloc[::-1].round(4).to_dict()

    with open(os.path.join(CACHE, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(CACHE, "figs.json"), "w") as f:
        json.dump(figs, f)
    print(json.dumps({k: res[k] for k in ("cv", "baseline", "n_features")}, indent=2))
    print("top features:", list(imp.iloc[::-1].index[:10]))


if __name__ == "__main__":
    main()

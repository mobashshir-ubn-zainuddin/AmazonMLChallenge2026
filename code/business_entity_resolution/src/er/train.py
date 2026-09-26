"""
Stage 4 — train the pair classifier and tune the decision rule for macro F0.5.

Split: queries that touch a validation S1 (either as candidate or as true match)
form the validation set; every other selected query is training data. So no
query contributes to both, and validation S1 rows see their COMPLETE set of
competing queries, which makes the validation F0.5 an honest estimate.

Decision rule (shared with predict.py):
  1. each query is assigned only to its highest-probability S1
     (ground truth: an S2/S3 record belongs to at most one S1)
  2. keep the assignment if p >= threshold
  3. per S1 keep at most 5 S2 and 6 S3 records (max seen in train GT)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xgboost as xgb

from .common import load_pickle, log, save_json, split_work_dir, timer
from .features import FEATURE_NAMES
from .metrics import oracle_f05, per_s1_f05

MAX_S2, MAX_S3 = 5, 6


def pick_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def decide(q: np.ndarray, s1: np.ndarray, p: np.ndarray, is_s3_q: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean mask of pairs kept as final matches."""
    n = len(q)
    if n == 0:
        return np.zeros(0, bool)
    order = np.lexsort((-p, q))
    first = np.ones(n, bool)
    first[1:] = q[order][1:] != q[order][:-1]
    best = np.zeros(n, bool)
    best[order[first]] = True
    keep = best & (p >= threshold)
    # per-S1 caps by source
    idx = np.flatnonzero(keep)
    src = is_s3_q[q[idx]].astype(np.int8)
    o = np.lexsort((-p[idx], src, s1[idx]))
    key = s1[idx][o] * 2 + src[o]
    starts = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
    grp = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(key)]))
    rank = np.arange(len(key)) - starts[grp]
    cap = np.where(src[o] == 1, MAX_S3, MAX_S2)
    drop = idx[o][rank >= cap]
    keep[drop] = False
    return keep


def evaluate(q, s1, p, is_s3_q, q_true, n_s1, s1_mask, threshold) -> dict:
    keep = decide(q, s1, p, is_s3_q, threshold)
    f = per_s1_f05(n_s1, q_true, q[keep], s1[keep])[s1_mask]
    ntrue = np.bincount(q_true[q_true >= 0], minlength=n_s1)[s1_mask]
    return {
        "f05": float(f.mean()),
        "f05_zero_match_s1": float(f[ntrue == 0].mean()) if (ntrue == 0).any() else float("nan"),
        "f05_single_s1": float(f[ntrue == 1].mean()) if (ntrue == 1).any() else float("nan"),
        "f05_multi_s1": float(f[ntrue > 1].mean()) if (ntrue > 1).any() else float("nan"),
        "pred_pairs": int(keep.sum()),
    }


def run(work_dir: Path, rounds: int = 1500) -> None:
    wdir = split_work_dir(work_dir, "train")
    X = np.load(wdir / "X.npy", mmap_mode="r")
    m = np.load(wdir / "pairs_meta.npz")
    q, s1, y = m["q"], m["s1"], m["y"]
    va = m["q_touch_val"]
    tr = ~va
    q_true = np.load(wdir / "q_true_s1.npy")
    val_s1 = np.load(wdir / "val_s1.npy")
    is_s3_q = load_pickle(wdir / "q.pkl")["is_s3"].to_numpy()
    n_s1 = len(val_s1)
    log(f"train pairs {int(tr.sum()):,} (pos {int(y[tr].sum()):,}) | val pairs {int(va.sum()):,} (pos {int(y[va].sum()):,})")

    device = pick_device()
    params = {
        "objective": "binary:logistic", "eval_metric": ["logloss", "aucpr"],
        "tree_method": "hist", "device": device, "eta": 0.08, "max_depth": 9,
        "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
        "lambda": 2.0, "max_bin": 256,
    }
    with timer(f"build DMatrix ({device})"):
        dtr = xgb.QuantileDMatrix(np.ascontiguousarray(X[tr]), label=y[tr], feature_names=FEATURE_NAMES)
        dva = xgb.QuantileDMatrix(np.ascontiguousarray(X[va]), label=y[va], feature_names=FEATURE_NAMES, ref=dtr)
    with timer("train XGBoost"):
        bst = xgb.train(params, dtr, num_boost_round=rounds, evals=[(dtr, "train"), (dva, "val")],
                        early_stopping_rounds=60, verbose_eval=100)
    bst.save_model(str(wdir / "matcher.json"))
    del dtr

    p = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
    qv, sv = q[va], s1[va]
    orc = oracle_f05(n_s1, q_true, qv, sv)[val_s1].mean()
    log(f"validation S1: {int(val_s1.sum()):,} | oracle F0.5 ceiling of candidates: {orc:.5f}")

    best_t, best = 0.5, None
    for t in np.r_[np.arange(0.05, 0.96, 0.05), np.arange(0.30, 0.80, 0.01)]:
        r = evaluate(qv, sv, p, is_s3_q, q_true, n_s1, val_s1, float(t))
        if best is None or r["f05"] > best["f05"]:
            best_t, best = float(t), r
    log(f"best threshold {best_t:.2f}: " + ", ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v:,}"
                                                  for k, v in best.items()))
    imp = bst.get_score(importance_type="gain")
    top = sorted(imp.items(), key=lambda kv: -kv[1])[:15]
    log("top features by gain: " + ", ".join(f"{k}:{v:.0f}" for k, v in top))
    save_json({"threshold": best_t, "best_iteration": int(bst.best_iteration), "val": best,
               "oracle_ceiling": float(orc), "device": device}, wdir / "matcher_meta.json")

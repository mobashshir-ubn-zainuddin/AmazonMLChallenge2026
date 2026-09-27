"""
Stage 4 — train the matcher (2 stages) and tune the decision rule for macro F0.5.

Data split (by S1, honest for the per-S1 metric):
  * validation S1 = random `val_frac` of S1; every query that has a validation S1
    among its candidates (or as its true S1) is a validation query, so each
    validation S1 sees its COMPLETE set of competing queries
  * training queries = a random `train_query_frac` of the remaining queries

Stage 1: pair classifier on the pair features, trained as K fold models (split by
query). Every train pair is scored out-of-fold; every other pair (validation,
unused, test) gets the mean of the fold models.
Stage 2: pair features + group features (group.py) computed from the stage-1
probabilities of ALL candidate pairs -> second classifier.
The stage (1 or 2) with the better validation macro F0.5 is used for test.

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

from .common import load_pickle, log, save_json, split_work_dir, timer, free
from .features import FEATURE_NAMES, load_cands, open_matrix
from .group import GROUP_FEATURES, best_mask, group_features
from .metrics import oracle_f05, per_s1_f05

MAX_S2, MAX_S3 = 5, 6
# Features whose SCALE depends on how dense the S1 index / record pool is. The test set has
# ~23% more S2/S3 records per S1 and (for the US) an S1 index half the size of train, so
# these features shift between train and test. `--drop density` trains without them.
DENSITY_FEATURES = ["gap_best", "margin_other", "sn_gap", "sa_gap", "q_comb1", "q_comb2",
                    "n_q", "rank_s1", "n_s1", "s1_top_s2", "s1_top_s3"]
XGB_PARAMS = {
    "objective": "binary:logistic", "eval_metric": ["logloss", "aucpr"],
    "tree_method": "hist", "eta": 0.1, "max_depth": 9,
    "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
    "lambda": 2.0, "max_bin": 256,
}


def pick_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def query_fold(q: np.ndarray, folds: int) -> np.ndarray:
    return ((q.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(2 ** 32) % np.uint64(folds)).astype(np.int8)


def decide(q: np.ndarray, s1: np.ndarray, p: np.ndarray, is_s3_q: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean mask of pairs kept as final matches (global threshold rule)."""
    if len(q) == 0:
        return np.zeros(0, bool)
    best, _ = best_mask(q, p)
    return apply_caps(best & (p >= threshold), s1, q, p, is_s3_q)


def decide_expected(q: np.ndarray, s1: np.ndarray, p: np.ndarray, is_s3_q: np.ndarray, pmin: float) -> np.ndarray:
    """
    Per-S1 expected-F0.5 set selection.

    For each S1, take the records that chose it as their best S1 (p >= pmin), sorted by p.
    Keep the prefix of size k maximizing  E[F0.5] ≈ 1.25*sum_{i<=k} p_i / (0.25*E[#true] + k),
    with E[#true] = sum of p over all records that chose this S1; the empty set scores
    P(no true match) ≈ prod(1 - p_i). Picks "no match" when that is the better bet.
    """
    if len(q) == 0:
        return np.zeros(0, bool)
    best, _ = best_mask(q, p)
    bi = np.flatnonzero(best)
    n_s1 = int(s1.max()) + 1
    pc = np.clip(p, 0.0, 1.0 - 1e-6).astype(np.float64)
    exp_true = np.bincount(s1[bi], weights=pc[bi], minlength=n_s1)
    log_empty = np.bincount(s1[bi], weights=np.log1p(-pc[bi]), minlength=n_s1)
    idx = bi[p[bi] >= pmin]
    keep = np.zeros(len(q), bool)
    if len(idx) == 0:
        return keep
    o = np.lexsort((-p[idx], s1[idx]))
    ii = idx[o]
    ss = s1[ii]
    starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
    grp = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(ii)]))
    rank = np.arange(len(ii)) - starts[grp]
    cs = np.cumsum(pc[ii])
    cum = cs - np.r_[0.0, cs][starts][grp]
    ef = 1.25 * cum / (0.25 * exp_true[ss] + rank + 1)
    gmax = np.maximum.reduceat(ef, starts)
    first_max = np.minimum.reduceat(np.where(ef >= gmax[grp] - 1e-12, rank, 1 << 30), starts)
    empty_ef = np.exp(log_empty[ss[starts]])
    kstar = np.where(gmax > empty_ef, first_max + 1, 0)
    keep[ii[rank < kstar[grp]]] = True
    return apply_caps(keep, s1, q, p, is_s3_q)


def apply_caps(keep: np.ndarray, s1: np.ndarray, q: np.ndarray, p: np.ndarray, is_s3_q: np.ndarray) -> np.ndarray:
    """Per S1 keep at most MAX_S2 S2 and MAX_S3 S3 records (highest p first)."""
    idx = np.flatnonzero(keep)
    src = is_s3_q[q[idx]].astype(np.int8)
    o = np.lexsort((-p[idx], src, s1[idx]))
    key = s1[idx][o].astype(np.int64) * 2 + src[o]
    starts = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
    grp = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(key)]))
    rank = np.arange(len(key)) - starts[grp]
    cap = np.where(src[o] == 1, MAX_S3, MAX_S2)
    keep[idx[o][rank >= cap]] = False
    return keep


def evaluate(q, s1, p, is_s3_q, q_true, n_s1, s1_mask, threshold, keep=None) -> dict:
    if keep is None:
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


def sweep(q, s1, p, is_s3_q, q_true, n_s1, s1_mask) -> tuple[float, dict]:
    """Fast threshold search (argmax computed once, caps applied only to the final pick)."""
    best, _ = best_mask(q, p)
    bq, bs, bp = q[best], s1[best], p[best]
    correct = q_true[bq] == bs
    ntrue = np.bincount(q_true[q_true >= 0], minlength=n_s1).astype(np.float64)
    top_t, top_f = 0.5, -1.0
    for t in np.r_[np.arange(0.05, 0.96, 0.05), np.arange(0.30, 0.90, 0.01)]:
        m = bp >= t
        npred = np.bincount(bs[m], minlength=n_s1)
        tp = np.bincount(bs[m & correct], minlength=n_s1)
        den = 0.25 * ntrue + npred
        f = np.where(den > 0, 1.25 * tp / np.maximum(den, 1e-12), 1.0)[s1_mask].mean()
        if f > top_f:
            top_t, top_f = float(t), float(f)
    return top_t, evaluate(q, s1, p, is_s3_q, q_true, n_s1, s1_mask, top_t)


def sweep_expected(q, s1, p, is_s3_q, q_true, n_s1, s1_mask) -> tuple[float, dict]:
    """Tune the p floor of the expected-F0.5 rule on validation."""
    top_m, top = 0.3, None
    for pmin in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        r = evaluate(q, s1, p, is_s3_q, q_true, n_s1, s1_mask, None,
                     keep=decide_expected(q, s1, p, is_s3_q, pmin))
        if top is None or r["f05"] > top["f05"]:
            top_m, top = pmin, r
    return top_m, top


def predict_rows(bst: xgb.Booster, it: int, X: np.ndarray, rows=None, extra=None, chunk: int = 2_000_000,
                 cols=None) -> np.ndarray:
    """Predict for all rows (or selected sorted `rows`) of memmap X (optionally only feature
    columns `cols`), optionally hstacking `extra`."""
    n = X.shape[0] if rows is None else len(rows)
    out = np.empty(n, np.float32)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        blk = np.asarray(X[a:b] if rows is None else X[rows[a:b]])
        if cols is not None:
            blk = blk[:, cols]
        if extra is not None:
            blk = np.hstack([blk, extra[a:b] if rows is None else extra[rows[a:b]]])
        out[a:b] = bst.inplace_predict(np.ascontiguousarray(blk, dtype=np.float32), iteration_range=(0, it))
    return out


def fit(X_tr, y_tr, X_es, y_es, names, rounds, device) -> xgb.Booster:
    dtr = xgb.QuantileDMatrix(X_tr, label=y_tr, feature_names=names)
    des = xgb.QuantileDMatrix(X_es, label=y_es, feature_names=names, ref=dtr)
    bst = xgb.train({**XGB_PARAMS, "device": device}, dtr, num_boost_round=rounds,
                    evals=[(dtr, "train"), (des, "val")], early_stopping_rounds=50, verbose_eval=100)
    return bst


def run(work_dir: Path, rounds: int = 1500, train_query_frac: float = 0.35, val_frac: float = 0.02,
        folds: int = 2, seed: int = 11, drop: str = "none", no_stage2: bool = False) -> None:
    wdir = split_work_dir(work_dir, "train")
    device = pick_device()
    X = open_matrix(wdir)
    c = load_cands(wdir)
    cq, cs = c["q"], c["s1"]
    del c
    q_true = np.load(wdir / "q_true_s1.npy")
    qt = load_pickle(wdir / "q.pkl")
    is_s3_q = qt["is_s3"].to_numpy().astype(bool)
    q_nk = qt["name_key"].to_numpy()
    q_ad = qt["addr_clean"].to_numpy()
    del qt
    n_q = len(q_true)
    n_s1 = len(load_pickle(wdir / "s1.pkl"))
    y = (q_true[cq] == cs).astype(np.int8)

    rng = np.random.default_rng(seed)
    val_s1 = rng.random(n_s1) < val_frac
    touch_val = np.zeros(n_q, bool)
    touch_val[cq[val_s1[cs]]] = True
    touch_val[np.flatnonzero((q_true >= 0) & val_s1[np.maximum(q_true, 0)])] = True
    train_q = ~touch_val & (rng.random(n_q) < train_query_frac)
    fold_q = query_fold(np.arange(n_q), folds)
    tr_rows = np.flatnonzero(train_q[cq])
    va_rows = np.flatnonzero(touch_val[cq])
    es_rows = np.sort(rng.choice(va_rows, size=min(3_000_000, len(va_rows)), replace=False))
    log(f"S1 {n_s1:,} (val {int(val_s1.sum()):,}) | train pairs {len(tr_rows):,} (pos {int(y[tr_rows].sum()):,}) "
        f"| val pairs {len(va_rows):,} | early-stop pairs {len(es_rows):,} | device {device}")
    orc = float(oracle_f05(n_s1, q_true, cq[va_rows], cs[va_rows])[val_s1].mean())
    log(f"oracle F0.5 ceiling on validation S1: {orc:.5f}")

    # ---------------- stage 1: fold models ----------------
    dropped = set(DENSITY_FEATURES) if drop == "density" else set()
    cols = [i for i, n in enumerate(FEATURE_NAMES) if n not in dropped] if dropped else None
    names1 = [n for n in FEATURE_NAMES if n not in dropped]
    sub = (lambda A: A[:, cols]) if cols is not None else (lambda A: A)
    log(f"features: {len(names1)} used" + (f" (dropped density features: {sorted(dropped)})" if dropped else ""))
    X_es = np.ascontiguousarray(sub(np.asarray(X[es_rows])))
    y_es = y[es_rows]
    models, iters = [], []
    # Stage-1 models from a previous run (restored artifacts) are reused when they were
    # trained with the same split settings (same seed -> identical folds/validation).
    old = None
    if (wdir / "matcher_meta.json").is_file():
        from .common import load_json
        m = load_json(wdir / "matcher_meta.json")
        if (m.get("folds") == folds and m.get("val_frac") == val_frac and m.get("train_query_frac") == train_query_frac
                and m.get("drop", "none") == drop and all((wdir / f"stage1_fold{k}.json").is_file() for k in range(folds))):
            old = m
    for k in range(folds):
        if old is not None:
            bst = xgb.Booster()
            bst.load_model(str(wdir / f"stage1_fold{k}.json"))
            bst.set_param({"device": device})
            models.append(bst)
            iters.append(int(old["stage1_iterations"][k]))
            log(f"stage-1 fold {k}: reusing saved model ({iters[-1]} trees)")
            continue
        rows = tr_rows[fold_q[cq[tr_rows]] == k]
        with timer(f"stage-1 fold {k}: train on {len(rows):,} pairs"):
            Xk = np.ascontiguousarray(sub(np.asarray(X[rows])))
            bst = fit(Xk, y[rows], X_es, y_es, names1, rounds, device)
            del Xk
            free()
        bst.save_model(str(wdir / f"stage1_fold{k}.json"))
        models.append(bst)
        iters.append(int(bst.best_iteration) + 1)

    with timer("stage-1 scoring of all train pairs"):
        P = np.stack([predict_rows(m, it, X, cols=cols) for m, it in zip(models, iters)])
        p1 = P.mean(axis=0)
        in_tr = train_q[cq]
        fq = fold_q[cq]
        if folds == 2:
            p1[in_tr] = np.where(fq[in_tr] == 0, P[1][in_tr], P[0][in_tr])
        else:
            for k in range(folds):
                m = in_tr & (fq == k)
                p1[m] = np.delete(P, k, axis=0)[:, m].mean(axis=0)
        del P
        free()

    t1, r1 = sweep(cq[va_rows], cs[va_rows], p1[va_rows], is_s3_q, q_true, n_s1, val_s1)
    log(f"STAGE 1 validation @ {t1:.2f}: " + ", ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v:,}"
                                                     for k, v in r1.items()))

    # ---------------- stage 2: + group features ----------------
    if no_stage2:
        bst2, it2, t2, r2, p2_va = None, 0, t1, {"f05": -1.0}, None
    else:
        bst2, it2, t2, r2, p2_va = _stage2(X, sub, cols, cq, cs, p1, is_s3_q, n_q, n_s1, q_nk, q_ad, tr_rows,
                                           va_rows, es_rows, X_es, y, y_es, names1, rounds, device, wdir,
                                           q_true, val_s1)
    use2 = r2["f05"] > r1["f05"]
    if bst2 is not None:
        imp = bst2.get_score(importance_type="gain")
        log("stage-2 top features: " + ", ".join(f"{k}:{v:.0f}" for k, v in sorted(imp.items(), key=lambda kv: -kv[1])[:15]))
    _finish(locals())


def _stage2(X, sub, cols, cq, cs, p1, is_s3_q, n_q, n_s1, q_nk, q_ad, tr_rows, va_rows, es_rows, X_es, y, y_es,
            names1, rounds, device, wdir, q_true, val_s1):
    with timer("group features over all train pairs"):
        G = group_features(cq, cs, p1, is_s3_q, n_q, n_s1, q_nk, q_ad)
    names2 = names1 + GROUP_FEATURES
    with timer(f"stage-2: train on {len(tr_rows):,} pairs"):
        X2 = np.hstack([sub(np.asarray(X[tr_rows])), G[tr_rows]])
        X2_es = np.hstack([X_es, G[es_rows]])
        bst2 = fit(X2, y[tr_rows], X2_es, y_es, names2, rounds, device)
        del X2, X2_es
        free()
    bst2.save_model(str(wdir / "stage2.json"))
    it2 = int(bst2.best_iteration) + 1
    with timer("stage-2 validation scoring"):
        p2_va = predict_rows(bst2, it2, X, rows=va_rows, extra=G, cols=cols)
    t2, r2 = sweep(cq[va_rows], cs[va_rows], p2_va, is_s3_q, q_true, n_s1, val_s1)
    log(f"STAGE 2 validation @ {t2:.2f}: " + ", ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v:,}"
                                                     for k, v in r2.items()))
    return bst2, it2, t2, r2, p2_va


def _finish(v: dict) -> None:
    """Decision-rule selection + metadata (split out of run() for readability)."""
    cq, cs, va_rows, is_s3_q, q_true, n_s1, val_s1 = (v[k] for k in
        ("cq", "cs", "va_rows", "is_s3_q", "q_true", "n_s1", "val_s1"))
    use2, p1, p2_va, r1, r2, t1, t2 = (v[k] for k in ("use2", "p1", "p2_va", "r1", "r2", "t1", "t2"))

    # decision rule: global threshold vs per-S1 expected-F0.5 selection
    p_va = p2_va if use2 else p1[va_rows]
    r_thr = r2 if use2 else r1
    with timer("expected-F0.5 decision sweep"):
        pmin, r_exp = sweep_expected(cq[va_rows], cs[va_rows], p_va, is_s3_q, q_true, n_s1, val_s1)
    log(f"EXPECTED-F0.5 decision (pmin {pmin:.2f}): " + ", ".join(
        f"{kk}={vv:.5f}" if isinstance(vv, float) else f"{kk}={vv:,}" for kk, vv in r_exp.items()))
    use_exp = r_exp["f05"] > r_thr["f05"]

    import time as _time
    matched = q_true >= 0
    meta = {
        "folds": v["folds"], "stage1_iterations": v["iters"], "stage2_iteration": v["it2"],
        "stage1": {"threshold": t1, "val": r1}, "stage2": {"threshold": t2, "val": r2},
        "expected": {"pmin": pmin, "val": r_exp},
        "use_stage2": bool(use2), "threshold": t2 if use2 else t1,
        "decision": "expected" if use_exp else "threshold", "pmin": pmin,
        "val_f05": max(r_thr["f05"], r_exp["f05"]),
        "oracle_ceiling": v["orc"], "device": v["device"],
        "feature_names": v["names1"], "group_features": GROUP_FEATURES,
        "kept_idx": v["cols"], "drop": v["drop"], "no_stage2": bool(v["no_stage2"]),
        "val_frac": v["val_frac"], "train_query_frac": v["train_query_frac"],
        "train_prior": float(matched.mean()), "train_matches_per_s1": float(matched.sum() / n_s1),
        "model_id": _time.strftime("%Y%m%d-%H%M%S"),
    }
    save_json(meta, v["wdir"] / "matcher_meta.json")
    log(f"USING STAGE {'2' if use2 else '1'} + {meta['decision']} decision "
        f"(val F0.5 {meta['val_f05']:.5f}; threshold {meta['threshold']:.2f}, pmin {pmin:.2f})")

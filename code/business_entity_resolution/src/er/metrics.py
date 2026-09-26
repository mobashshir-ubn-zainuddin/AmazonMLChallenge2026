"""
Challenge metric: macro-average over S1 of per-S1 F0.5.

Per S1 with t true matches, p predictions, tp correct:
    F0.5 = 1.25*tp / (0.25*t + p)        (equivalent to 1.25PR/(0.25P+R))
    t == 0 and p == 0  -> 1.0  (correct singleton)
    t == 0 and p  > 0  -> 0.0
Everything is vectorized with bincount over integer indices.
"""

from __future__ import annotations

import numpy as np


def per_s1_f05(n_s1: int, q_true_s1: np.ndarray, pred_q: np.ndarray, pred_s1: np.ndarray) -> np.ndarray:
    """F0.5 for every S1 row given predicted (q, s1) pairs (each q at most once)."""
    ntrue = np.bincount(q_true_s1[q_true_s1 >= 0], minlength=n_s1).astype(np.float64)
    npred = np.bincount(pred_s1, minlength=n_s1).astype(np.float64)
    correct = q_true_s1[pred_q] == pred_s1
    tp = np.bincount(pred_s1[correct], minlength=n_s1).astype(np.float64)
    denom = 0.25 * ntrue + npred
    f = np.ones(n_s1, dtype=np.float64)
    nz = denom > 0
    f[nz] = 1.25 * tp[nz] / denom[nz]
    return f


def oracle_f05(n_s1: int, q_true_s1: np.ndarray, cand_q: np.ndarray, cand_s1: np.ndarray) -> np.ndarray:
    """Per-S1 F0.5 of a perfect matcher restricted to the candidate set (the ceiling)."""
    hit = q_true_s1[cand_q] == cand_s1
    return per_s1_f05(n_s1, q_true_s1, cand_q[hit], cand_s1[hit])


def candidate_report(n_s1: int, q_true_s1: np.ndarray, cand_q: np.ndarray, cand_s1: np.ndarray,
                     is_s3: np.ndarray, s1_mask: np.ndarray | None = None) -> dict:
    """Retrieval recall + oracle ceiling + candidate volume (per S1, the submission's view)."""
    if s1_mask is None:
        s1_mask = np.ones(n_s1, dtype=bool)
    hit = q_true_s1[cand_q] == cand_s1
    matched = q_true_s1 >= 0
    in_scope = matched & s1_mask[np.maximum(q_true_s1, 0)]
    found = np.zeros(len(q_true_s1), dtype=bool)
    found[cand_q[hit]] = True
    rec_all = found[in_scope].mean() if in_scope.any() else float("nan")
    rec_s2 = found[in_scope & ~is_s3].mean()
    rec_s3 = found[in_scope & is_s3].mean()
    ntrue = np.bincount(q_true_s1[matched], minlength=n_s1)
    nfound = np.bincount(q_true_s1[found], minlength=n_s1)
    with np.errstate(invalid="ignore", divide="ignore"):
        s1_rec = np.where(ntrue > 0, nfound / np.maximum(ntrue, 1), np.nan)
    counts = np.bincount(cand_s1, minlength=n_s1)[s1_mask]
    orc = oracle_f05(n_s1, q_true_s1, cand_q, cand_s1)[s1_mask]
    return {
        "pair_recall": float(rec_all),
        "s2_recall": float(rec_s2),
        "s3_recall": float(rec_s3),
        "macro_s1_recall(matched S1 only)": float(np.nanmean(s1_rec[s1_mask])),
        "oracle_f05_ceiling": float(orc.mean()),
        "cand_per_s1_mean": float(counts.mean()),
        "cand_per_s1_median": float(np.median(counts)),
        "cand_per_s1_p90": float(np.quantile(counts, 0.90)),
        "cand_per_s1_p95": float(np.quantile(counts, 0.95)),
        "cand_per_s1_p99": float(np.quantile(counts, 0.99)),
        "cand_per_s1_max": int(counts.max()) if len(counts) else 0,
        "zero_candidate_s1": int((counts == 0).sum()),
        "total_pairs": int(len(cand_q)),
    }


def format_report(rep: dict) -> str:
    return "\n".join(f"  {k:36s} {v:.6f}" if isinstance(v, float) else f"  {k:36s} {v:,}" for k, v in rep.items())

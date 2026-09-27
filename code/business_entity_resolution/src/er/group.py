"""
Stage-2 group features: context that a single (query, S1) pair cannot see.

Given stage-1 probabilities p for ALL candidate pairs, compute for each pair:
  * query side: is this the query's best S1, the query's best / second-best p,
    margin of this pair over the query's best alternative
  * S1 side (over the pairs that are their query's best choice): strongest
    OTHER query claiming this S1, total claim mass of the others, number of
    other confident claimants (overall and per source S2 / S3)
These let the stage-2 model learn e.g. "an S1 attracting only one weak record
is probably a singleton" or "this record is better explained elsewhere".
All numpy group operations; ~100M pairs take about a minute.
"""

from __future__ import annotations

import numpy as np

GROUP_FEATURES = [
    "p1", "g_is_best", "g_margin_q", "g_q_best", "g_q_second",
    "g_s1_max_other", "g_s1_sum_other", "g_s1_cnt_other", "g_s1_cnt_s2_other", "g_s1_cnt_s3_other",
]


def best_mask(q: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(is_best per pair, lexsort order by (q, -p))."""
    order = np.lexsort((-p, q))
    first = np.ones(len(q), bool)
    first[1:] = q[order][1:] != q[order][:-1]
    best = np.zeros(len(q), bool)
    best[order[first]] = True
    return best, order


def group_features(q: np.ndarray, s1: np.ndarray, p: np.ndarray, is_s3_q: np.ndarray,
                   n_q: int, n_s1: int) -> np.ndarray:
    """float32 matrix (n_pairs, len(GROUP_FEATURES))."""
    n = len(q)
    p = p.astype(np.float32)
    best, order = best_mask(q, p)

    # query side
    q_best = np.zeros(n_q, np.float32)
    q_best[q[best]] = p[best]
    qo = q[order]
    # second-best = element right after the first of each group (if same q)
    first_idx = np.flatnonzero(np.r_[True, qo[1:] != qo[:-1]])
    nxt = first_idx + 1
    ok = nxt < n
    ok[ok] = qo[nxt[ok]] == qo[first_idx[ok]]
    q_second = np.zeros(n_q, np.float32)
    q_second[qo[first_idx[ok]]] = p[order[nxt[ok]]]
    margin_q = p - np.where(best, q_second[q], q_best[q])

    # S1 side over "best" pairs
    conf = best & (p >= 0.5)
    w = np.where(best, p, 0).astype(np.float32)
    s1_sum = np.bincount(s1, weights=w, minlength=n_s1).astype(np.float32)
    s1_cnt = np.bincount(s1[conf], minlength=n_s1)
    s3 = is_s3_q[q]
    s1_cnt_s2 = np.bincount(s1[conf & ~s3], minlength=n_s1)
    s1_cnt_s3 = np.bincount(s1[conf & s3], minlength=n_s1)

    bi = np.flatnonzero(best)
    o = np.lexsort((-p[bi], s1[bi]))
    bs = s1[bi][o]
    bp = p[bi][o]
    starts = np.flatnonzero(np.r_[True, bs[1:] != bs[:-1]]) if len(bs) else np.zeros(0, int)
    top1 = np.zeros(n_s1, np.float32)
    top2 = np.zeros(n_s1, np.float32)
    top1_pair = np.full(n_s1, -1, np.int64)
    top1[bs[starts]] = bp[starts]
    top1_pair[bs[starts]] = bi[o][starts]
    nx = starts + 1
    ok = nx < len(bs)
    ok[ok] = bs[nx[ok]] == bs[starts[ok]]
    top2[bs[starts[ok]]] = bp[nx[ok]]
    is_self_top = top1_pair[s1] == np.arange(n)
    max_other = np.where(is_self_top, top2[s1], top1[s1])

    out = np.empty((n, len(GROUP_FEATURES)), np.float32)
    out[:, 0] = p
    out[:, 1] = best
    out[:, 2] = margin_q
    out[:, 3] = q_best[q]
    out[:, 4] = q_second[q]
    out[:, 5] = max_other
    out[:, 6] = s1_sum[s1] - w
    out[:, 7] = s1_cnt[s1] - conf
    out[:, 8] = s1_cnt_s2[s1] - (conf & ~s3)
    out[:, 9] = s1_cnt_s3[s1] - (conf & s3)
    return out

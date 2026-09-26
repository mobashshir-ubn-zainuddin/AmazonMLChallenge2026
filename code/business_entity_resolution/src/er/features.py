"""
Stage 3 — pair features for (query, S1) candidate pairs.

Three feature groups:
  * retrieval / competition: cosines, gap to the query's best S1, rank of this S1
    among the query's candidates, how many queries picked this S1 as their best
    (S1-side competition), rank of this query among the S1's candidates
  * string similarity (rapidfuzz, C++ + multithreaded) on several name keys and
    the normalized address
  * address-number agreement (street number, shared numbers)
All features are country-agnostic (no country one-hot), so they transfer to
France, which is absent from training.

On train this stage writes a feature matrix for a query subset (all queries
touching validation S1 + a random fraction of the rest). On test, features are
computed chunk-by-chunk inside predict.py so the full matrix never sits in RAM.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from .common import load_pickle, log, split_work_dir, timer, free

FEATURE_NAMES = [
    "sn", "sa", "comb", "gap_best", "margin_other", "sn_gap", "sa_gap", "q_comb1", "q_comb2",
    "rank_q", "n_q", "rank_s1", "n_s1", "s1_top_s2", "s1_top_s3", "is_top",
    "nk_ratio", "nk_tset", "nk_tsort", "nk_partial", "nk_jw", "nc_ratio", "nc_tset", "sq_ratio",
    "ad_ratio", "ad_tset", "ad_tsort", "ad_partial",
    "len_nk_q", "len_nk_s", "len_ad_q", "len_ad_s",
    "num_first_in_q", "num_first_eq", "num_shared", "num_q", "num_s",
    "q_addr_empty", "q_native", "is_s3", "w_name",
]


class Tables:
    """Record-level columns needed for features, as numpy object/number arrays."""

    def __init__(self, wdir: Path):
        s1 = load_pickle(wdir / "s1.pkl")
        q = load_pickle(wdir / "q.pkl")
        self.n_s1 = len(s1)
        self.n_q = len(q)
        self.s1_id = s1["entity_id"].to_numpy()
        self.q_id = q["entity_id"].to_numpy()
        self.s_nk = s1["name_key"].to_numpy()
        self.q_nk = q["name_key"].to_numpy()
        self.s_nc = s1["name_clean"].to_numpy()
        self.q_nc = q["name_clean"].to_numpy()
        self.s_sq = np.array([x.replace(" ", "") for x in self.s_nk], dtype=object)
        self.q_sq = np.array([x.replace(" ", "") for x in self.q_nk], dtype=object)
        self.s_ad = s1["addr_clean"].to_numpy()
        self.q_ad = q["addr_clean"].to_numpy()
        self.q_addr_empty = q["addr_empty"].to_numpy()
        self.q_native = q["name_native"].to_numpy()
        self.is_s3 = q["is_s3"].to_numpy()
        self.s_nums = np.load(wdir / "s1_nums.npy")
        self.q_nums = np.load(wdir / "q_nums.npy")
        del s1, q
        free()


def load_cands(wdir: Path) -> dict:
    z = np.load(wdir / "cands.npz")
    return {k: z[k] for k in z.files}


def pair_context(c: dict, t: Tables, w_name: float) -> dict:
    """Competition features over the FULL candidate set (cheap numpy group ops)."""
    q, s1 = c["q"], c["s1"]
    sn = c["sn"].astype(np.float32)
    sa = c["sa"].astype(np.float32)
    name_empty = np.array([len(x) == 0 for x in t.q_nk])
    wq = np.where(t.q_addr_empty, 1.0, w_name).astype(np.float32)
    wq[name_empty] = 0.0
    w = wq[q]
    comb = w * sn + (1 - w) * sa

    order = np.lexsort((-comb, q))
    starts = np.r_[0, np.flatnonzero(np.diff(q[order])) + 1]
    grp = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(q)]))
    rank_q = np.empty(len(q), np.int32)
    rank_q[order] = np.arange(len(q)) - starts[grp]
    n_q = np.bincount(q, minlength=t.n_q)[q]

    order = np.lexsort((-comb, s1))
    s1o = s1[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s1o)) + 1] if len(s1o) else np.zeros(0, int)
    grp = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(s1)]))
    rank_s1 = np.empty(len(s1), np.int32)
    rank_s1[order] = np.arange(len(s1)) - starts[grp]
    n_s1 = np.bincount(s1, minlength=t.n_s1)[s1]

    is_top = rank_q == 0
    s3 = t.is_s3[q]
    top_s2 = np.bincount(s1[is_top & ~s3], minlength=t.n_s1)[s1]
    top_s3 = np.bincount(s1[is_top & s3], minlength=t.n_s1)[s1]

    c1 = np.nan_to_num(c["q_comb1"].astype(np.float32))[q]
    c2 = np.nan_to_num(c["q_comb2"].astype(np.float32))[q]
    return {
        "sn": sn, "sa": sa, "comb": comb, "w_name": w,
        "gap_best": comb - c1,
        "margin_other": comb - np.where(is_top, c2, c1),
        "sn_gap": sn - np.nan_to_num(c["q_sn1"].astype(np.float32))[q],
        "sa_gap": sa - np.nan_to_num(c["q_sa1"].astype(np.float32))[q],
        "q_comb1": c1, "q_comb2": c2,
        "rank_q": rank_q, "n_q": n_q, "rank_s1": rank_s1, "n_s1": n_s1,
        "s1_top_s2": top_s2, "s1_top_s3": top_s3, "is_top": is_top,
    }


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def string_features(qi: np.ndarray, si: np.ndarray, t: Tables) -> dict:
    """String/number features for one chunk of pairs."""
    f = {}
    a, b = t.q_nk[qi], t.s_nk[si]
    f["nk_ratio"] = _cp(a, b, fuzz.ratio)
    f["nk_tset"] = _cp(a, b, fuzz.token_set_ratio)
    f["nk_tsort"] = _cp(a, b, fuzz.token_sort_ratio)
    f["nk_partial"] = _cp(a, b, fuzz.partial_ratio)
    f["nk_jw"] = _cp(a, b, JaroWinkler.normalized_similarity)
    f["len_nk_q"] = np.fromiter((len(x) for x in a), np.float32, len(a))
    f["len_nk_s"] = np.fromiter((len(x) for x in b), np.float32, len(b))
    a, b = t.q_nc[qi], t.s_nc[si]
    f["nc_ratio"] = _cp(a, b, fuzz.ratio)
    f["nc_tset"] = _cp(a, b, fuzz.token_set_ratio)
    f["sq_ratio"] = _cp(t.q_sq[qi], t.s_sq[si], fuzz.ratio)
    a, b = t.q_ad[qi], t.s_ad[si]
    f["ad_ratio"] = _cp(a, b, fuzz.ratio)
    f["ad_tset"] = _cp(a, b, fuzz.token_set_ratio)
    f["ad_tsort"] = _cp(a, b, fuzz.token_sort_ratio)
    f["ad_partial"] = _cp(a, b, fuzz.partial_ratio)
    f["len_ad_q"] = np.fromiter((len(x) for x in a), np.float32, len(a))
    f["len_ad_s"] = np.fromiter((len(x) for x in b), np.float32, len(b))

    qn, sn = t.q_nums[qi], t.s_nums[si]
    qv, sv = qn >= 0, sn >= 0
    eq = (qn[:, :, None] == sn[:, None, :]) & qv[:, :, None] & sv[:, None, :]
    f["num_shared"] = eq.any(axis=2).sum(axis=1).astype(np.float32)
    f["num_first_in_q"] = eq[:, :, 0].any(axis=1).astype(np.float32)
    f["num_first_eq"] = (sv[:, 0] & qv[:, 0] & (qn[:, 0] == sn[:, 0])).astype(np.float32)
    f["num_q"] = qv.sum(axis=1).astype(np.float32)
    f["num_s"] = sv.sum(axis=1).astype(np.float32)

    f["q_addr_empty"] = t.q_addr_empty[qi].astype(np.float32)
    f["q_native"] = t.q_native[qi].astype(np.float32)
    f["is_s3"] = t.is_s3[qi].astype(np.float32)
    return f


def build_matrix(idx: np.ndarray, c: dict, ctx: dict, t: Tables, chunk: int = 1_000_000) -> np.ndarray:
    """Feature matrix (len(idx), len(FEATURE_NAMES)) float32 for the selected pair indices."""
    X = np.empty((len(idx), len(FEATURE_NAMES)), dtype=np.float32)
    col = {n: i for i, n in enumerate(FEATURE_NAMES)}
    for start in range(0, len(idx), chunk):
        sel = idx[start:start + chunk]
        sf = string_features(c["q"][sel], c["s1"][sel], t)
        for n in FEATURE_NAMES:
            X[start:start + len(sel), col[n]] = sf[n] if n in sf else ctx[n][sel]
        if (start // chunk) % 10 == 0:
            log(f"    features {start + len(sel):,}/{len(idx):,}")
    return X


def run(work_dir: Path, split: str, train_query_frac: float = 0.2, val_frac: float = 0.1, seed: int = 11) -> None:
    if split != "train":
        log("features for test are computed chunk-wise inside `predict`; nothing to do here")
        return
    wdir = split_work_dir(work_dir, split)
    params = load_pickle(wdir / "retrieval_params.pkl")
    with timer("load tables + candidates"):
        t = Tables(wdir)
        c = load_cands(wdir)
        q_true = np.load(wdir / "q_true_s1.npy")
    with timer("competition context"):
        ctx = pair_context(c, t, params["w_name"])

    rng = np.random.default_rng(seed)
    val_s1 = rng.random(t.n_s1) < val_frac
    touch_val = np.zeros(t.n_q, bool)
    touch_val[c["q"][val_s1[c["s1"]]]] = True
    touch_val[np.flatnonzero((q_true >= 0) & val_s1[np.maximum(q_true, 0)])] = True
    take_q = touch_val | (rng.random(t.n_q) < train_query_frac)
    idx = np.flatnonzero(take_q[c["q"]])
    log(f"feature rows: {len(idx):,} pairs from {int(take_q.sum()):,} queries "
        f"({int(touch_val.sum()):,} touch validation S1)")
    with timer("string features"):
        X = build_matrix(idx, c, ctx, t)
    y = (q_true[c["q"][idx]] == c["s1"][idx]).astype(np.int8)
    np.save(wdir / "X.npy", X)
    np.savez(wdir / "pairs_meta.npz", idx=idx, q=c["q"][idx], s1=c["s1"][idx], y=y,
             val_pair=val_s1[c["s1"][idx]], q_touch_val=touch_val[c["q"][idx]])
    np.save(wdir / "val_s1.npy", val_s1)
    log(f"saved X {X.shape}, positives {int(y.sum()):,}")

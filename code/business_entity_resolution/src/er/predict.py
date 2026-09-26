"""
Stage 5 — score test candidates and write the two submission files.

    output/candidate_pairs.tsv   every candidate the matcher scored, per S1
    output/matching_results.tsv  final matches, per S1 (subset of candidates)

Features are computed and scored chunk-by-chunk, so the full test feature
matrix is never materialized.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xgboost as xgb

from .common import load_json, load_pickle, log, split_work_dir, timer, free
from .features import FEATURE_NAMES, Tables, load_cands, pair_context, string_features
from .train import decide, pick_device


def write_grouped(path: Path, header: str, s1_ids: np.ndarray, pair_s1: np.ndarray, pair_q_ids: np.ndarray) -> None:
    """One line per S1 (in test-file order) with comma-joined IDs; empty when none."""
    order = np.argsort(pair_s1, kind="stable")
    ids = pair_q_ids[order]
    ends = np.cumsum(np.bincount(pair_s1, minlength=len(s1_ids)))
    starts = ends - np.bincount(pair_s1, minlength=len(s1_ids))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header + "\n")
        for i, sid in enumerate(s1_ids):
            a, b = starts[i], ends[i]
            f.write(sid + "\t" + (",".join(ids[a:b]) if b > a else "") + "\n")


def run(work_dir: Path, out_dir: Path, chunk: int = 2_000_000) -> None:
    tr_dir = split_work_dir(work_dir, "train")
    wdir = split_work_dir(work_dir, "test")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = load_json(tr_dir / "matcher_meta.json")
    params = load_pickle(wdir / "retrieval_params.pkl")
    bst = xgb.Booster()
    bst.load_model(str(tr_dir / "matcher.json"))
    bst.set_param({"device": pick_device()})
    it = (0, meta["best_iteration"] + 1)

    with timer("load test tables + candidates"):
        t = Tables(wdir)
        c = load_cands(wdir)
    with timer("competition context"):
        ctx = pair_context(c, t, params["w_name"])
    n = len(c["q"])
    p = np.empty(n, dtype=np.float32)
    col = {name: i for i, name in enumerate(FEATURE_NAMES)}
    with timer(f"score {n:,} candidate pairs"):
        for start in range(0, n, chunk):
            sel = np.arange(start, min(start + chunk, n))
            sf = string_features(c["q"][sel], c["s1"][sel], t)
            X = np.empty((len(sel), len(FEATURE_NAMES)), dtype=np.float32)
            for name in FEATURE_NAMES:
                X[:, col[name]] = sf[name] if name in sf else ctx[name][sel]
            p[sel] = bst.inplace_predict(X, iteration_range=it)
            log(f"    scored {sel[-1] + 1:,}/{n:,}")
    del ctx
    free()

    keep = decide(c["q"], c["s1"], p, t.is_s3, meta["threshold"])
    log(f"threshold {meta['threshold']:.2f}: {int(keep.sum()):,} matched pairs; "
        f"S1 with >=1 match: {len(np.unique(c['s1'][keep])):,}/{t.n_s1:,}")
    np.savez(wdir / "test_scores.npz", p=p, keep=keep)

    with timer("write candidate_pairs.tsv"):
        write_grouped(out_dir / "candidate_pairs.tsv", "source1_entity_id\tcandidate_entity_ids",
                      t.s1_id, c["s1"], t.q_id[c["q"]])
    with timer("write matching_results.tsv"):
        write_grouped(out_dir / "matching_results.tsv", "source1_entity_id\tmatched_entity_ids",
                      t.s1_id, c["s1"][keep], t.q_id[c["q"][keep]])
    log(f"outputs written to {out_dir}")

"""
Stage 5 — score test candidates and write the two submission files.

    <out>/candidate_pairs.tsv   every candidate the matcher scored, per S1
    <out>/matching_results.tsv  final matches, per S1 (subset of candidates)

Uses the test feature memmap written by `features --split test`:
stage-1 probability = mean of the fold models; if validation chose stage 2,
group features are computed from those probabilities and the stage-2 model
gives the final probability. Per-pair probabilities are saved so the threshold
can be changed later without re-scoring (`predict --threshold X`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xgboost as xgb

from .common import load_json, load_pickle, log, split_work_dir, timer, free
from .features import load_cands, open_matrix
from .group import group_features
from .train import decide, pick_device, predict_rows


def write_grouped(path: Path, header: str, s1_ids: np.ndarray, pair_s1: np.ndarray, pair_q_ids: np.ndarray) -> None:
    """One line per S1 (in test-file order) with comma-joined IDs; empty when none."""
    order = np.argsort(pair_s1, kind="stable")
    ids = pair_q_ids[order]
    counts = np.bincount(pair_s1, minlength=len(s1_ids))
    ends = np.cumsum(counts)
    starts = ends - counts
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header + "\n")
        for i, sid in enumerate(s1_ids):
            a, b = starts[i], ends[i]
            f.write(sid + "\t" + (",".join(ids[a:b]) if b > a else "") + "\n")


def _load(path: Path) -> xgb.Booster:
    bst = xgb.Booster()
    bst.load_model(str(path))
    bst.set_param({"device": pick_device()})
    return bst


def run(work_dir: Path, out_dir: Path, threshold: float | None = None, write_candidates: bool = True) -> None:
    tr_dir = split_work_dir(work_dir, "train")
    wdir = split_work_dir(work_dir, "test")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = load_json(tr_dir / "matcher_meta.json")
    s1 = load_pickle(wdir / "s1.pkl")
    q = load_pickle(wdir / "q.pkl")
    s1_ids = s1["entity_id"].to_numpy()
    q_ids = q["entity_id"].to_numpy()
    is_s3_q = q["is_s3"].to_numpy().astype(bool)
    del s1, q
    free()
    c = load_cands(wdir)
    cq, cs = c["q"], c["s1"]
    del c

    scores = wdir / "test_scores.npz"
    if threshold is not None and scores.is_file():
        p = np.load(scores)["p"]
        log(f"re-using saved test probabilities ({scores.name})")
    else:
        X = open_matrix(wdir)
        with timer(f"stage-1 scoring of {len(cq):,} test pairs ({meta['folds']} fold models)"):
            p1 = np.zeros(len(cq), np.float32)
            for k, it in enumerate(meta["stage1_iterations"]):
                p1 += predict_rows(_load(tr_dir / f"stage1_fold{k}.json"), it, X)
            p1 /= meta["folds"]
        p = p1
        if meta["use_stage2"]:
            with timer("group features + stage-2 scoring"):
                G = group_features(cq, cs, p1, is_s3_q, len(q_ids), len(s1_ids))
                p = predict_rows(_load(tr_dir / "stage2.json"), meta["stage2_iteration"], X, extra=G)
                del G
        np.savez(scores, p=p, p1=p1)

    t = meta["threshold"] if threshold is None else threshold
    keep = decide(cq, cs, p, is_s3_q, t)
    log(f"stage {'2' if meta['use_stage2'] else '1'}, threshold {t:.2f}: {int(keep.sum()):,} matched pairs; "
        f"S1 with >=1 match: {len(np.unique(cs[keep])):,}/{len(s1_ids):,}")

    if write_candidates:
        with timer("write candidate_pairs.tsv"):
            write_grouped(out_dir / "candidate_pairs.tsv", "source1_entity_id\tcandidate_entity_ids",
                          s1_ids, cs, q_ids[cq])
    with timer("write matching_results.tsv"):
        write_grouped(out_dir / "matching_results.tsv", "source1_entity_id\tmatched_entity_ids",
                      s1_ids, cs[keep], q_ids[cq[keep]])
    log(f"outputs written to {out_dir}")

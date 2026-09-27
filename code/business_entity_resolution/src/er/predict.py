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
from .train import decide, decide_expected, pick_device, predict_rows


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


def run(work_dir: Path, out_dir: Path, threshold: float | None = None, write_candidates: bool = True,
        prob: str = "auto", prior_test: str | None = None, out_name: str = "matching_results.tsv") -> None:
    """
    prob:       auto (stage chosen on validation) | p1 (stage-1 fold mean) | p2 (stage-2)
    prior_test: None | "auto" | a float. Prior-shift correction: the test set has more
                distractor records than train (5.75 vs 4.68 S2/S3 records per S1), so the share
                of records that truly match is lower. Probabilities are re-calibrated with
                logit(p) += logit(prior_test) - logit(prior_train) before thresholding.
                "auto" estimates prior_test = train matches-per-S1 * n_S1 / n_records.
    """
    tr_dir = split_work_dir(work_dir, "train")
    wdir = split_work_dir(work_dir, "test")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = load_json(tr_dir / "matcher_meta.json")
    s1 = load_pickle(wdir / "s1.pkl")
    q = load_pickle(wdir / "q.pkl")
    s1_ids = s1["entity_id"].to_numpy()
    q_ids = q["entity_id"].to_numpy()
    is_s3_q = q["is_s3"].to_numpy().astype(bool)
    q_nk = q["name_key"].to_numpy()
    q_ad = q["addr_clean"].to_numpy()
    del s1, q
    free()
    c = load_cands(wdir)
    cq, cs = c["q"], c["s1"]
    del c

    scores = wdir / "test_scores.npz"
    reuse = scores.is_file() and (threshold is not None or prob != "auto" or prior_test is not None)
    if reuse:
        z = np.load(scores)
        saved_id = str(z["model_id"]) if "model_id" in z.files else None
        if saved_id != meta.get("model_id"):
            log("saved test probabilities belong to a different model; re-scoring")
            reuse = False
    if reuse:
        p1 = z["p1"]
        p = z["p"]
        log(f"re-using saved test probabilities ({scores.name})")
    else:
        X = open_matrix(wdir)
        with timer(f"stage-1 scoring of {len(cq):,} test pairs ({meta['folds']} fold models)"):
            p1 = np.zeros(len(cq), np.float32)
            cols = meta.get("kept_idx")
            for k, it in enumerate(meta["stage1_iterations"]):
                p1 += predict_rows(_load(tr_dir / f"stage1_fold{k}.json"), it, X, cols=cols)
            p1 /= meta["folds"]
        p = p1
        if meta["use_stage2"] and not meta.get("no_stage2"):
            with timer("group features + stage-2 scoring"):
                G = group_features(cq, cs, p1, is_s3_q, len(q_ids), len(s1_ids), q_nk, q_ad)
                p = predict_rows(_load(tr_dir / "stage2.json"), meta["stage2_iteration"], X, extra=G, cols=cols)
                del G
        np.savez(scores, p=p, p1=p1, model_id=str(meta.get("model_id")))

    use2 = meta["use_stage2"] if prob == "auto" else prob == "p2"
    pp = p if use2 else p1
    stage_name = "2" if use2 else "1"
    if threshold is None and prob != "auto":
        threshold = meta["stage2" if use2 else "stage1"]["threshold"]
    if prior_test is not None:
        n_s1, n_q = len(s1_ids), len(q_ids)
        prior_train = meta.get("train_prior", 7638365 / 10320219)
        pt = (min(0.95, meta.get("train_matches_per_s1", 3.4613) * n_s1 / n_q)
              if prior_test == "auto" else float(prior_test))
        shift = float(np.log(pt / (1 - pt)) - np.log(prior_train / (1 - prior_train)))
        pc = np.clip(pp.astype(np.float64), 1e-7, 1 - 1e-7)
        pp = (1 / (1 + np.exp(-(np.log(pc / (1 - pc)) + shift)))).astype(np.float32)
        log(f"prior-shift correction: train prior {prior_train:.3f} -> test prior {pt:.3f} (logit shift {shift:+.3f})")
    best = np.zeros(len(pp), bool)
    order = np.lexsort((-pp, cq))
    first = np.r_[True, cq[order][1:] != cq[order][:-1]]
    best[order[first]] = True
    log(f"diagnostic: mean best-p per record {pp[best].mean():.3f}; records with best-p>=0.5: "
        f"{(pp[best] >= 0.5).mean():.3f} (train matched share {meta.get('train_prior', 0.740):.3f})")
    p = pp
    method = meta.get("decision", "threshold") if threshold is None else "threshold"
    if method == "expected":
        keep = decide_expected(cq, cs, p, is_s3_q, meta["pmin"])
        rule = f"expected-F0.5 (pmin {meta['pmin']:.2f})"
    else:
        t = meta["threshold"] if threshold is None else threshold
        keep = decide(cq, cs, p, is_s3_q, t)
        rule = f"threshold {t:.2f}"
    log(f"stage {stage_name}, {rule}: {int(keep.sum()):,} matched pairs; "
        f"S1 with >=1 match: {len(np.unique(cs[keep])):,}/{len(s1_ids):,}")

    if write_candidates:
        with timer("write candidate_pairs.tsv"):
            write_grouped(out_dir / "candidate_pairs.tsv", "source1_entity_id\tcandidate_entity_ids",
                          s1_ids, cs, q_ids[cq])
    with timer("write matching_results.tsv"):
        write_grouped(out_dir / out_name, "source1_entity_id\tmatched_entity_ids",
                      s1_ids, cs[keep], q_ids[cq[keep]])
    log(f"outputs written to {out_dir}")

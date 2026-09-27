"""
Command line entry point for the v2 pipeline.

    python -m business_entity_resolution.src.er.run <stage> --data DATA --work WORK --out OUT [options]

Stages (each caches its outputs in WORK and skips work that is already done):
    sample     build a small local dev sample (optional)
    restore    --from DIR   copy saved artifacts (a previous run's OUT/artifacts) into WORK
    prepare    --split train|test     normalize raw TSVs (+ learn translit on train)
    retrieve   --split train|test     GPU top-K S1 per S2/S3 record -> candidates
    features   --split train|test     features for ALL candidate pairs (disk memmap)
    train                             stage-1 fold models + stage-2 group model, threshold for macro F0.5
    predict                           test scoring -> OUT/matching_results.tsv + candidate_pairs.tsv
    persist    [--what all|model|train|test]   copy artifacts to OUT/artifacts (kept by Kaggle)
    package    --team NAME            build NAME_submission.zip in the required layout
    all                               prepare/retrieve/features train+test, train, predict, persist
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import log


def main() -> None:
    p = argparse.ArgumentParser(description="Business entity resolution v2 pipeline")
    p.add_argument("stage", choices=["sample", "restore", "prepare", "retrieve", "features", "train",
                                     "predict", "persist", "package", "all"])
    p.add_argument("--data", type=Path, default=Path("dataset"), help="folder containing train/ and test/")
    p.add_argument("--work", type=Path, default=Path("work"), help="cache folder for intermediate artifacts")
    p.add_argument("--out", type=Path, default=Path("output"), help="folder for the submission TSVs")
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--workers", type=int, default=0, help="CPU processes (0 = all cores)")
    # sample / restore / persist / package
    p.add_argument("--fraction", type=float, default=0.02)
    p.add_argument("--from", dest="from_dir", type=Path, default=Path("/kaggle/input"))
    p.add_argument("--what", default="all", choices=["all", "model", "train", "test"])
    p.add_argument("--team", default="team")
    # retrieval
    p.add_argument("--k-name", type=int, default=3)
    p.add_argument("--k-addr", type=int, default=3)
    p.add_argument("--k-comb", type=int, default=8)
    p.add_argument("--w-name", type=float, default=0.5)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--device", default="auto", help="auto | cuda | cpu | numpy")
    p.add_argument("--gpu-budget-gb", type=float, default=3.5)
    # training
    p.add_argument("--train-query-frac", type=float, default=0.35,
                   help="fraction of non-validation queries used for training")
    p.add_argument("--val-frac", type=float, default=0.02, help="fraction of S1 held out for validation")
    p.add_argument("--folds", type=int, default=2)
    p.add_argument("--rounds", type=int, default=1500)
    # predict
    p.add_argument("--threshold", type=float, default=None,
                   help="override the tuned threshold (re-uses saved test probabilities)")
    p.add_argument("--no-candidates", action="store_true", help="do not rewrite candidate_pairs.tsv")
    a = p.parse_args()

    if a.stage == "sample":
        from . import make_sample
        make_sample.run(a.data, a.out, a.fraction)
        return
    if a.stage == "restore":
        from . import artifacts
        artifacts.restore(a.from_dir, a.work)
        return
    if a.stage == "persist":
        from . import artifacts
        artifacts.persist(a.work, a.out, a.what)
        return
    if a.stage == "package":
        from . import package
        package.run(a.out, a.team)
        return

    def prepare(split):
        from . import prepare as m
        m.run(a.data, a.work, split, a.workers)

    def retrieve(split):
        from . import retrieve as m
        m.run(a.work, split, a.k_name, a.k_addr, a.k_comb, a.w_name, a.dim, a.device, a.gpu_budget_gb, a.workers)

    def features(split):
        from . import features as m
        m.run(a.work, split)

    def train():
        from . import train as m
        m.run(a.work, a.rounds, a.train_query_frac, a.val_frac, a.folds)

    def predict():
        from . import predict as m
        m.run(a.work, a.out, a.threshold, not a.no_candidates)

    if a.stage == "prepare":
        prepare(a.split)
    elif a.stage == "retrieve":
        retrieve(a.split)
    elif a.stage == "features":
        features(a.split)
    elif a.stage == "train":
        train()
    elif a.stage == "predict":
        predict()
    elif a.stage == "all":
        from . import artifacts
        for split in ("train", "test"):
            prepare(split)
            retrieve(split)
            features(split)
        train()
        predict()
        artifacts.persist(a.work, a.out, "all")
    log("finished")


if __name__ == "__main__":
    main()

"""
Command line entry point for the v2 pipeline.

    python -m business_entity_resolution.src.er.run <stage> --data DATA --work WORK [options]

Stages (run in order; each caches its outputs in WORK):
    sample     build a small local dev sample (optional)
    prepare    --split train|test     normalize raw TSVs (+ learn translit on train)
    retrieve   --split train|test     GPU top-K S1 per S2/S3 record -> candidates
    features   --split train|test     pair features for candidates
    train                             XGBoost matcher + threshold tuned for macro F0.5
    predict                           test scoring -> output/matching_results.tsv + candidate_pairs.tsv
    all                               prepare/retrieve/features for train+test, train, predict
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import log


def main() -> None:
    p = argparse.ArgumentParser(description="Business entity resolution v2 pipeline")
    p.add_argument("stage", choices=["sample", "prepare", "retrieve", "features", "train", "predict", "all"])
    p.add_argument("--data", type=Path, default=Path("dataset"), help="folder containing train/ and test/")
    p.add_argument("--work", type=Path, default=Path("work"), help="cache folder for intermediate artifacts")
    p.add_argument("--out", type=Path, default=Path("output"), help="folder for the submission TSVs")
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--workers", type=int, default=0, help="CPU processes (0 = all cores)")
    # sample
    p.add_argument("--fraction", type=float, default=0.02)
    # retrieval
    p.add_argument("--k-name", type=int, default=3)
    p.add_argument("--k-addr", type=int, default=3)
    p.add_argument("--k-comb", type=int, default=8)
    p.add_argument("--w-name", type=float, default=0.5)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--device", default="auto", help="auto | cuda | cpu | numpy")
    p.add_argument("--gpu-budget-gb", type=float, default=3.5)
    # training
    p.add_argument("--train-query-frac", type=float, default=0.12,
                   help="fraction of train queries (besides all validation-touching ones) used for training")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--rounds", type=int, default=1500)
    a = p.parse_args()

    if a.stage == "sample":
        from . import make_sample
        make_sample.run(a.data, a.out, a.fraction)
        return

    def prepare(split):
        from . import prepare as m
        m.run(a.data, a.work, split, a.workers)

    def retrieve(split):
        from . import retrieve as m
        m.run(a.work, split, a.k_name, a.k_addr, a.k_comb, a.w_name, a.dim, a.device, a.gpu_budget_gb, a.workers)

    def features(split):
        from . import features as m
        m.run(a.work, split, a.train_query_frac, a.val_frac)

    def train():
        from . import train as m
        m.run(a.work, a.rounds)

    def predict():
        from . import predict as m
        m.run(a.work, a.out)

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
        for split in ("train", "test"):
            prepare(split)
            retrieve(split)
            features(split)
        train()
        predict()
    log("finished")


if __name__ == "__main__":
    main()

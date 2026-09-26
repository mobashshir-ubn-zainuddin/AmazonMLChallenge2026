"""
Build a small closed-world dev sample with the same file layout as the real data.

Takes a fraction of S1 (deterministic by row), ALL their true S2/S3 matches, and
the same fraction of unmatched S2/S3 records. Writes <out_dir>/train/*.tsv and a
fake <out_dir>/test/ (the sample's sources without GT) so every stage, including
test output writing, can be smoke-tested locally in minutes.

Note: the sample S1 index is smaller than the real one, so retrieval recall here
is optimistic. Use it for correctness checks, not for final numbers.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .common import SOURCE_FILES, find_split_dir, log, read_ground_truth, read_source


def _write(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar=None, encoding="utf-8")


def run(data_dir: Path, out_dir: Path, fraction: float = 0.02, seed: int = 7) -> None:
    src = find_split_dir(data_dir, "train")
    f = SOURCE_FILES["train"]
    rng = np.random.default_rng(seed)
    gt = read_ground_truth(src / f["gt"])
    keep_gt = gt[rng.random(len(gt)) < fraction]
    matched = set(",".join(keep_gt["matched_entity_ids"]).split(",")) - {""}
    all_matched = set(",".join(gt["matched_entity_ids"]).split(",")) - {""}
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "test").mkdir(parents=True, exist_ok=True)
    s1 = read_source(src / f["s1"])
    s1 = s1[s1["entity_id"].isin(set(keep_gt["source1_entity_id"]))]
    _write(s1, out_dir / "train" / f["s1"])
    _write(s1, out_dir / "test" / "test_source1.tsv")
    for k in ("s2", "s3"):
        df = read_source(src / f[k])
        is_m = df["entity_id"].isin(all_matched).to_numpy()
        take = df["entity_id"].isin(matched).to_numpy() | (~is_m & (rng.random(len(df)) < fraction))
        _write(df[take], out_dir / "train" / f[k])
        _write(df[take], out_dir / "test" / f"test_source{k[1]}.tsv")
        log(f"{k}: {int(take.sum()):,} rows")
    _write(keep_gt, out_dir / "train" / f["gt"])
    log(f"sample: S1 {len(s1):,}, GT rows {len(keep_gt):,} -> {out_dir}")

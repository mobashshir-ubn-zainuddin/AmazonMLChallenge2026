"""
Shared utilities for the v2 entity-resolution pipeline (package `er`).

Pipeline direction (see Implementation_Plan_v2.txt):
    every S2/S3 record ("query", q) retrieves its top-K S1 records, a pair
    classifier scores (q, s1) pairs, and each q is assigned to at most one S1.

Artifacts of every stage are written under <work_dir>/<split>/ so stages can
be re-run independently (important on Kaggle where sessions are limited).
"""

from __future__ import annotations

import csv
import gc
import json
import os
import pickle
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_FILES = {
    "train": {
        "s1": "train_source1.tsv",
        "s2": "train_source2.tsv",
        "s3": "train_source3.tsv",
        "gt": "train_ground_truth.tsv",
    },
    "test": {
        "s1": "test_source1.tsv",
        "s2": "test_source2.tsv",
        "s3": "test_source3.tsv",
    },
}


# ---------------------------------------------------------------------
# Logging / timing
# ---------------------------------------------------------------------

_T0 = time.time()


def log(msg: str) -> None:
    """Print a timestamped progress message (flushed, so Kaggle shows it live)."""
    print(f"[{time.time() - _T0:8.1f}s] {msg}", flush=True)


@contextmanager
def timer(name: str):
    """Log the wall time of a block."""
    start = time.time()
    log(f"{name} ...")
    yield
    log(f"{name} done in {time.time() - start:.1f}s")


def free() -> None:
    gc.collect()


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

def find_split_dir(data_dir: Path, split: str) -> Path:
    """
    Locate the folder that contains <split>_source1.tsv.

    Accepts dataset/dataset/<split>, dataset/<split>, or any nesting under
    data_dir (Kaggle datasets often add an extra folder level).
    """
    target = SOURCE_FILES[split]["s1"]
    for cand in [data_dir / split, data_dir / "dataset" / split, data_dir / "dataset" / "dataset" / split, data_dir]:
        if (cand / target).is_file():
            return cand
    # Kaggle mounts inputs through symlinks; Path.rglob does not follow them, os.walk can.
    roots = [data_dir] + ([Path("/kaggle/input")] if Path("/kaggle/input").is_dir() else [])
    for root in roots:
        for dirpath, _, filenames in os.walk(root, followlinks=True):
            if target in filenames:
                if root != data_dir:
                    log(f"NOTE: {target} not under {data_dir}; using {dirpath}")
                return Path(dirpath)
    raise FileNotFoundError(f"{target} not found under {data_dir} (or /kaggle/input)")


def split_work_dir(work_dir: Path, split: str) -> Path:
    path = work_dir / split
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------

def read_source(path: Path) -> pd.DataFrame:
    """
    Read a raw source TSV exactly as the scorer does: tab separated, no quote
    handling (names contain quote characters), everything as strings.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
        na_filter=False,
        encoding="utf-8",
    )
    df.columns = [c.strip() for c in df.columns]
    return df[["entity_id", "business_name", "business_address", "country"]]


def read_ground_truth(path: Path) -> pd.DataFrame:
    """Ground truth: matched_entity_ids is COMMA separated (never ';')."""
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
        na_filter=False,
        encoding="utf-8",
    )


def save_pickle(obj, path: Path) -> None:
    if isinstance(obj, pd.DataFrame):
        # plain object/bool/number columns only, so pickles load under any pandas version
        obj = pd.DataFrame({c: (obj[c].to_numpy(dtype=object) if obj[c].dtype.kind not in "biuf" else obj[c].to_numpy())
                            for c in obj.columns})
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# Record tables
# ---------------------------------------------------------------------

def load_tables(wdir: Path, columns=None):
    """
    Load the normalized S1 table and the concatenated query table (S2 then S3).

    Query index q in [0, n_s2) is S2, [n_s2, n_s2 + n_s3) is S3.
    """
    s1 = load_pickle(wdir / "s1.pkl")
    q = load_pickle(wdir / "q.pkl")
    if columns is not None:
        s1 = s1[[c for c in columns if c in s1.columns]]
        q = q[[c for c in columns + ["is_s3"] if c in q.columns]]
    return s1, q


def true_s1_for_queries(wdir: Path) -> np.ndarray:
    """int32 array: q index -> true S1 row index, -1 when q matches no S1."""
    return np.load(wdir / "q_true_s1.npy")


def env_threads() -> int:
    return max(1, os.cpu_count() or 1)

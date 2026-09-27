"""
Persist / restore the expensive pipeline artifacts.

persist:  <work>/...  ->  <out>/artifacts/...   (Kaggle keeps /kaggle/working/output after a run)
restore:  <root>/**/artifacts/...  ->  <work>/...  (e.g. a previous notebook version's output
          attached as an input) so retrieval and training can be skipped.

What is saved (~4 GB): transliteration dictionaries, train+test candidate sets and
retrieval parameters/report, all trained models + decision metadata, test probabilities.
Normalized tables and feature matrices are NOT saved (they are cheap to rebuild with
`prepare` / `features`, and the feature matrices are ~19 GB each).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .common import log

FILES = {
    ".": ["translit.json"],
    "train": ["cands.npz", "retrieval_params.pkl", "retrieval_report.pkl", "matcher_meta.json",
              "stage1_fold0.json", "stage1_fold1.json", "stage1_fold2.json", "stage2.json"],
    "test": ["cands.npz", "retrieval_params.pkl", "test_scores.npz"],
}


def persist(work_dir: Path, out_dir: Path, what: str = "all") -> None:
    """Copy artifacts into <out>/artifacts. what: all | model | train | test."""
    dest = out_dir / "artifacts"
    groups = {"all": list(FILES), "model": ["."], "train": [".", "train"], "test": ["test"]}[what]
    if what == "model":
        items = [(".", "translit.json")] + [("train", f) for f in FILES["train"] if f.endswith(".json")]
    else:
        items = [(g, f) for g in groups for f in FILES[g]]
    n = 0
    for sub, name in items:
        src = work_dir / sub / name
        if src.is_file():
            (dest / sub).mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / sub / name)
            n += 1
            log(f"  persisted {sub}/{name} ({src.stat().st_size / 1e6:.1f} MB)")
    log(f"persisted {n} artifact file(s) to {dest}")


def restore(root: Path, work_dir: Path) -> bool:
    """Find an `artifacts` folder containing translit.json under root and copy it into work_dir."""
    found = None
    for dirpath, _, files in os.walk(root, followlinks=True):
        if Path(dirpath).name == "artifacts" and "translit.json" in files:
            found = Path(dirpath)
            break
    if found is None:
        log(f"no artifacts found under {root}; running the full pipeline")
        return False
    n = 0
    for sub, names in FILES.items():
        for name in names:
            src = found / sub / name
            if src.is_file():
                (work_dir / sub).mkdir(parents=True, exist_ok=True)
                dst = work_dir / sub / name
                if not dst.is_file():
                    shutil.copy2(src, dst)
                    n += 1
                    log(f"  restored {sub}/{name}")
    log(f"restored {n} artifact file(s) from {found}")
    return True

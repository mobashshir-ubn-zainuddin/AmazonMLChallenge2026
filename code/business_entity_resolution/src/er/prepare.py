"""
Stage 1 — prepare: raw TSV -> normalized record tables (+ GT arrays on train).

Outputs in <work_dir>/<split>/:
    s1.pkl          DataFrame: entity_id, country, name_clean, name_key, addr_clean, name_native, addr_empty
    q.pkl           same columns for S2 followed by S3, plus is_s3
    s1_nums.npy     int64 (n_s1, 6) address numbers (-1 pad)
    q_nums.npy      int64 (n_q, 6)
    q_true_s1.npy   (train only) int32 q -> true S1 row, -1 if none
And <work_dir>/translit.json (learned on train, reused for test).
"""

from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from . import text
from .common import (
    SOURCE_FILES, find_split_dir, log, read_ground_truth, read_source,
    save_json, load_json, save_pickle, split_work_dir, timer, env_threads,
)

_TRANSLIT = None
_SEG = None


def _init(translit, seg):
    global _TRANSLIT, _SEG
    _TRANSLIT, _SEG = translit, seg


def _norm_chunk(args):
    names, addrs = args
    out_nc, out_nk, out_ac, out_nums, out_native = [], [], [], [], []
    for n, a in zip(names, addrs):
        nc, nk = text.normalize_name(n, _TRANSLIT)
        ac, nums = text.normalize_address(a, _TRANSLIT, _SEG)
        out_nc.append(nc)
        out_nk.append(nk)
        out_ac.append(ac)
        out_nums.append(nums)
        out_native.append(bool(n) and any(ch.isalpha() for ch in n) and not text.has_latin_letter(n))
    return out_nc, out_nk, out_ac, text.numbers_to_array(out_nums), out_native


def normalize_frame(df: pd.DataFrame, translit, seg, workers: int) -> tuple[pd.DataFrame, np.ndarray]:
    names = df["business_name"].tolist()
    addrs = df["business_address"].tolist()
    step = 20_000
    chunks = [(names[i:i + step], addrs[i:i + step]) for i in range(0, len(names), step)]
    nc, nk, ac, nums, native = [], [], [], [], []
    if workers > 1:
        with Pool(workers, initializer=_init, initargs=(translit, seg)) as pool:
            results = pool.imap(_norm_chunk, chunks, chunksize=1)
            for r in results:
                nc += r[0]; nk += r[1]; ac += r[2]; nums.append(r[3]); native += r[4]
    else:
        _init(translit, seg)
        for c in chunks:
            r = _norm_chunk(c)
            nc += r[0]; nk += r[1]; ac += r[2]; nums.append(r[3]); native += r[4]
    out = pd.DataFrame({
        "entity_id": df["entity_id"].to_numpy(),
        "country": df["country"].str.strip().str.casefold().to_numpy(),
        "name_clean": nc,
        "name_key": nk,
        "addr_clean": ac,
        "name_native": np.array(native, dtype=bool),
    })
    out["addr_empty"] = out["addr_clean"].eq("")
    num_arr = np.concatenate(nums) if nums else np.full((0, 6), -1, np.int64)
    return out, num_arr


def _gt_pairs(gt: pd.DataFrame) -> pd.DataFrame:
    g = gt[gt["matched_entity_ids"].str.len() > 0][["source1_entity_id", "matched_entity_ids"]].copy()
    g["m"] = g["matched_entity_ids"].str.split(",")
    g = g.explode("m")[["source1_entity_id", "m"]]
    g["m"] = g["m"].str.strip()
    return g[g["m"] != ""]


def learn_translit(raw: dict, gt: pd.DataFrame, work_dir: Path, max_pairs: int = 1_500_000) -> dict:
    """Learn native-script name/segment dictionaries from TRAIN ground-truth pairs only."""
    pairs = _gt_pairs(gt)
    s1 = raw["s1"].set_index("entity_id")
    q = pd.concat([raw["s2"], raw["s3"]]).set_index("entity_id")
    # Only pairs whose match name has no Latin letter are useful for names.
    qn = q["business_name"]
    native_mask = ~qn.str.contains(r"[A-Za-z]", regex=True) & qn.str.len().gt(0)
    native_ids = set(qn.index[native_mask.to_numpy()])
    pn = pairs[pairs["m"].isin(native_ids)].head(max_pairs)
    log(f"translit: {len(pn):,} native-name GT pairs")
    name_dict = text.learn_name_translit(
        s1.loc[pn["source1_entity_id"], "business_name"].tolist(),
        q.loc[pn["m"], "business_name"].tolist(),
    )
    qa = q["business_address"]
    addr_native = qa.str.contains(r"[ऀ-෿]", regex=True)
    addr_ids = set(qa.index[addr_native.to_numpy()])
    pa = pairs[pairs["m"].isin(addr_ids)].head(max_pairs)
    log(f"translit: {len(pa):,} native-address GT pairs")
    seg_dict = text.learn_segment_translit(
        s1.loc[pa["source1_entity_id"], "business_address"].tolist(),
        q.loc[pa["m"], "business_address"].tolist(),
    )
    log(f"translit: learned {len(name_dict):,} name tokens, {len(seg_dict):,} address segments")
    obj = {"name": name_dict, "segment": seg_dict}
    save_json(obj, work_dir / "translit.json")
    return obj


def run(data_dir: Path, work_dir: Path, split: str, workers: int = 0) -> None:
    workers = workers or env_threads()
    src_dir = find_split_dir(data_dir, split)
    wdir = split_work_dir(work_dir, split)
    files = SOURCE_FILES[split]

    raw = {}
    with timer(f"read raw {split} sources from {src_dir}"):
        for k in ("s1", "s2", "s3"):
            raw[k] = read_source(src_dir / files[k])
            log(f"  {k}: {len(raw[k]):,} rows")

    gt = None
    if split == "train":
        gt = read_ground_truth(src_dir / files["gt"])
        if (work_dir / "translit.json").is_file():
            # restored from a previous run: keep it so normalization matches saved artifacts
            tr = load_json(work_dir / "translit.json")
            log(f"reusing existing translit.json ({len(tr['name']):,} name tokens)")
        else:
            with timer("learn transliteration dictionaries"):
                tr = learn_translit(raw, gt, work_dir)
    else:
        path = work_dir / "translit.json"
        tr = load_json(path) if path.is_file() else {"name": {}, "segment": {}}
        if not path.is_file():
            log("WARNING: translit.json not found (run prepare on train first); continuing without it")

    with timer("normalize S1"):
        s1, s1_nums = normalize_frame(raw["s1"], tr["name"], tr["segment"], workers)
    save_pickle(s1, wdir / "s1.pkl")
    np.save(wdir / "s1_nums.npy", s1_nums)

    parts, nums = [], []
    for k in ("s2", "s3"):
        with timer(f"normalize {k.upper()}"):
            df, arr = normalize_frame(raw[k], tr["name"], tr["segment"], workers)
        df["is_s3"] = k == "s3"
        parts.append(df)
        nums.append(arr)
        del raw[k]
    q = pd.concat(parts, ignore_index=True)
    save_pickle(q, wdir / "q.pkl")
    np.save(wdir / "q_nums.npy", np.concatenate(nums))
    log(f"S1 {len(s1):,} | queries {len(q):,} (S2 {int((~q.is_s3).sum()):,}, S3 {int(q.is_s3.sum()):,})")

    if gt is not None:
        pairs = _gt_pairs(gt)
        s1_pos = pd.Index(s1["entity_id"]).get_indexer(pairs["source1_entity_id"])
        q_pos = pd.Index(q["entity_id"]).get_indexer(pairs["m"])
        ok = (s1_pos >= 0) & (q_pos >= 0)
        if not ok.all():
            log(f"WARNING: {int((~ok).sum()):,} GT pairs reference unknown IDs")
        true_s1 = np.full(len(q), -1, dtype=np.int32)
        true_s1[q_pos[ok]] = s1_pos[ok]
        np.save(wdir / "q_true_s1.npy", true_s1)
        log(f"GT pairs: {int(ok.sum()):,}; unmatched queries: {int((true_s1 < 0).sum()):,}")

"""
Stage 2 — reverse retrieval: every S2/S3 record (query) -> top-K S1 records.

Channels (all exact brute force inside the query's country):
    name     : cosine of name_key vectors               -> top k_name
    address  : cosine of addr_clean vectors             -> top k_addr
    combined : w*name + (1-w)*address (w=1 if the query
               has no address, w=0 if it has no name)   -> top k_comb
The union per query is the candidate set (this is what candidate_pairs.tsv
contains, inverted to S1 rows).

Backends: torch on 1+ CUDA GPUs (S1 matrix is column-sharded across GPUs),
torch CPU, or numpy (local smoke tests without torch).

Output <work_dir>/<split>/cands.npz:
    q, s1 (int32), sn, sa (float16)                 one row per candidate pair
    q_comb1, q_comb2, q_sn1, q_sa1 (float16, per q)  best/second-best scores
"""

from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .common import env_threads, load_pickle, log, save_pickle, split_work_dir, timer
from .metrics import candidate_report, format_report
from .vectors import HashedTfidf

try:
    import torch
except Exception:  # torch not installed locally -> numpy backend
    torch = None

_VN = None
_VA = None


def _init_worker(vn, va):
    global _VN, _VA
    _VN, _VA = vn, va


def _vectorize(args):
    names, addrs = args
    return _VN.transform(names), _VA.transform(addrs)


class Backend:
    """Holds the S1 shards of one country and runs scored top-K for query blocks."""

    def __init__(self, device: str):
        self.kind = "numpy"
        self.devices = ["cpu"]
        if torch is not None and device != "numpy":
            if device in ("auto", "cuda") and torch.cuda.is_available():
                self.kind = "cuda"
                self.devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            elif device in ("auto", "cpu", "torch-cpu"):
                self.kind = "torch-cpu"
        log(f"retrieval backend: {self.kind} {self.devices}")

    def load(self, Sn: np.ndarray, Sa: np.ndarray) -> None:
        n = Sn.shape[0]
        bounds = np.linspace(0, n, len(self.devices) + 1).astype(int)
        self.offsets = bounds[:-1]
        self.shards = []
        for d, (a, b) in zip(self.devices, zip(bounds[:-1], bounds[1:])):
            if self.kind == "numpy":
                self.shards.append((Sn[a:b].astype(np.float32), Sa[a:b].astype(np.float32)))
            else:
                dt = torch.float16 if self.kind == "cuda" else torch.float32
                self.shards.append((torch.from_numpy(Sn[a:b]).to(d, dt), torch.from_numpy(Sa[a:b]).to(d, dt)))
        self.shard_sizes = bounds[1:] - bounds[:-1]

    def free(self):
        self.shards = []
        if self.kind == "cuda":
            torch.cuda.empty_cache()

    def batch_size(self, budget_gb: float) -> int:
        biggest = max(int(self.shard_sizes.max()), 1)
        bytes_per_row = biggest * (2 if self.kind == "cuda" else 4) * 4
        return int(np.clip(budget_gb * 1e9 / bytes_per_row, 32, 8192))

    def topk(self, qn, qa, wn, k):
        """Returns per channel (idx, sn, sa) numpy arrays of shape (B, k_total_over_shards)."""
        outs = {c: [] for c in ("n", "a", "c")}
        for (Sn, Sa), off, size in zip(self.shards, self.offsets, self.shard_sizes):
            kk = {c: min(k[c], int(size)) for c in k}
            if self.kind == "numpy":
                sn = qn.astype(np.float32) @ Sn.T
                sa = qa.astype(np.float32) @ Sa.T
                comb = sn * wn[:, None] + sa * (1 - wn)[:, None]
                for c, M in (("n", sn), ("a", sa), ("c", comb)):
                    idx = np.argpartition(-M, kk[c] - 1, axis=1)[:, :kk[c]]
                    outs[c].append((idx + off, np.take_along_axis(sn, idx, 1), np.take_along_axis(sa, idx, 1)))
            else:
                dev = Sn.device
                tq = torch.from_numpy(qn).to(dev, Sn.dtype)
                ta = torch.from_numpy(qa).to(dev, Sn.dtype)
                tw = torch.from_numpy(wn).to(dev, Sn.dtype)[:, None]
                sn = tq @ Sn.T
                sa = ta @ Sa.T
                comb = sn * tw + sa * (1 - tw)
                for c, M in (("n", sn), ("a", sa), ("c", comb)):
                    idx = torch.topk(M, kk[c], dim=1, sorted=False).indices
                    outs[c].append((
                        (idx + int(off)).cpu().numpy(),
                        torch.gather(sn, 1, idx).float().cpu().numpy(),
                        torch.gather(sa, 1, idx).float().cpu().numpy(),
                    ))
                del sn, sa, comb
        merged = {}
        for c, parts in outs.items():
            merged[c] = tuple(np.concatenate([p[i] for p in parts], axis=1) for i in range(3))
        return merged


def _select(idx, sn, sa, key, k):
    """Keep the k best columns per row by `key` (descending), sorted."""
    k = min(k, key.shape[1])
    order = np.argsort(-key, axis=1, kind="stable")[:, :k]
    t = lambda a: np.take_along_axis(a, order, 1)
    return t(idx), t(sn), t(sa), t(key)


def run(work_dir: Path, split: str, k_name: int = 3, k_addr: int = 3, k_comb: int = 8,
        w_name: float = 0.5, dim: int = 1024, device: str = "auto",
        gpu_budget_gb: float = 3.5, workers: int = 0) -> None:
    wdir = split_work_dir(work_dir, split)
    workers = max(1, (workers or env_threads()) - 1)
    s1 = load_pickle(wdir / "s1.pkl")[["country", "name_key", "addr_clean"]]
    q = load_pickle(wdir / "q.pkl")[["country", "name_key", "addr_clean", "addr_empty", "is_s3"]]
    n_q = len(q)

    with timer("fit hashed TF-IDF on S1"):
        vn = HashedTfidf(dim).fit(s1["name_key"].tolist())
        va = HashedTfidf(dim, seed=29).fit(s1["addr_clean"].tolist())

    backend = Backend(device)
    s1_country = s1["country"].to_numpy()
    q_country = q["country"].to_numpy()
    q_names = q["name_key"].to_numpy()
    q_addrs = q["addr_clean"].to_numpy()
    q_addr_empty = q["addr_empty"].to_numpy()

    out_q, out_s1, out_sn, out_sa = [], [], [], []
    stats = {k: np.full(n_q, np.nan, dtype=np.float16) for k in ("q_comb1", "q_comb2", "q_sn1", "q_sa1")}
    k = {"n": k_name, "a": k_addr, "c": max(k_comb, 2)}

    with Pool(workers, initializer=_init_worker, initargs=(vn, va)) as pool:
        for country in sorted(set(q_country)):
            s1_rows = np.flatnonzero(s1_country == country)
            q_rows = np.flatnonzero(q_country == country)
            if len(s1_rows) == 0:
                log(f"country '{country}': {len(q_rows):,} queries but no S1 -> no candidates")
                continue
            with timer(f"country '{country}': vectorize {len(s1_rows):,} S1"):
                names = s1["name_key"].to_numpy()[s1_rows]
                addrs = s1["addr_clean"].to_numpy()[s1_rows]
                step = 50_000
                parts = list(pool.imap(_vectorize, [(names[i:i + step].tolist(), addrs[i:i + step].tolist())
                                                    for i in range(0, len(names), step)]))
                Sn = np.concatenate([p[0] for p in parts])
                Sa = np.concatenate([p[1] for p in parts])
                del parts
                backend.load(Sn, Sa)
                del Sn, Sa
            B = backend.batch_size(gpu_budget_gb)
            chunks = [q_rows[i:i + B] for i in range(0, len(q_rows), B)]
            jobs = ((q_names[c].tolist(), q_addrs[c].tolist()) for c in chunks)
            with timer(f"country '{country}': retrieve {len(q_rows):,} queries (batch {B})"):
                for ci, (rows, (Qn, Qa)) in enumerate(zip(chunks, pool.imap(_vectorize, jobs, chunksize=1))):
                    name_empty = ~Qn.any(axis=1)
                    wn = np.where(q_addr_empty[rows], 1.0, w_name).astype(np.float32)
                    wn[name_empty] = 0.0
                    m = backend.topk(Qn, Qa, wn, k)
                    n_idx, n_sn, n_sa, n_key = _select(*m["n"], m["n"][1], k_name)
                    a_idx, a_sn, a_sa, a_key = _select(*m["a"], m["a"][2], k_addr)
                    c_comb = m["c"][1] * wn[:, None] + m["c"][2] * (1 - wn)[:, None]
                    c_idx, c_sn, c_sa, c_key = _select(*m["c"], c_comb, max(k_comb, 2))
                    stats["q_comb1"][rows] = c_key[:, 0]
                    stats["q_comb2"][rows] = c_key[:, 1] if c_key.shape[1] > 1 else 0
                    stats["q_sn1"][rows] = n_key[:, 0]
                    stats["q_sa1"][rows] = a_key[:, 0]
                    c_idx, c_sn, c_sa = c_idx[:, :k_comb], c_sn[:, :k_comb], c_sa[:, :k_comb]
                    idx = np.concatenate([c_idx, n_idx, a_idx], axis=1)
                    sn = np.concatenate([c_sn, n_sn, a_sn], axis=1)
                    sa = np.concatenate([c_sa, n_sa, a_sa], axis=1)
                    order = np.argsort(idx, axis=1, kind="stable")
                    idx = np.take_along_axis(idx, order, 1)
                    keep = np.ones_like(idx, dtype=bool)
                    keep[:, 1:] = idx[:, 1:] != idx[:, :-1]
                    rr = np.repeat(rows[:, None], idx.shape[1], axis=1)
                    out_q.append(rr[keep].astype(np.int32))
                    out_s1.append(s1_rows[idx[keep]].astype(np.int32))
                    out_sn.append(np.take_along_axis(sn, order, 1)[keep].astype(np.float16))
                    out_sa.append(np.take_along_axis(sa, order, 1)[keep].astype(np.float16))
                    if ci % 200 == 0:
                        log(f"    {min((ci + 1) * B, len(q_rows)):,}/{len(q_rows):,}")
            backend.free()

    cq = np.concatenate(out_q) if out_q else np.zeros(0, np.int32)
    order = np.argsort(cq, kind="stable")
    cands = {
        "q": cq[order],
        "s1": np.concatenate(out_s1)[order] if out_s1 else np.zeros(0, np.int32),
        "sn": np.concatenate(out_sn)[order] if out_sn else np.zeros(0, np.float16),
        "sa": np.concatenate(out_sa)[order] if out_sa else np.zeros(0, np.float16),
        **stats,
    }
    np.savez(wdir / "cands.npz", **cands)
    save_pickle({"k_name": k_name, "k_addr": k_addr, "k_comb": k_comb, "w_name": w_name, "dim": dim},
                wdir / "retrieval_params.pkl")
    log(f"saved {len(cands['q']):,} candidate pairs ({len(cands['q']) / max(n_q, 1):.2f} per query)")

    true_path = wdir / "q_true_s1.npy"
    if true_path.is_file():
        rep = candidate_report(len(s1), np.load(true_path), cands["q"], cands["s1"], q["is_s3"].to_numpy())
        log("RETRIEVAL REPORT (train ground truth):\n" + format_report(rep))
        save_pickle(rep, wdir / "retrieval_report.pkl")

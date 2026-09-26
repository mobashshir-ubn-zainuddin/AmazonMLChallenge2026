"""
Stage 2 — reverse retrieval: every S2/S3 record (query) -> top-K S1 records.

Channels (all exact brute force inside the query's country):
    name     : cosine of name_key vectors               -> top k_name
    address  : cosine of addr_clean vectors             -> top k_addr
    combined : w*name + (1-w)*address (w=1 if the query
               has no address, w=0 if it has no name)   -> top k_comb
The union per query is the candidate set (this is what candidate_pairs.tsv
contains, inverted to S1 rows).

Backends: torch on 1+ CUDA GPUs (S1 matrix is column-sharded across GPUs and
the shards run in parallel threads), torch CPU, or numpy (local smoke tests).

Memory discipline (Kaggle has ~30 GB RAM):
  * the vectorizer process pool is created BEFORE the big query table is loaded,
    so forked workers do not slowly copy it (copy-on-write + refcounts)
  * query/S1 vectorization uses a bounded prefetch queue (never more than a few
    chunks in flight), unlike Pool.imap which buffers results without limit
  * S1 vectors are written straight into one preallocated float16 array
  * each country's candidates are checkpointed; a rerun skips finished countries

Output <work_dir>/<split>/cands.npz:
    q, s1 (int32), sn, sa (float16)                 one row per candidate pair
    q_comb1, q_comb2, q_sn1, q_sa1 (float16, per q)  best/second-best scores
"""

from __future__ import annotations

import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .common import env_threads, free, load_pickle, log, save_pickle, split_work_dir, timer
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


def bounded_imap(pool, fn, iterable, depth: int):
    """Ordered pool map that keeps at most `depth` tasks (and results) in flight."""
    pending = deque()
    for args in iterable:
        pending.append(pool.apply_async(fn, (args,)))
        if len(pending) >= depth:
            yield pending.popleft().get()
    while pending:
        yield pending.popleft().get()


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
        self.shards = []
        self.threads = None
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
        if len(self.shards) > 1 and self.threads is None:
            self.threads = ThreadPoolExecutor(len(self.shards))

    def free(self):
        self.shards = []
        if self.kind == "cuda":
            torch.cuda.empty_cache()

    def batch_size(self, budget_gb: float) -> int:
        biggest = max(int(self.shard_sizes.max()), 1)
        bytes_per_row = biggest * (2 if self.kind == "cuda" else 4) * 4
        return int(np.clip(budget_gb * 1e9 / bytes_per_row, 32, 8192))

    def _shard_topk(self, s, qn, qa, wn, k):
        (Sn, Sa), off, size = self.shards[s], self.offsets[s], self.shard_sizes[s]
        kk = {c: min(k[c], int(size)) for c in k}
        out = {}
        if self.kind == "numpy":
            sn = qn.astype(np.float32) @ Sn.T
            sa = qa.astype(np.float32) @ Sa.T
            comb = sn * wn[:, None] + sa * (1 - wn)[:, None]
            for c, M in (("n", sn), ("a", sa), ("c", comb)):
                idx = np.argpartition(-M, kk[c] - 1, axis=1)[:, :kk[c]]
                out[c] = (idx + off, np.take_along_axis(sn, idx, 1), np.take_along_axis(sa, idx, 1))
            return out
        with torch.no_grad():
            dev = Sn.device
            tq = torch.from_numpy(qn).to(dev, Sn.dtype, non_blocking=True)
            ta = torch.from_numpy(qa).to(dev, Sn.dtype, non_blocking=True)
            tw = torch.from_numpy(wn).to(dev, Sn.dtype)[:, None]
            sn = tq @ Sn.T
            sa = ta @ Sa.T
            comb = torch.addcmul(sa * (1 - tw), sn, tw)
            for c, M in (("n", sn), ("a", sa), ("c", comb)):
                idx = torch.topk(M, kk[c], dim=1, sorted=False).indices
                out[c] = (
                    (idx + int(off)).cpu().numpy(),
                    torch.gather(sn, 1, idx).float().cpu().numpy(),
                    torch.gather(sa, 1, idx).float().cpu().numpy(),
                )
            del sn, sa, comb
        return out

    def topk(self, qn, qa, wn, k):
        """Per channel (idx, sn, sa) numpy arrays of shape (B, k summed over shards)."""
        if self.threads is not None:
            outs = list(self.threads.map(lambda s: self._shard_topk(s, qn, qa, wn, k), range(len(self.shards))))
        else:
            outs = [self._shard_topk(s, qn, qa, wn, k) for s in range(len(self.shards))]
        return {c: tuple(np.concatenate([o[c][i] for o in outs], axis=1) for i in range(3)) for c in ("n", "a", "c")}


def _select(idx, sn, sa, key, k):
    """Keep the k best columns per row by `key` (descending), sorted."""
    k = min(k, key.shape[1])
    order = np.argsort(-key, axis=1, kind="stable")[:, :k]
    t = lambda a: np.take_along_axis(a, order, 1)
    return t(idx), t(sn), t(sa), t(key)


def _part_path(wdir: Path, country: str, tag: str) -> Path:
    safe = re.sub(r"[^0-9a-zA-Z]+", "_", country) or "empty"
    return wdir / "cand_parts" / f"{safe}__{tag}.npz"


def run(work_dir: Path, split: str, k_name: int = 3, k_addr: int = 3, k_comb: int = 8,
        w_name: float = 0.5, dim: int = 1024, device: str = "auto",
        gpu_budget_gb: float = 3.5, workers: int = 0) -> None:
    wdir = split_work_dir(work_dir, split)
    (wdir / "cand_parts").mkdir(exist_ok=True)
    workers = max(1, min((workers or env_threads()) - 1, 3))
    tag = f"k{k_name}-{k_addr}-{k_comb}_w{w_name}_d{dim}"

    s1 = load_pickle(wdir / "s1.pkl")
    s1_country = s1["country"].to_numpy()
    s1_names = s1["name_key"].to_numpy()
    s1_addrs = s1["addr_clean"].to_numpy()
    n_s1 = len(s1)
    del s1
    with timer("fit hashed TF-IDF on S1"):
        vn = HashedTfidf(dim).fit(s1_names.tolist())
        va = HashedTfidf(dim, seed=29).fit(s1_addrs.tolist())

    # Pool first (small parent), big query table afterwards.
    pool = Pool(workers, initializer=_init_worker, initargs=(vn, va))
    depth = workers + 2
    try:
        q = load_pickle(wdir / "q.pkl")
        n_q = len(q)
        q_country = q["country"].to_numpy()
        q_names = q["name_key"].to_numpy()
        q_addrs = q["addr_clean"].to_numpy()
        q_addr_empty = q["addr_empty"].to_numpy().astype(bool)
        is_s3 = q["is_s3"].to_numpy().astype(bool)
        del q
        free()

        backend = Backend(device)
        k = {"n": k_name, "a": k_addr, "c": max(k_comb, 2)}
        part_files = []
        for country in sorted(set(q_country)):
            part = _part_path(wdir, country, tag)
            part_files.append(part)
            if part.is_file():
                log(f"country '{country}': checkpoint found, skipping ({part.name})")
                continue
            s1_rows = np.flatnonzero(s1_country == country)
            q_rows = np.flatnonzero(q_country == country)
            out_q, out_s1, out_sn, out_sa = [], [], [], []
            st = {name: np.zeros(len(q_rows), np.float16) for name in ("q_comb1", "q_comb2", "q_sn1", "q_sa1")}
            if len(s1_rows) == 0:
                log(f"country '{country}': {len(q_rows):,} queries but no S1 -> no candidates")
                np.savez(part, rows=q_rows.astype(np.int32), q=np.zeros(0, np.int32), s1=np.zeros(0, np.int32),
                         sn=np.zeros(0, np.float16), sa=np.zeros(0, np.float16),
                         **{name: np.full(len(q_rows), np.nan, np.float16) for name in st})
                continue

            with timer(f"country '{country}': vectorize {len(s1_rows):,} S1"):
                Sn = np.empty((len(s1_rows), dim), dtype=np.float16)
                Sa = np.empty((len(s1_rows), dim), dtype=np.float16)
                step = 50_000
                jobs = ((s1_names[s1_rows[i:i + step]].tolist(), s1_addrs[s1_rows[i:i + step]].tolist())
                        for i in range(0, len(s1_rows), step))
                for i, (a, b) in zip(range(0, len(s1_rows), step), bounded_imap(pool, _vectorize, jobs, depth)):
                    Sn[i:i + len(a)] = a
                    Sa[i:i + len(b)] = b
                backend.load(Sn, Sa)
                del Sn, Sa
                free()

            B = backend.batch_size(gpu_budget_gb)
            chunks = [q_rows[i:i + B] for i in range(0, len(q_rows), B)]
            jobs = ((q_names[c].tolist(), q_addrs[c].tolist()) for c in chunks)
            pos = 0
            with timer(f"country '{country}': retrieve {len(q_rows):,} queries (batch {B})"):
                for ci, (rows, (Qn, Qa)) in enumerate(zip(chunks, bounded_imap(pool, _vectorize, jobs, depth))):
                    name_empty = ~Qn.any(axis=1)
                    wn = np.where(q_addr_empty[rows], 1.0, w_name).astype(np.float32)
                    wn[name_empty] = 0.0
                    m = backend.topk(Qn, Qa, wn, k)
                    n_idx, n_sn, n_sa, n_key = _select(*m["n"], m["n"][1], k_name)
                    a_idx, a_sn, a_sa, a_key = _select(*m["a"], m["a"][2], k_addr)
                    c_comb = m["c"][1] * wn[:, None] + m["c"][2] * (1 - wn)[:, None]
                    c_idx, c_sn, c_sa, c_key = _select(*m["c"], c_comb, max(k_comb, 2))
                    sl = slice(pos, pos + len(rows))
                    pos += len(rows)
                    st["q_comb1"][sl] = c_key[:, 0]
                    st["q_comb2"][sl] = c_key[:, 1] if c_key.shape[1] > 1 else 0
                    st["q_sn1"][sl] = n_key[:, 0]
                    st["q_sa1"][sl] = a_key[:, 0]
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
                        log(f"    {pos:,}/{len(q_rows):,}")
            backend.free()
            np.savez(part, rows=q_rows.astype(np.int32), q=np.concatenate(out_q), s1=np.concatenate(out_s1),
                     sn=np.concatenate(out_sn), sa=np.concatenate(out_sa), **st)
            log(f"country '{country}': checkpoint saved ({part.name})")
            del out_q, out_s1, out_sn, out_sa, st
            free()
    finally:
        pool.terminate()

    with timer("merge country checkpoints"):
        stats = {name: np.full(n_q, np.nan, dtype=np.float16) for name in ("q_comb1", "q_comb2", "q_sn1", "q_sa1")}
        cq, cs, csn, csa = [], [], [], []
        for part in part_files:
            z = np.load(part)
            for name in stats:
                stats[name][z["rows"]] = z[name]
            cq.append(z["q"]); cs.append(z["s1"]); csn.append(z["sn"]); csa.append(z["sa"])
        cq = np.concatenate(cq)
        order = np.argsort(cq, kind="stable")
        cands = {"q": cq[order], "s1": np.concatenate(cs)[order],
                 "sn": np.concatenate(csn)[order], "sa": np.concatenate(csa)[order], **stats}
        del cq, cs, csn, csa
        np.savez(wdir / "cands.npz", **cands)
    save_pickle({"k_name": k_name, "k_addr": k_addr, "k_comb": k_comb, "w_name": w_name, "dim": dim},
                wdir / "retrieval_params.pkl")
    log(f"saved {len(cands['q']):,} candidate pairs ({len(cands['q']) / max(n_q, 1):.2f} per query)")

    true_path = wdir / "q_true_s1.npy"
    if true_path.is_file():
        rep = candidate_report(n_s1, np.load(true_path), cands["q"], cands["s1"], is_s3)
        log("RETRIEVAL REPORT (train ground truth):\n" + format_report(rep))
        save_pickle(rep, wdir / "retrieval_report.pkl")

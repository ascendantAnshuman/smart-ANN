"""Resumable exact top-k by brute force.

Needed only for the QA chunk index -- both ANN datasets ship exact top-100
neighbours in their HDF5 files (ADR-0011).

Ties are broken by ascending index, deterministically. That matters more than it
looks: normalised inner products collide often enough at float32 precision that
an unspecified tie-break would make ground truth depend on block size, and
recall would then drift when the block size changed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from smartann.data.registry import Space

__all__ = ["GroundTruthResult", "compute_topk", "recall_at_k"]

_BYTES_PER_FLOAT = 4


@dataclass(frozen=True)
class GroundTruthResult:
    neighbors: np.ndarray  # (n_queries, k) int32
    scores: np.ndarray  # (n_queries, k) float32 -- similarity for ip, distance for l2
    space: Space
    n_base: int
    resumed_from_block: int


def _block_size(dim: int, n_queries_block: int, budget_mb: float) -> int:
    """Base rows per block so one score matrix stays inside *budget_mb*."""
    per_row = (dim + n_queries_block) * _BYTES_PER_FLOAT
    return max(1, int((budget_mb * 1024 * 1024) // per_row))


def _merge_topk(
    best_scores: np.ndarray,
    best_idx: np.ndarray,
    new_scores: np.ndarray,
    new_idx: np.ndarray,
    k: int,
    larger_is_better: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge a new score block into the running top-k, tie-broken by index."""
    scores = np.concatenate([best_scores, new_scores], axis=1)
    idx = np.concatenate([best_idx, np.broadcast_to(new_idx, new_scores.shape)], axis=1)

    # lexsort's last key is primary. Ascending index breaks ties; the score key
    # is negated when larger is better so that one ascending sort serves both.
    primary = -scores if larger_is_better else scores
    order = np.lexsort((idx, primary), axis=1)[:, :k]

    rows = np.arange(scores.shape[0])[:, None]
    return scores[rows, order].astype(np.float32), idx[rows, order].astype(np.int32)


def compute_topk(
    base: np.ndarray,
    queries: np.ndarray,
    k: int = 100,
    space: Space = "ip",
    checkpoint: Path | None = None,
    block_mb: float = 1024.0,
    query_block: int = 256,
    quiet: bool = False,
) -> GroundTruthResult:
    """Exact top-*k* neighbours of every row of *queries* within *base*.

    Chunked so that neither the base set nor the score matrix is ever fully
    resident. Resumable: *checkpoint* records the running top-k and the index of
    the last completed base block, so an interrupted run continues rather than
    restarting.
    """
    if base.ndim != 2 or queries.ndim != 2:
        raise ValueError(f"expected 2-D arrays, got base{base.shape} queries{queries.shape}")
    if base.shape[1] != queries.shape[1]:
        raise ValueError(f"dim mismatch: base {base.shape[1]} vs queries {queries.shape[1]}")
    if k > base.shape[0]:
        raise ValueError(f"k={k} exceeds the base set size {base.shape[0]}")
    if space not in ("ip", "l2", "cosine"):
        raise ValueError(f"unsupported space {space!r}")

    larger_is_better = space in ("ip", "cosine")
    n_q, dim = queries.shape
    n_base = base.shape[0]
    bsize = _block_size(dim, query_block, block_mb)

    fill = -np.inf if larger_is_better else np.inf
    best_scores = np.full((n_q, k), fill, dtype=np.float32)
    best_idx = np.full((n_q, k), -1, dtype=np.int32)
    start_block = 0

    if checkpoint is not None and checkpoint.exists():
        ck = np.load(checkpoint)
        if int(ck["k"]) == k and int(ck["n_base"]) == n_base and int(ck["bsize"]) == bsize:
            best_scores = ck["best_scores"]
            best_idx = ck["best_idx"]
            start_block = int(ck["next_block"])

    blocks = list(range(start_block, (n_base + bsize - 1) // bsize))
    q = np.ascontiguousarray(queries, dtype=np.float32)

    for bi in tqdm(blocks, desc="ground truth", disable=quiet or not sys.stderr.isatty(), leave=False):
        lo = bi * bsize
        hi = min(lo + bsize, n_base)
        block = np.ascontiguousarray(base[lo:hi], dtype=np.float32)
        block_idx = np.arange(lo, hi, dtype=np.int32)[None, :]

        for qlo in range(0, n_q, query_block):
            qhi = min(qlo + query_block, n_q)
            qb = q[qlo:qhi]
            if larger_is_better:
                scores = qb @ block.T
            else:
                # ||a-b||^2 = ||a||^2 - 2ab + ||b||^2; the ||a||^2 term is
                # constant per query and cannot change the ordering, so it is
                # dropped to save an allocation.
                scores = (block * block).sum(1)[None, :] - 2.0 * (qb @ block.T)
            best_scores[qlo:qhi], best_idx[qlo:qhi] = _merge_topk(
                best_scores[qlo:qhi],
                best_idx[qlo:qhi],
                scores.astype(np.float32),
                block_idx,
                k,
                larger_is_better,
            )

        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                checkpoint,
                best_scores=best_scores,
                best_idx=best_idx,
                next_block=bi + 1,
                k=k,
                n_base=n_base,
                bsize=bsize,
            )

    if np.any(best_idx < 0):
        raise RuntimeError("ground truth incomplete: some neighbours were never filled")

    return GroundTruthResult(
        neighbors=best_idx,
        scores=best_scores,
        space=space,
        n_base=n_base,
        resumed_from_block=start_block,
    )


def recall_at_k(retrieved: np.ndarray, truth: np.ndarray, k: int) -> np.ndarray:
    """Per-query ``|retrieved_top_k & true_top_k| / k``."""
    if k > truth.shape[1]:
        raise ValueError(f"k={k} exceeds available ground truth depth {truth.shape[1]}")
    out = np.empty(retrieved.shape[0], dtype=np.float64)
    for i in range(retrieved.shape[0]):
        out[i] = len(set(retrieved[i, :k].tolist()) & set(truth[i, :k].tolist())) / k
    return out

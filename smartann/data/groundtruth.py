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

__all__ = [
    "DIST_CONVENTION",
    "GroundTruthResult",
    "MonotonicityReport",
    "check_row_monotonic",
    "compute_topk",
    "gather_distances",
    "recall_at_k",
]

# The units we store ground-truth distances in. These are hnswlib's own
# conventions, not ours: its "l2" space returns SQUARED euclidean distance and
# its "ip" space returns ``1 - inner_product`` (both pinned in
# tests/test_hnswlib_contract.py). Storing anything else would mean converting
# at every comparison site, and one missed conversion inverts a ranking
# silently. Smaller is better in both.
DIST_CONVENTION: dict[str, str] = {
    "l2": "squared_l2",
    "ip": "one_minus_ip",
    "cosine": "one_minus_ip",
}

_BYTES_PER_FLOAT = 4


@dataclass(frozen=True)
class GroundTruthResult:
    neighbors: np.ndarray  # (n_queries, k) int32
    scores: np.ndarray  # (n_queries, k) float32 -- similarity for ip, squared distance for l2
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
                # ||q-b||^2 = ||q||^2 - 2qb + ||b||^2. The ||q||^2 term is
                # constant per query and cannot change the ORDERING -- which is
                # why it was originally dropped -- but these scores are stored
                # and later compared against a threshold, and without it every
                # value is off by a per-query constant. Correct for ranking is
                # not correct for a value that leaves the function.
                scores = (
                    (qb * qb).sum(1)[:, None]
                    + (block * block).sum(1)[None, :]
                    - 2.0 * (qb @ block.T)
                )
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


def gather_distances(
    base: np.ndarray,
    queries: np.ndarray,
    neighbors: np.ndarray,
    space: Space,
    query_block: int = 256,  # float64 intermediates are 8x the float32 ones
) -> np.ndarray:
    """Distance from each query to each of *its own* stored neighbours.

    Not a re-run of :func:`compute_topk`. We already have the neighbour ids --
    shipped by the dataset and remapped onto the compacted base (ADR-0014), or
    computed once for the QA corpus -- and re-deriving them would risk a
    different tie ordering, leaving distances silently misaligned with the ids
    they are supposed to describe. Gathering instead guarantees
    ``out[q, j] == dist(queries[q], base[neighbors[q, j]])`` by construction,
    and costs ``n_queries * depth`` comparisons rather than ``n_queries * n_base``.

    Computed and returned in **float64**, deliberately. ``1 - ip`` suffers
    catastrophic cancellation as the inner product approaches 1: float32 eps
    near 1.0 is 1.19e-07, so every pair closer than that collapses onto the same
    stored value and a genuinely-worse neighbour becomes indistinguishable from
    the true k-th. Retrieved distances come back from hnswlib as float32 by
    necessity; keeping the ground-truth side exact means ``atol`` in the recall
    band covers the float32 noise floor once rather than twice. 8 MB per dataset.

    Returns float64 in :data:`DIST_CONVENTION` units -- smaller is better.
    """
    if neighbors.shape[0] != queries.shape[0]:
        raise ValueError(
            f"neighbours cover {neighbors.shape[0]} queries but {queries.shape[0]} were given"
        )
    if base.shape[1] != queries.shape[1]:
        raise ValueError(f"dim mismatch: base {base.shape[1]} vs queries {queries.shape[1]}")
    if space not in DIST_CONVENTION:
        raise ValueError(f"unsupported space {space!r}")
    lo, hi = int(neighbors.min()), int(neighbors.max())
    if lo < 0 or hi >= base.shape[0]:
        raise ValueError(
            f"neighbour ids span [{lo}, {hi}], outside the base set [0, {base.shape[0] - 1}]"
        )

    n_q, depth = neighbors.shape
    out = np.empty((n_q, depth), dtype=np.float64)

    for qlo in range(0, n_q, query_block):
        qhi = min(qlo + query_block, n_q)
        qb = np.ascontiguousarray(queries[qlo:qhi], dtype=np.float64)
        # (block, depth, dim) -- bounded by query_block, not by n_base
        nb = np.asarray(base[neighbors[qlo:qhi].ravel()], dtype=np.float64).reshape(
            qhi - qlo, depth, -1
        )
        if space == "l2":
            diff = nb - qb[:, None, :]
            out[qlo:qhi] = np.einsum("qdi,qdi->qd", diff, diff)
        else:
            out[qlo:qhi] = 1.0 - np.einsum("qdi,qi->qd", nb, qb)

    return out


@dataclass(frozen=True)
class MonotonicityReport:
    """Whether stored neighbour ids are ordered consistently with their distances."""

    ok: bool
    worst_violation: float  # most negative diff seen; 0.0 when perfectly sorted
    worst_query: int
    worst_depth: int
    n_violating_rows: int
    n_queries: int
    tol: float


def check_row_monotonic(dists: np.ndarray, tol: float = 1e-5) -> MonotonicityReport:
    """Assert every ground-truth row is non-decreasing in distance.

    This is the sharp test of ADR-0014's claim. The neighbour ids were shipped
    by the dataset and then remapped onto a compacted base; the distances here
    were computed independently from the compacted vectors. If the remapping
    had shifted any id onto the wrong row, the ordering would break. Run over
    every query, not a sample -- a remapping bug can touch one row in a million.
    """
    if dists.ndim != 2 or dists.shape[1] < 2:
        raise ValueError(f"expected a (n_queries, depth>=2) array, got {dists.shape}")
    diffs = np.diff(dists.astype(np.float64), axis=1)
    flat = int(np.argmin(diffs))
    qi, di = np.unravel_index(flat, diffs.shape)
    worst = float(diffs[qi, di])
    return MonotonicityReport(
        ok=worst >= -tol,
        worst_violation=min(worst, 0.0),
        worst_query=int(qi),
        worst_depth=int(di),
        n_violating_rows=int(np.sum(np.any(diffs < -tol, axis=1))),
        n_queries=int(dists.shape[0]),
        tol=tol,
    )

"""Recall metrics. Every recall number in this project comes from here.

Two definitions, computed together by one function so that two call sites can
never disagree about ``k`` or the tolerance band:

**Score-equality recall** counts a retrieved neighbour as correct when its
distance is at least as good as the true k-th distance. It is the honest metric
when a corpus contains duplicate vectors: NYTimes has 40,793 byte-identical
repeat rows (ADR-0014 amendment), and one query has 95 neighbours at the same
distance as its 10th. Index-overlap recall caps that query at 10/95 no matter
how good the search is -- it would be measuring the arbitrary tie-break of the
ground-truth generator, not the retrieval.

**Index-overlap recall** is the conventional ``|retrieved ∩ true| / k``. Kept
because it is what the literature reports and what MobileRAG's numbers are, so
the comparison has to be like for like. The gap between the two is the tie rate.

Both operate on **distances, smaller is better**, in the convention ground truth
is stored in (ADR-0018). There is deliberately no ``space`` parameter: after
ADR-0018 both spaces store distances, so a sense flag could only ever be passed
wrong, and passing it wrong would invert a ranking silently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "DEFAULT_ATOL",
    "DEFAULT_RTOL",
    "RecallResult",
    "recall_at_k",
    "recall_at_k_batch",
    "score_band",
    "tie_rate",
]

# Sized from measurement, not chosen. The retrieved side is float32 (hnswlib);
# ground truth is float64. Comparing the two over the stored top-100 of all
# three datasets, the largest disagreement observed was:
#
#   sift1m   0.0        (integer-valued vectors: squared L2 is exact in float32)
#   nytimes  2.02e-07
#   squad    2.92e-07
#
# atol=1e-6 covers the worst of those about 3.4x over. rtol carries the band on
# large-magnitude spaces -- SIFT's k-th distances run to 1.3e+05, where an
# absolute tolerance means nothing.
DEFAULT_ATOL = 1e-6
DEFAULT_RTOL = 1e-6


def score_band(threshold: float | np.ndarray, rtol: float, atol: float) -> float | np.ndarray:
    """``atol + rtol * |threshold|`` -- the combined band, and why it is combined.

    A pure *relative* band collapses to nothing as the threshold approaches
    zero, which is exactly where duplicate vectors put it: NYTimes' k-th
    distance reaches ``-1.92e-07``, where a float32 retrieved value of
    ``-2.38e-07`` is 24% away in relative terms and 4.7e-08 away in absolute
    terms. With ``rtol=1e-6`` alone, a genuinely-correct neighbour is scored
    wrong. A pure *absolute* band is the mirror failure on SIFT, whose k-th
    distances run to 1.3e+05.

    ``abs()`` is load-bearing rather than defensive: stored distances really do
    go slightly negative. For a duplicate vector the float64 inner product of
    two float32-normalised rows exceeds 1.0, so ``1 - ip`` lands just below zero.
    """
    return atol + rtol * np.abs(threshold)


@dataclass(frozen=True)
class RecallResult:
    """Both recall definitions for one query, plus what separates them."""

    score_equality: float
    index_overlap: float
    threshold: float  # the true k-th distance
    band: float
    n_tied_in_truth: int  # true neighbours indistinguishable from the k-th
    n_retrieved: int  # how many candidates the search actually returned

    @property
    def tie_gap(self) -> float:
        """How much the tie-break of the ground-truth generator was worth."""
        return self.score_equality - self.index_overlap


def _validate(
    retrieved_ids: np.ndarray, retrieved_dists: np.ndarray, gt_dists: np.ndarray, k: int
) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if retrieved_ids.shape != retrieved_dists.shape:
        raise ValueError(
            f"ids {retrieved_ids.shape} and distances {retrieved_dists.shape} must align"
        )
    if gt_dists.shape[-1] < k:
        raise ValueError(f"k={k} exceeds the ground-truth depth {gt_dists.shape[-1]}")


def recall_at_k(
    retrieved_ids: np.ndarray,
    retrieved_dists: np.ndarray,
    gt_ids: np.ndarray,
    gt_dists: np.ndarray,
    k: int,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> RecallResult:
    """Both recall definitions for a single query.

    *retrieved_ids* and *retrieved_dists* are the search output in rank order;
    *gt_ids* and *gt_dists* are the stored ground truth for the same query.
    Neither list is required to be length *k* -- a search that returns fewer
    than *k* candidates has genuinely lost recall, and the denominator stays *k*.
    """
    _validate(retrieved_ids, retrieved_dists, gt_dists, k)

    ids = np.asarray(retrieved_ids).ravel()[:k]
    dists = np.asarray(retrieved_dists, dtype=np.float64).ravel()[:k]
    if np.unique(ids).size != ids.size:
        # A vector reachable from two cluster files would be counted twice and
        # inflate recall. IVF assignment is exclusive, so this is a build bug.
        raise ValueError(f"retrieved ids contain duplicates: {ids.tolist()}")

    truth = np.asarray(gt_dists, dtype=np.float64).ravel()
    threshold = float(truth[k - 1])
    band = float(score_band(threshold, rtol, atol))

    n_correct = int(np.count_nonzero(dists <= threshold + band))
    overlap = len(set(ids.tolist()) & set(np.asarray(gt_ids).ravel()[:k].tolist()))

    return RecallResult(
        score_equality=n_correct / k,
        index_overlap=overlap / k,
        threshold=threshold,
        band=band,
        n_tied_in_truth=int(np.count_nonzero(np.abs(truth - threshold) <= band)),
        n_retrieved=int(ids.size),
    )


def recall_at_k_batch(
    retrieved_ids: np.ndarray,
    retrieved_dists: np.ndarray,
    gt_ids: np.ndarray,
    gt_dists: np.ndarray,
    k: int,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-query ``(score_equality, index_overlap)`` for a whole query set.

    Vectorised rather than a loop over :func:`recall_at_k`, but the two are
    asserted equal on real data in the tests -- a fast path that can disagree
    with the reference is worse than no fast path.
    """
    _validate(retrieved_ids, retrieved_dists, gt_dists, k)
    dists = np.asarray(retrieved_dists, dtype=np.float64)[:, :k]
    truth = np.asarray(gt_dists, dtype=np.float64)

    threshold = truth[:, k - 1]
    band = np.asarray(score_band(threshold, rtol, atol))
    score_eq = np.count_nonzero(dists <= (threshold + band)[:, None], axis=1) / k

    ids = np.asarray(retrieved_ids)[:, :k]
    gt = np.asarray(gt_ids)[:, :k]
    overlap = np.array(
        [len(set(a.tolist()) & set(b.tolist())) / k for a, b in zip(ids, gt, strict=True)]
    )
    return score_eq, overlap


def tie_rate(
    gt_dists: np.ndarray,
    k: int,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> float:
    """Fraction of queries with a true neighbour indistinguishable from the k-th.

    The upper bound on how far the two recall definitions can diverge. Measured
    at k=10: SIFT 1.37%, SQuAD 0.30%, NYTimes 13.55%.
    """
    truth = np.asarray(gt_dists, dtype=np.float64)
    if truth.shape[1] <= k:
        raise ValueError(f"need ground-truth depth > k={k} to see beyond the boundary")
    threshold = truth[:, k - 1]
    band = np.asarray(score_band(threshold, rtol, atol))
    beyond = np.abs(truth[:, k:] - threshold[:, None]) <= band[:, None]
    return float(np.mean(np.any(beyond, axis=1)))

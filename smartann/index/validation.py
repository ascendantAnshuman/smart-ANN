"""Exhaustive-equivalence checking. Never imported by the search path.

Acceptance check 1 asks whether EcoVector, when it probes *every* cluster,
returns exactly what brute force returns. That is a correctness proof for the
id mapping, the per-cluster storage, the flat/graph distance conventions and the
merge -- everything except the approximation itself.

It lives in its own module for a reason. An ``exhaustive=True`` flag on
:class:`~smartann.types.AnnConfig` would be reachable from any policy, would
serialise into configs, and would eventually be set by accident in a run whose
numbers then look wonderful. Here it cannot be reached without importing a
module named validation and passing a literal confirmation string.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smartann.index.ecovector import EcoVectorIndex, SearchResult, _brute_force
from smartann.types import AnnConfig

__all__ = ["EquivalenceReport", "check_equivalence", "exhaustive_search"]

CONFIRM = "validation-only"

# Probing every cluster is O(nc) file opens per query. On SIFT1M with nc=2000
# that is 2000 opens for one query, so the harness refuses to run on a full-size
# index unless the caller raises the ceiling deliberately.
DEFAULT_MAX_CLUSTERS = 512


@dataclass(frozen=True)
class EquivalenceReport:
    n_queries: int
    n_exact_id_matches: int
    n_score_equal: int
    max_distance_error: float
    worst_query: int
    tied_queries: int

    @property
    def ok(self) -> bool:
        """Score equality is the criterion, not id equality.

        Two vectors at identical distance are interchangeable in a correct
        top-k, and which one a search returns is a tie-break, not an error.
        SIFT has 14,538 duplicate rows and NYTimes 40,793, so demanding
        identical ids would fail a correct implementation.
        """
        return self.n_score_equal == self.n_queries


def exhaustive_search(
    index: EcoVectorIndex,
    query: np.ndarray,
    k: int,
    *,
    confirm: str,
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
) -> SearchResult:
    """Search every non-empty cluster. Validation only; never use in a run."""
    if confirm != CONFIRM:
        raise ValueError(
            f"exhaustive_search is validation-only; pass confirm={CONFIRM!r} if you mean it"
        )
    live = int(np.count_nonzero(index.sizes > 0))
    if live > max_clusters:
        raise ValueError(
            f"index has {live} non-empty clusters, above the {max_clusters} ceiling. "
            "Exhaustive search opens every cluster file per query; raise max_clusters "
            "explicitly if that is really what you want."
        )
    cfg = AnnConfig(
        nprobe=live,
        ef_centroid=max(live, 16),
        ef_list=max(index.meta.largest_cluster, k),
        k=k,
        per_cluster_k=max(k, min(index.meta.largest_cluster, 2048)),
    )
    return index.search(query, cfg)


def check_equivalence(
    index: EcoVectorIndex,
    base: np.ndarray,
    queries: np.ndarray,
    k: int = 10,
    *,
    confirm: str = CONFIRM,
    atol: float = 1e-5,
    rtol: float = 1e-6,
) -> EquivalenceReport:
    """Compare exhaustive EcoVector against brute force over the same vectors."""
    n_exact = n_equal = tied = 0
    worst = 0.0
    worst_q = -1

    for qi in range(queries.shape[0]):
        q = np.asarray(queries[qi], dtype=np.float32)
        got = exhaustive_search(index, q, k, confirm=confirm)

        truth_d = _brute_force(q, np.asarray(base, dtype=np.float32), index.meta.space)
        order = np.lexsort((np.arange(truth_d.size), truth_d))[:k]
        want_d = truth_d[order]

        if got.ids.size == k and np.array_equal(got.ids, order.astype(np.int64)):
            n_exact += 1
        err = float(np.max(np.abs(got.dists - want_d))) if got.dists.size == k else float("inf")
        if err > worst:
            worst, worst_q = err, qi
        if err <= atol + rtol * float(np.max(np.abs(want_d))):
            n_equal += 1
        if np.any(np.abs(np.diff(want_d)) <= 0.0):
            tied += 1

    return EquivalenceReport(
        n_queries=int(queries.shape[0]),
        n_exact_id_matches=n_exact,
        n_score_equal=n_equal,
        max_distance_error=worst,
        worst_query=worst_q,
        tied_queries=tied,
    )

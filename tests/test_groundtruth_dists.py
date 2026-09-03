"""Ground-truth distances: correct VALUES, not merely correct ordering.

Every test in this file compares numbers rather than rankings. That distinction
is the point: ``compute_topk`` dropped the ``||q||^2`` term from its L2
expression because it cannot change the ordering, and so every ordering test the
module had passed while the stored values were each off by a per-query constant.
Score-equality recall compares a retrieved distance against a stored threshold,
so a constant offset is not harmless there -- it is silent, unbounded error.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from smartann.data.groundtruth import (
    DIST_CONVENTION,
    check_row_monotonic,
    compute_topk,
    gather_distances,
)

DIM = 12


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    return (
        rng.standard_normal((300, DIM)).astype(np.float32) * 3.0,
        rng.standard_normal((40, DIM)).astype(np.float32) * 3.0,
    )


def _unit(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


# ---------------------------------------------------------------------------
# compute_topk
# ---------------------------------------------------------------------------
def test_compute_topk_l2_scores_are_true_squared_distances(
    data: tuple[np.ndarray, np.ndarray],
) -> None:
    base, queries = data
    result = compute_topk(base, queries, k=10, space="l2", quiet=True)

    want = cdist(queries, base, "sqeuclidean")
    got = np.take_along_axis(want, result.neighbors.astype(np.int64), axis=1)
    np.testing.assert_allclose(result.scores, got, rtol=1e-4, atol=1e-3)


def test_dropping_the_query_norm_term_is_invisible_to_ordering(
    data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Why the bug survived: it is an exact no-op for every ranking test."""
    base, queries = data
    full = cdist(queries, base, "sqeuclidean")
    without = (base * base).sum(1)[None, :] - 2.0 * (queries @ base.T)

    np.testing.assert_array_equal(np.argsort(full, axis=1), np.argsort(without, axis=1))
    assert not np.allclose(full, without), "the two expressions must differ in value"


def test_compute_topk_ip_scores_are_raw_inner_products(
    data: tuple[np.ndarray, np.ndarray],
) -> None:
    base, queries = _unit(data[0]), _unit(data[1])
    result = compute_topk(base, queries, k=10, space="ip", quiet=True)

    for qi in range(queries.shape[0]):
        want = base[result.neighbors[qi]] @ queries[qi]
        np.testing.assert_allclose(result.scores[qi], want, atol=1e-5)


# ---------------------------------------------------------------------------
# gather_distances
# ---------------------------------------------------------------------------
def test_gather_distances_l2_matches_cdist(data: tuple[np.ndarray, np.ndarray]) -> None:
    base, queries = data
    rng = np.random.default_rng(0)
    nbrs = rng.integers(0, base.shape[0], size=(queries.shape[0], 20)).astype(np.int32)

    got = gather_distances(base, queries, nbrs, "l2")
    want = np.take_along_axis(cdist(queries, base, "sqeuclidean"), nbrs.astype(np.int64), axis=1)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-3)


def test_gather_distances_ip_is_one_minus_inner_product(
    data: tuple[np.ndarray, np.ndarray],
) -> None:
    base, queries = _unit(data[0]), _unit(data[1])
    rng = np.random.default_rng(1)
    nbrs = rng.integers(0, base.shape[0], size=(queries.shape[0], 20)).astype(np.int32)

    got = gather_distances(base, queries, nbrs, "ip")
    for qi in range(queries.shape[0]):
        np.testing.assert_allclose(got[qi], 1.0 - base[nbrs[qi]] @ queries[qi], atol=1e-5)


@pytest.mark.parametrize("space", ["l2", "ip"])
def test_gather_distances_agrees_with_hnswlib_itself(
    space: str, data: tuple[np.ndarray, np.ndarray]
) -> None:
    """The whole reason for the convention: our numbers and the library's must
    be directly comparable, with no transform at the comparison site."""
    hnswlib = pytest.importorskip("hnswlib")
    base, queries = (_unit(data[0]), _unit(data[1])) if space == "ip" else data

    idx = hnswlib.Index(space=space, dim=DIM)
    idx.init_index(max_elements=base.shape[0], M=16, ef_construction=200, random_seed=5)
    idx.add_items(base, np.arange(base.shape[0]))
    idx.set_ef(base.shape[0])  # effectively exhaustive on 300 points
    labels, lib_dists = idx.knn_query(queries, k=10)

    ours = gather_distances(base, queries, labels.astype(np.int32), space)
    np.testing.assert_allclose(ours, lib_dists, rtol=1e-4, atol=1e-4)


def test_gather_distances_follows_the_ids_it_was_given(
    data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Alignment, not ordering: permute the ids and the distances must permute too."""
    base, queries = data
    rng = np.random.default_rng(2)
    nbrs = rng.integers(0, base.shape[0], size=(queries.shape[0], 8)).astype(np.int32)
    perm = rng.permutation(8)

    straight = gather_distances(base, queries, nbrs, "l2")
    shuffled = gather_distances(base, queries, nbrs[:, perm], "l2")
    np.testing.assert_allclose(shuffled, straight[:, perm], rtol=1e-6)


def test_gather_distances_rejects_ids_outside_the_base(
    data: tuple[np.ndarray, np.ndarray],
) -> None:
    """A stale ground-truth file against a compacted base must fail loudly."""
    base, queries = data
    nbrs = np.zeros((queries.shape[0], 4), dtype=np.int32)
    nbrs[0, 0] = base.shape[0]  # one past the end
    with pytest.raises(ValueError, match="outside the base set"):
        gather_distances(base, queries, nbrs, "l2")


def test_gather_distances_is_block_size_invariant(data: tuple[np.ndarray, np.ndarray]) -> None:
    base, queries = data
    rng = np.random.default_rng(3)
    nbrs = rng.integers(0, base.shape[0], size=(queries.shape[0], 8)).astype(np.int32)
    a = gather_distances(base, queries, nbrs, "l2", query_block=1024)
    b = gather_distances(base, queries, nbrs, "l2", query_block=7)
    np.testing.assert_array_equal(a, b)


def test_every_space_has_a_declared_convention() -> None:
    assert set(DIST_CONVENTION) == {"l2", "ip", "cosine"}
    assert DIST_CONVENTION["cosine"] == DIST_CONVENTION["ip"]


# ---------------------------------------------------------------------------
# check_row_monotonic
# ---------------------------------------------------------------------------
def test_monotonicity_passes_on_real_ground_truth(data: tuple[np.ndarray, np.ndarray]) -> None:
    base, queries = data
    result = compute_topk(base, queries, k=25, space="l2", quiet=True)
    dists = gather_distances(base, queries, result.neighbors, "l2")

    report = check_row_monotonic(dists)
    assert report.ok, report
    assert report.n_violating_rows == 0
    assert report.n_queries == queries.shape[0]


def test_monotonicity_catches_a_single_swapped_pair() -> None:
    """One id in one row landing on the wrong vector is what this must catch."""
    dists = np.tile(np.arange(10, dtype=np.float32), (500, 1))
    dists[317, 4], dists[317, 5] = dists[317, 5], dists[317, 4]

    report = check_row_monotonic(dists)
    assert not report.ok
    assert report.worst_query == 317
    assert report.worst_depth == 4
    assert report.n_violating_rows == 1
    assert report.worst_violation == pytest.approx(-1.0)  # [.., 3, 5, 4, ..] -> 4 - 5


def test_monotonicity_tolerates_float_noise_at_the_tolerance() -> None:
    dists = np.tile(np.arange(10, dtype=np.float32), (50, 1))
    dists[0, 3] += 1e-7  # below tol, must not trip
    assert check_row_monotonic(dists, tol=1e-5).ok

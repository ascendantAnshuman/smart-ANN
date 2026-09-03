"""Phase 1: zero-norm handling for cosine datasets (ADR-0014).

NYTimes really does ship 239 zero-norm base vectors and 9 zero-norm queries.
These tests pin down what happens to them, because the failure mode is silent:
an un-normalised zero vector scores exactly 0.0 under inner product, which ranks
it above every genuinely anti-correlated vector and displaces correct results.
"""

from __future__ import annotations

import numpy as np
import pytest

from smartann.data.readers import (
    ZeroNormError,
    drop_zero_norm_rows,
    l2_normalize,
    remap_neighbors,
)


def _with_zeros(n: int, dim: int, zero_at: list[int]) -> np.ndarray:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    x[zero_at] = 0.0
    return x


def test_drop_reports_exactly_which_rows_went() -> None:
    x = _with_zeros(20, 8, [3, 7, 15])
    rep = drop_zero_norm_rows(x)

    assert rep.n_dropped == 3
    assert rep.dropped_indices.tolist() == [3, 7, 15]
    assert rep.kept.shape == (17, 8)
    assert np.all(np.linalg.norm(rep.kept, axis=1) > 0)


def test_old_to_new_maps_survivors_and_marks_dropped() -> None:
    x = _with_zeros(10, 4, [2, 5])
    rep = drop_zero_norm_rows(x)

    assert rep.old_to_new[0] == 0
    assert rep.old_to_new[1] == 1
    assert rep.old_to_new[2] == -1  # dropped
    assert rep.old_to_new[3] == 2  # shifted down by one
    assert rep.old_to_new[5] == -1
    assert rep.old_to_new[6] == 4  # shifted down by two


def test_kept_rows_are_the_original_vectors() -> None:
    x = _with_zeros(12, 6, [4])
    rep = drop_zero_norm_rows(x)
    expected = np.delete(x, 4, axis=0)
    np.testing.assert_array_equal(rep.kept, expected)


def test_no_zero_rows_is_a_no_op() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((10, 4)).astype(np.float32)
    rep = drop_zero_norm_rows(x)
    assert rep.n_dropped == 0
    np.testing.assert_array_equal(rep.kept, x)
    np.testing.assert_array_equal(rep.old_to_new, np.arange(10))


# ---------------------------------------------------------------------------
# Remapping
# ---------------------------------------------------------------------------
def test_remap_shifts_indices_into_the_compacted_space() -> None:
    x = _with_zeros(10, 4, [2, 5])
    rep = drop_zero_norm_rows(x)
    neighbors = np.array([[0, 1, 3], [6, 7, 9]], dtype=np.int32)

    got = remap_neighbors(neighbors, rep.old_to_new)
    assert got.tolist() == [[0, 1, 2], [4, 5, 7]]
    assert got.dtype == np.int32


def test_remap_refuses_when_ground_truth_references_a_dropped_row() -> None:
    """Truncating a neighbour list would silently overstate recall."""
    x = _with_zeros(10, 4, [2])
    rep = drop_zero_norm_rows(x)
    neighbors = np.array([[0, 2, 3]], dtype=np.int32)  # 2 was dropped

    with pytest.raises(ValueError, match="references 1 dropped row"):
        remap_neighbors(neighbors, rep.old_to_new)


def test_remapped_neighbors_still_point_at_the_same_vectors() -> None:
    """The whole point of remapping: identity is preserved, only indices move."""
    x = _with_zeros(30, 5, [4, 11, 20])
    rep = drop_zero_norm_rows(x)
    neighbors = np.array([[0, 6, 15, 29]], dtype=np.int32)

    remapped = remap_neighbors(neighbors, rep.old_to_new)
    for old, new in zip(neighbors[0], remapped[0], strict=True):
        np.testing.assert_array_equal(x[old], rep.kept[new])


# ---------------------------------------------------------------------------
# The interaction that matters
# ---------------------------------------------------------------------------
def test_normalize_still_refuses_zero_rows_after_the_drop_helper_exists() -> None:
    """Dropping must stay an explicit step, never a side effect of normalising."""
    with pytest.raises(ZeroNormError):
        l2_normalize(_with_zeros(5, 4, [1]))


def test_drop_then_normalize_is_the_supported_pipeline() -> None:
    rep = drop_zero_norm_rows(_with_zeros(50, 16, [0, 25, 49]))
    out = l2_normalize(rep.kept, "test")
    assert out.shape == (47, 16)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)


def test_zero_vector_would_outrank_anticorrelated_vectors_under_ip() -> None:
    """Demonstrates why dropping is necessary rather than merely tidy."""
    query = np.array([1.0, 0.0], dtype=np.float32)
    anticorrelated = np.array([-1.0, 0.0], dtype=np.float32)
    zero = np.array([0.0, 0.0], dtype=np.float32)

    assert float(zero @ query) > float(anticorrelated @ query)

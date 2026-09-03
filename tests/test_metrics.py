"""Recall metrics, with the tolerance band tested at REAL values.

The four corners below are not plausible-looking constants. Each is a number
measured off the prepared datasets, so the tests keep testing the right thing
when the tolerance defaults change and they document the pathology that forced
the design. The near-zero corner in particular is a NYTimes query whose ten
nearest neighbours are the same vector repeated -- the case that broke the first
version of this band.
"""

from __future__ import annotations

import numpy as np
import pytest

from smartann.bench.metrics import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    recall_at_k,
    recall_at_k_batch,
    score_band,
    tie_rate,
)

# --- real values, measured off the prepared data ----------------------------
# NYTimes query 9777: its depth-10 neighbour is a duplicate of the query, so
# 1 - ip lands just BELOW zero in float64 while hnswlib's float32 lands further
# below. Both are correct; they differ by 4.67e-08.
NYT_DUP_GT64 = -1.9175753518396732e-07
NYT_DUP_F32 = -2.384185791015625e-07

NYT_TYPICAL = 0.61210734948344037  # median depth-10 distance, nytimes
SQUAD_TYPICAL = 0.1471977906533084  # median depth-10 distance, squad
SIFT_LARGE = 51109.0  # median depth-10 squared L2, sift1m
SIFT_MAX = 130947.0  # largest depth-10 squared L2, sift1m

CORNERS = [
    ("near-zero and negative (nytimes duplicate)", NYT_DUP_GT64),
    ("small positive (squad)", SQUAD_TYPICAL),
    ("order one (nytimes)", NYT_TYPICAL),
    ("large (sift squared L2)", SIFT_LARGE),
]


@pytest.mark.parametrize(("label", "threshold"), CORNERS)
def test_band_is_positive_and_finite_at_every_corner(label: str, threshold: float) -> None:
    band = float(score_band(threshold, DEFAULT_RTOL, DEFAULT_ATOL))
    assert band > 0.0, f"{label}: band collapsed to {band}"
    assert np.isfinite(band)


@pytest.mark.parametrize(("label", "threshold"), CORNERS)
def test_band_covers_the_measured_float32_noise_at_every_corner(
    label: str, threshold: float
) -> None:
    """float32 round-trip error must fall inside the band at every magnitude."""
    noise = abs(float(np.float32(threshold)) - threshold)
    assert noise <= float(score_band(threshold, DEFAULT_RTOL, DEFAULT_ATOL)), label


def test_pure_relative_band_fails_at_the_real_near_zero_value() -> None:
    """The bug this design exists to prevent, reproduced from measured numbers.

    A float32 retrieval of a duplicate vector is 24% away from the float64 truth
    in relative terms. With a relative-only band it scores WRONG; the neighbour
    is in fact the query's own duplicate and could not be more correct.
    """
    err = abs(NYT_DUP_F32 - NYT_DUP_GT64)
    relative_only = DEFAULT_RTOL * abs(NYT_DUP_GT64)
    assert err > relative_only * 1000, "relative-only must fail here, and by a wide margin"
    assert err / abs(NYT_DUP_GT64) == pytest.approx(0.2433, abs=1e-3)

    combined = float(score_band(NYT_DUP_GT64, DEFAULT_RTOL, DEFAULT_ATOL))
    assert err <= combined, "the combined band must accept it"


def test_pure_absolute_band_is_the_mirror_failure_on_sift() -> None:
    """Why the band is not just atol: 1e-6 means nothing against 1.3e+05."""
    noise = SIFT_MAX * 1e-7  # a plausible float32 relative slip at that scale
    assert noise > DEFAULT_ATOL, "absolute-only would reject a correct neighbour"
    assert noise <= float(score_band(SIFT_MAX, DEFAULT_RTOL, DEFAULT_ATOL))


# ---------------------------------------------------------------------------
# recall semantics
# ---------------------------------------------------------------------------
def _gt(dists: list[float]) -> tuple[np.ndarray, np.ndarray]:
    return np.arange(len(dists)), np.array(dists, dtype=np.float64)


def test_score_equality_removes_the_tie_ceiling() -> None:
    """The NYTimes case: 95 neighbours at the same distance, in miniature.

    Index-overlap caps this at 3/10 because the ground-truth generator broke the
    tie one way and the search broke it another. Nothing about that measures
    retrieval quality.
    """
    truth_d = np.array([0.5] * 12 + [0.9] * 88, dtype=np.float64)
    truth_i = np.arange(100)
    got_i = np.array([9, 10, 11, 0, 1, 2, 3, 4, 5, 6])  # 10 of the 12 tied
    got_d = np.full(10, 0.5)

    r = recall_at_k(got_i, got_d, truth_i, truth_d, k=10)
    assert r.score_equality == 1.0
    assert r.index_overlap == 0.8
    assert r.n_tied_in_truth == 12
    assert r.tie_gap == pytest.approx(0.2)


def test_the_two_metrics_agree_when_there_are_no_ties() -> None:
    truth_i, truth_d = _gt([0.1 * i for i in range(1, 21)])
    got_i = np.arange(10)
    r = recall_at_k(got_i, truth_d[:10], truth_i, truth_d, k=10)
    assert r.score_equality == 1.0 == r.index_overlap
    assert r.n_tied_in_truth == 1  # only the k-th matches itself


def test_a_worse_neighbour_outside_the_band_is_not_credited() -> None:
    """The failure mode the band must NOT introduce."""
    truth_i, truth_d = _gt([0.1 * i for i in range(1, 21)])
    got_d = [*truth_d[:9].tolist(), truth_d[9] + 0.05]  # 10th is genuinely worse
    r = recall_at_k(np.arange(10), np.array(got_d), truth_i, truth_d, k=10)
    assert r.score_equality == 0.9


def test_returning_fewer_than_k_loses_recall() -> None:
    """A short result list is a real loss, not a smaller denominator."""
    truth_i, truth_d = _gt([0.1 * i for i in range(1, 21)])
    r = recall_at_k(np.arange(6), truth_d[:6], truth_i, truth_d, k=10)
    assert r.score_equality == 0.6
    assert r.n_retrieved == 6


def test_duplicate_retrieved_ids_are_rejected() -> None:
    """A vector reachable from two cluster files would double-count."""
    truth_i, truth_d = _gt([0.1 * i for i in range(1, 21)])
    ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 8])
    with pytest.raises(ValueError, match="duplicates"):
        recall_at_k(ids, truth_d[:10], truth_i, truth_d, k=10)


def test_k_beyond_ground_truth_depth_is_rejected() -> None:
    truth_i, truth_d = _gt([0.1] * 5)
    with pytest.raises(ValueError, match="exceeds the ground-truth depth"):
        recall_at_k(np.arange(5), truth_d, truth_i, truth_d, k=10)


def test_batch_matches_the_reference_implementation() -> None:
    """A fast path that can disagree with the reference is worse than none."""
    rng = np.random.default_rng(4)
    n, depth, k = 60, 40, 10
    truth_d = np.sort(rng.random((n, depth)), axis=1)
    truth_i = np.tile(np.arange(depth), (n, 1))
    got_i = np.stack([rng.permutation(depth)[:k] for _ in range(n)])
    got_d = np.take_along_axis(truth_d, got_i, axis=1)

    se, ov = recall_at_k_batch(got_i, got_d, truth_i, truth_d, k=k)
    for i in range(n):
        r = recall_at_k(got_i[i], got_d[i], truth_i[i], truth_d[i], k=k)
        assert se[i] == pytest.approx(r.score_equality)
        assert ov[i] == pytest.approx(r.index_overlap)


def test_tie_rate_needs_depth_beyond_k() -> None:
    with pytest.raises(ValueError, match="depth > k"):
        tie_rate(np.zeros((5, 10)), k=10)


def test_tie_rate_counts_only_boundary_ties() -> None:
    truth = np.tile(np.arange(20, dtype=np.float64), (4, 1))
    assert tie_rate(truth, k=10) == 0.0
    truth[1, 10] = truth[1, 9]  # one query gains a tie just past the boundary
    assert tie_rate(truth, k=10) == 0.25

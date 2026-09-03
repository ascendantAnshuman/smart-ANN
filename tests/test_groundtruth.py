"""Phase 1: brute-force ground truth must be exact, deterministic and resumable."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from smartann.data.groundtruth import compute_topk, recall_at_k


def _unit(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    x = rng.standard_normal((n, d)).astype(np.float32)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


# ---------------------------------------------------------------------------
# Exactness against scipy
# ---------------------------------------------------------------------------
def test_ip_topk_matches_cdist_cosine() -> None:
    rng = np.random.default_rng(0)
    base, queries = _unit(rng, 5_000, 32), _unit(rng, 200, 32)

    got = compute_topk(base, queries, k=10, space="ip", block_mb=0.05, quiet=True)
    want = np.argsort(cdist(queries, base, metric="cosine"), axis=1)[:, :10]

    np.testing.assert_array_equal(got.neighbors, want.astype(np.int32))


def test_l2_topk_matches_cdist_euclidean() -> None:
    rng = np.random.default_rng(1)
    base = rng.standard_normal((3_000, 24)).astype(np.float32)
    queries = rng.standard_normal((150, 24)).astype(np.float32)

    got = compute_topk(base, queries, k=10, space="l2", block_mb=0.05, quiet=True)
    want = np.argsort(cdist(queries, base, metric="euclidean"), axis=1)[:, :10]

    np.testing.assert_array_equal(got.neighbors, want.astype(np.int32))


def test_block_size_does_not_change_the_answer() -> None:
    """Ground truth must not depend on how the work was chunked."""
    rng = np.random.default_rng(2)
    base, queries = _unit(rng, 2_000, 16), _unit(rng, 64, 16)

    small = compute_topk(base, queries, k=20, space="ip", block_mb=0.01, quiet=True)
    large = compute_topk(base, queries, k=20, space="ip", block_mb=100.0, quiet=True)

    np.testing.assert_array_equal(small.neighbors, large.neighbors)


def test_ties_break_by_ascending_index() -> None:
    """Duplicated vectors collide exactly; the lower index must win."""
    base = np.zeros((10, 4), dtype=np.float32)
    base[:, 0] = 1.0  # every row identical
    queries = np.zeros((1, 4), dtype=np.float32)
    queries[0, 0] = 1.0

    got = compute_topk(base, queries, k=5, space="ip", block_mb=0.001, quiet=True)
    np.testing.assert_array_equal(got.neighbors[0], np.arange(5, dtype=np.int32))


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------
def test_resume_from_checkpoint_gives_identical_output(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    base, queries = _unit(rng, 1_500, 16), _unit(rng, 40, 16)
    ckpt = tmp_path / "gt.ckpt.npz"

    full = compute_topk(base, queries, k=10, space="ip", block_mb=0.01, quiet=True)

    # Run once to populate the checkpoint, then run again: the second call must
    # resume rather than restart, and must land on the same answer.
    compute_topk(base, queries, k=10, space="ip", block_mb=0.01, checkpoint=ckpt, quiet=True)
    assert ckpt.exists()
    resumed = compute_topk(
        base, queries, k=10, space="ip", block_mb=0.01, checkpoint=ckpt, quiet=True
    )

    assert resumed.resumed_from_block > 0, "the checkpoint was not used"
    np.testing.assert_array_equal(resumed.neighbors, full.neighbors)


def test_stale_checkpoint_with_different_k_is_ignored(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    base, queries = _unit(rng, 500, 8), _unit(rng, 20, 8)
    ckpt = tmp_path / "gt.ckpt.npz"

    compute_topk(base, queries, k=5, space="ip", block_mb=0.01, checkpoint=ckpt, quiet=True)
    got = compute_topk(base, queries, k=15, space="ip", block_mb=0.01, checkpoint=ckpt, quiet=True)

    assert got.resumed_from_block == 0
    assert got.neighbors.shape == (20, 15)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
def test_k_larger_than_base_is_rejected() -> None:
    rng = np.random.default_rng(5)
    with pytest.raises(ValueError, match="exceeds the base set"):
        compute_topk(_unit(rng, 5, 4), _unit(rng, 2, 4), k=10, quiet=True)


def test_dim_mismatch_is_rejected() -> None:
    rng = np.random.default_rng(6)
    with pytest.raises(ValueError, match="dim mismatch"):
        compute_topk(_unit(rng, 50, 4), _unit(rng, 2, 8), k=3, quiet=True)


# ---------------------------------------------------------------------------
# recall@k
# ---------------------------------------------------------------------------
def test_recall_at_k_bounds() -> None:
    truth = np.arange(20, dtype=np.int32).reshape(2, 10)
    assert recall_at_k(truth.copy(), truth, 10).tolist() == [1.0, 1.0]

    wrong = truth + 1000
    assert recall_at_k(wrong, truth, 10).tolist() == [0.0, 0.0]


def test_recall_at_k_partial() -> None:
    truth = np.array([[0, 1, 2, 3]], dtype=np.int32)
    retrieved = np.array([[0, 1, 99, 98]], dtype=np.int32)
    assert recall_at_k(retrieved, truth, 4)[0] == pytest.approx(0.5)


def test_recall_at_k_rejects_excessive_depth() -> None:
    truth = np.zeros((1, 5), dtype=np.int32)
    with pytest.raises(ValueError, match="exceeds available ground truth depth"):
        recall_at_k(truth, truth, 10)

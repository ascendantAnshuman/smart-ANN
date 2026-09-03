"""Contract tests for the hnswlib binary we actually depend on (ADR-0017).

We install `chroma-hnswlib`, Chroma's redistribution of hnswlib, because
upstream `hnswlib` publishes no binary wheels for any Python version and MSVC
is unavailable on this machine. It is the same library under a different
distribution name, but "should be the same" is not a thing to assume about a
fork that the entire project sits on.

These tests pin the surface and the semantics Phase 2 relies on, so a version
bump that breaks any of them fails here rather than as unexplained recall loss
in Phase 8. The `1 - inner_product` convention in particular is a sign-error
waiting to happen: our ground truth stores raw similarities, the library returns
distances, and nothing about the types tells you they disagree.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

hnswlib = pytest.importorskip("hnswlib")

DIM = 16
N = 4000

# The API surface smartann/index/ uses. Anything removed upstream breaks us.
REQUIRED_METHODS = [
    "init_index",
    "add_items",
    "save_index",
    "load_index",
    "set_ef",
    "knn_query",
    "get_items",
    "get_current_count",
    "get_max_elements",
    "set_num_threads",
    "get_ids_list",
]


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    base = rng.standard_normal((N, DIM)).astype(np.float32)
    queries = rng.standard_normal((100, DIM)).astype(np.float32)
    return base, queries


@pytest.fixture(scope="module")
def l2_index(data: tuple[np.ndarray, np.ndarray]) -> object:
    base, _ = data
    idx = hnswlib.Index(space="l2", dim=DIM)
    idx.init_index(max_elements=N, M=16, ef_construction=200, random_seed=42)
    idx.add_items(base, np.arange(N))
    return idx


@pytest.mark.parametrize("method", REQUIRED_METHODS)
def test_required_method_exists(method: str) -> None:
    assert hasattr(hnswlib.Index(space="l2", dim=DIM), method), f"{method} is gone"


@pytest.mark.parametrize("space", ["l2", "ip", "cosine"])
def test_all_three_spaces_construct(space: str) -> None:
    """The registry can ask for any of these (ADR-0010)."""
    hnswlib.Index(space=space, dim=DIM)


def test_set_ef_changes_recall_monotonically(
    l2_index: object, data: tuple[np.ndarray, np.ndarray]
) -> None:
    """The premise the whole controller rests on.

    If per-query ef did not trade recall for work, there would be nothing for
    the adaptive layer to adapt.
    """
    base, queries = data
    exact = np.argsort((base * base).sum(1)[None, :] - 2.0 * (queries @ base.T), axis=1)[:, :10]

    recalls = []
    for ef in (8, 16, 32, 64, 128, 256):
        l2_index.set_ef(ef)  # type: ignore[attr-defined]
        labels, _ = l2_index.knn_query(queries, k=10)  # type: ignore[attr-defined]
        recalls.append(
            float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(labels, exact, strict=True)]))
        )

    assert all(b >= a - 1e-9 for a, b in itertools.pairwise(recalls)), recalls
    assert recalls[0] < recalls[-1], "ef had no effect at all"
    assert recalls[-1] >= 0.95, f"high effort only reached {recalls[-1]:.3f}"


def test_knn_query_returns_float32_distances(
    l2_index: object, data: tuple[np.ndarray, np.ndarray]
) -> None:
    """Score-equality recall needs distances, not just labels."""
    _, queries = data
    l2_index.set_ef(64)  # type: ignore[attr-defined]
    labels, dists = l2_index.knn_query(queries[:5], k=10)  # type: ignore[attr-defined]
    assert dists.shape == labels.shape == (5, 10)
    assert dists.dtype == np.float32


def test_l2_space_returns_squared_distance(
    l2_index: object, data: tuple[np.ndarray, np.ndarray]
) -> None:
    """Ground truth must be stored in the same units the library reports."""
    base, queries = data
    l2_index.set_ef(256)  # type: ignore[attr-defined]
    labels, dists = l2_index.knn_query(queries[:3], k=5)  # type: ignore[attr-defined]

    for qi in range(3):
        for li, d in zip(labels[qi], dists[qi], strict=True):
            squared = float(np.sum((queries[qi] - base[li]) ** 2))
            assert d == pytest.approx(squared, rel=1e-4), "l2 space is not squared distance"


def test_ip_space_returns_one_minus_inner_product(data: tuple[np.ndarray, np.ndarray]) -> None:
    """The sign convention that will otherwise cost a week.

    Our ground truth stores raw inner products (larger is better). hnswlib's
    "ip" space returns ``1 - inner_product`` (smaller is better). Comparing them
    directly would invert the ranking silently.
    """
    base, queries = data
    bn = (base / np.linalg.norm(base, axis=1, keepdims=True)).astype(np.float32)
    qn = (queries[:3] / np.linalg.norm(queries[:3], axis=1, keepdims=True)).astype(np.float32)

    idx = hnswlib.Index(space="ip", dim=DIM)
    idx.init_index(max_elements=N, M=16, ef_construction=200, random_seed=1)
    idx.add_items(bn, np.arange(N))
    idx.set_ef(256)
    labels, dists = idx.knn_query(qn, k=5)

    for qi in range(3):
        raw_ip = bn[labels[qi]] @ qn[qi]
        np.testing.assert_allclose(dists[qi], 1.0 - raw_ip, atol=1e-5)


def test_save_load_round_trip_is_exact(tmp_path, data: tuple[np.ndarray, np.ndarray]) -> None:  # type: ignore[no-untyped-def]
    """Per-cluster graphs are serialised and reloaded on every query."""
    base, queries = data
    original = hnswlib.Index(space="l2", dim=DIM)
    original.init_index(max_elements=N, M=16, ef_construction=200, random_seed=7)
    original.add_items(base, np.arange(N))
    original.set_ef(64)
    want_labels, want_dists = original.knn_query(queries[:10], k=10)

    path = tmp_path / "cluster.bin"
    original.save_index(str(path))
    assert path.stat().st_size > 0

    reloaded = hnswlib.Index(space="l2", dim=DIM)
    reloaded.load_index(str(path), max_elements=N)
    reloaded.set_ef(64)
    got_labels, got_dists = reloaded.knn_query(queries[:10], k=10)

    np.testing.assert_array_equal(got_labels, want_labels)
    np.testing.assert_allclose(got_dists, want_dists, rtol=1e-6)


def test_custom_labels_are_preserved(data: tuple[np.ndarray, np.ndarray]) -> None:
    """Per-cluster graphs store GLOBAL vector ids as labels, not local offsets.

    If labels were not preserved, every retrieved id would need a separate
    lookup table and the id-mapping bug surface would double.
    """
    base, queries = data
    labels_in = np.arange(1_000_000, 1_000_000 + 500, dtype=np.int64)
    idx = hnswlib.Index(space="l2", dim=DIM)
    idx.init_index(max_elements=500, M=16, ef_construction=100, random_seed=3)
    idx.add_items(base[:500], labels_in)
    idx.set_ef(100)

    out, _ = idx.knn_query(queries[:1], k=5)
    assert set(out[0].tolist()) <= set(labels_in.tolist())
    assert sorted(idx.get_ids_list()) == sorted(labels_in.tolist())


def test_k_larger_than_element_count_raises(data: tuple[np.ndarray, np.ndarray]) -> None:
    """Small clusters are the norm on real k-means output; the edge must be known."""
    base, queries = data
    idx = hnswlib.Index(space="l2", dim=DIM)
    idx.init_index(max_elements=5, M=16, ef_construction=100, random_seed=4)
    idx.add_items(base[:5], np.arange(5))
    idx.set_ef(50)

    with pytest.raises(RuntimeError):
        idx.knn_query(queries[:1], k=10)

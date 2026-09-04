"""EcoVector index: storage, build, search, and exhaustive equivalence.

Small synthetic corpora so the suite stays fast and offline. The real-data
checks live in scripts/phase2_acceptance.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from smartann.index.builder import build_ecovector, fit_partition, group_by_cluster
from smartann.index.ecovector import EcoVectorIndex, _brute_force
from smartann.index.storage import (
    CLUSTER_EMPTY,
    CLUSTER_FLAT,
    CLUSTER_GRAPH,
    IndexBuildConfig,
    IndexMeta,
    IndexPaths,
    cluster_kind,
    memory_model_mb,
)
from smartann.types import AnnConfig

pytest.importorskip("hnswlib")

DIM = 8


def _corpus(n: int = 600, dim: int = DIM, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def _unit(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture
def built(tmp_path: Path) -> tuple[EcoVectorIndex, np.ndarray]:
    """A small l2 index whose partition produces BOTH flat and graph clusters."""
    base = _corpus()
    out = IndexPaths(tmp_path / "idx")
    build_ecovector(
        base,
        dataset="synth",
        space="l2",
        nc=30,
        m=8,
        ef_construction=64,
        min_graph_size=20,
        seed=1,
        config_hash="deadbeef",
        out=out,
        quiet=True,
    )
    return EcoVectorIndex.open(out.root), base


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------
def test_cluster_kind_treats_empty_as_its_own_case() -> None:
    """An empty cluster has no file; a small one has a file searched by brute force."""
    assert cluster_kind(0, 32) == CLUSTER_EMPTY
    assert cluster_kind(1, 32) == CLUSTER_FLAT
    assert cluster_kind(31, 32) == CLUSTER_FLAT
    assert cluster_kind(32, 32) == CLUSTER_GRAPH


def test_empty_cluster_has_no_path() -> None:
    with pytest.raises(ValueError, match="no file on disk"):
        IndexPaths(Path("x")).cluster_file(0, CLUSTER_EMPTY)


def test_group_by_cluster_matches_the_naive_version() -> None:
    """The O(nc * n) version is the definition; this is the one that scales."""
    rng = np.random.default_rng(3)
    a = rng.integers(0, 12, size=500).astype(np.int32)
    order, offsets = group_by_cluster(a, 12)
    for cid in range(12):
        got = sorted(order[offsets[cid] : offsets[cid + 1]].tolist())
        assert got == sorted(np.flatnonzero(a == cid).tolist())


def test_memory_model_separates_its_terms() -> None:
    """A single total would hide which term the measurement disagreed with."""
    m = memory_model_mb(nc=2000, dim=128, n_base=1_000_000, largest_cluster=5000, m=32)
    assert set(m) == {"centroids", "centroid_graph", "assignments", "cluster_peak", "total"}
    assert m["total"] == pytest.approx(sum(v for k, v in m.items() if k != "total"))
    assert m["assignments"] == pytest.approx(1_000_000 * 4 / 2**20)


def test_index_meta_round_trips(tmp_path: Path, built: tuple[EcoVectorIndex, np.ndarray]) -> None:
    idx, _ = built
    p = tmp_path / "copy.json"
    idx.meta.to_json(p)
    assert IndexMeta.from_json(p) == idx.meta


def test_index_meta_rejects_unknown_keys(tmp_path: Path) -> None:
    """A meta.json from a different build version must not load silently."""
    p = tmp_path / "meta.json"
    p.write_text('{"dataset": "x", "surprise": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        IndexMeta.from_json(p)


def test_build_config_has_no_space_field() -> None:
    """The metric belongs to the dataset registry, not to the build config."""
    assert "space" not in IndexBuildConfig.__dataclass_fields__


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def test_build_produces_both_cluster_kinds(built: tuple[EcoVectorIndex, np.ndarray]) -> None:
    idx, _ = built
    assert idx.meta.n_graph_clusters > 0, "no graph clusters -- the flat path is untested"
    assert idx.meta.n_flat_clusters > 0, "no flat clusters -- the graph path is untested"
    assert idx.meta.n_graph_clusters + idx.meta.n_flat_clusters + idx.meta.n_empty_clusters == 30


def test_every_non_empty_cluster_has_its_file(built: tuple[EcoVectorIndex, np.ndarray]) -> None:
    idx, _ = built
    for cid, size in enumerate(idx.sizes):
        kind = cluster_kind(int(size), idx.meta.min_graph_size)
        if kind == CLUSTER_EMPTY:
            continue
        assert idx.paths.cluster_file(cid, kind).exists(), f"cluster {cid} ({kind}) has no file"


def test_every_vector_lands_in_exactly_one_cluster(
    built: tuple[EcoVectorIndex, np.ndarray],
) -> None:
    """The premise the duplicate-id rejection in recall_at_k relies on."""
    idx, base = built
    assert idx.assignments.shape[0] == base.shape[0]
    assert int(idx.sizes.sum()) == base.shape[0]


def test_spherical_kmeans_normalises_its_centroids() -> None:
    """An unnormalised centroid compares wrongly under inner product."""
    part = fit_partition(_unit(_corpus(400)), nc=12, space="ip", seed=2)
    np.testing.assert_allclose(np.linalg.norm(part.centroids, axis=1), 1.0, atol=1e-5)


def test_l2_kmeans_does_not_normalise() -> None:
    part = fit_partition(_corpus(400), nc=12, space="l2", seed=2)
    assert not np.allclose(np.linalg.norm(part.centroids, axis=1), 1.0)


def test_partition_reports_empty_and_duplicate_centroids() -> None:
    """Both are measured on every build; NYTimes is one-seventh duplicate rows."""
    part = fit_partition(_corpus(300), nc=10, space="l2", seed=5)
    assert part.n_empty == int(np.count_nonzero(part.sizes == 0))
    assert part.n_duplicate_centroids >= 0
    assert part.largest_duplicate_group >= 1


def test_partition_is_deterministic_under_a_fixed_seed() -> None:
    base = _corpus(300)
    a = fit_partition(base, nc=10, space="l2", seed=7)
    b = fit_partition(base, nc=10, space="l2", seed=7)
    np.testing.assert_array_equal(a.assignments, b.assignments)
    np.testing.assert_array_equal(a.centroids, b.centroids)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------
def test_probe_touches_only_the_centroid_graph(built: tuple[EcoVectorIndex, np.ndarray]) -> None:
    idx, base = built
    cfg = AnnConfig(nprobe=4, ef_centroid=16, ef_list=32, k=5, per_cluster_k=10)
    p = idx.probe(base[0], cfg)
    assert p.cluster_ids.size == 4
    assert p.centroid_dists.size == 4
    assert np.all(np.diff(p.centroid_dists) >= -1e-6), "probed clusters must be nearest-first"
    assert p.t_centroid_ms > 0.0


def test_probe_never_returns_an_empty_cluster(built: tuple[EcoVectorIndex, np.ndarray]) -> None:
    """The centroid graph indexes live clusters only, so this holds by construction."""
    idx, base = built
    cfg = AnnConfig(
        nprobe=min(8, int(np.count_nonzero(idx.sizes > 0))),
        ef_centroid=32,
        ef_list=32,
        k=5,
        per_cluster_k=10,
    )
    for qi in range(20):
        for cid in idx.probe(base[qi], cfg).cluster_ids:
            assert idx.sizes[int(cid)] > 0


def test_search_returns_k_sorted_results(built: tuple[EcoVectorIndex, np.ndarray]) -> None:
    idx, base = built
    cfg = AnnConfig(nprobe=6, ef_centroid=32, ef_list=64, k=5, per_cluster_k=10)
    r = idx.search(base[3], cfg)
    assert r.ids.size == 5
    assert np.all(np.diff(r.dists) >= -1e-9), "merged results must be sorted by distance"
    assert np.unique(r.ids).size == r.ids.size, "a vector belongs to exactly one cluster"


def test_search_ids_are_global_not_cluster_local(built: tuple[EcoVectorIndex, np.ndarray]) -> None:
    """A cluster-local id would index the wrong row of base.npy, silently."""
    idx, base = built
    cfg = AnnConfig(nprobe=30, ef_centroid=64, ef_list=64, k=5, per_cluster_k=20)
    r = idx.search(base[11], cfg)
    for got_id, got_d in zip(r.ids, r.dists, strict=True):
        want = float(np.sum((base[int(got_id)] - base[11]) ** 2))
        assert got_d == pytest.approx(want, rel=1e-4, abs=1e-5)


def test_flat_and_graph_clusters_report_the_same_units(
    built: tuple[EcoVectorIndex, np.ndarray],
) -> None:
    """ADR-0019's trap: mixed conventions sort wrong across cluster types.

    Compare a flat cluster's brute-force output against the graph library's for
    the same query. If one returned raw L2 and the other squared L2, a merged
    list would be silently misordered.
    """
    import hnswlib

    idx, base = built
    flat = [c for c, s in enumerate(idx.sizes) if cluster_kind(int(s), 20) == CLUSTER_FLAT]
    graph = [c for c, s in enumerate(idx.sizes) if cluster_kind(int(s), 20) == CLUSTER_GRAPH]
    assert flat and graph, "need both kinds to compare them"

    q = base[0]
    with np.load(idx.paths.cluster_file(flat[0], CLUSTER_FLAT)) as z:
        flat_d = _brute_force(q, np.asarray(z["vectors"]), "l2")
        flat_ids = np.asarray(z["ids"])

    g = hnswlib.Index(space="l2", dim=DIM)
    g.load_index(
        str(idx.paths.cluster_file(graph[0], CLUSTER_GRAPH)), max_elements=int(idx.sizes[graph[0]])
    )
    g.set_ef(64)
    labels, dists = g.knn_query(q.reshape(1, -1), k=1)

    # both must equal the same expression computed independently
    assert flat_d[0] == pytest.approx(float(np.sum((base[int(flat_ids[0])] - q) ** 2)), rel=1e-5)
    assert dists[0][0] == pytest.approx(float(np.sum((base[int(labels[0][0])] - q) ** 2)), rel=1e-4)


def test_instrumentation_splits_exact_from_modelled(
    built: tuple[EcoVectorIndex, np.ndarray],
) -> None:
    idx, base = built
    cfg = AnnConfig(nprobe=30, ef_centroid=64, ef_list=64, k=5, per_cluster_k=10)
    r = idx.search(base[0], cfg)
    assert r.n_flat_clusters_scanned > 0
    assert r.n_dist_exact > 0, "flat clusters count their comparisons exactly"
    assert r.n_dist_modeled > 0, "graph traversal can only be modelled"
    assert r.n_distance_computations == r.n_dist_exact + r.n_dist_modeled
    assert r.n_cluster_loads > 0
    assert r.bytes_read > 0
    assert r.t_total > 0.0


# ---------------------------------------------------------------------------
# negative distances -- real, not hypothetical
# ---------------------------------------------------------------------------
NYT_REAL_NEGATIVE = -1.9175753518396732e-07


def test_search_handles_negative_distances(tmp_path: Path) -> None:
    """NYTimes' depth-10 distance really reaches -1.918e-07.

    A duplicate vector's float64 inner product exceeds 1.0, so 1 - ip goes
    below zero, and hnswlib computes the same expression in float32. Nothing in
    the merge may assume otherwise.
    """
    base = _unit(_corpus(200, seed=9))
    base[5] = base[0]  # an exact duplicate, which is what produces the negative
    out = IndexPaths(tmp_path / "neg")
    build_ecovector(
        base,
        dataset="s",
        space="ip",
        nc=8,
        m=8,
        ef_construction=64,
        min_graph_size=10,
        seed=1,
        config_hash="h",
        out=out,
        quiet=True,
    )
    idx = EcoVectorIndex.open(out.root)
    r = idx.search(base[0], AnnConfig(nprobe=8, ef_centroid=16, ef_list=64, k=5, per_cluster_k=10))

    assert r.dists[0] <= 1e-6, "the query's own duplicate must come first"
    assert np.all(np.diff(r.dists) >= -1e-9), "sorting must survive negative values"
    assert np.isfinite(r.dists).all()


def test_brute_force_may_return_negative_and_that_is_fine() -> None:
    v = np.array([[1.0, 0.0]], dtype=np.float32)
    d = _brute_force(np.array([1.0, 0.0], dtype=np.float32), v, "ip")
    assert np.isfinite(d).all()
    assert abs(float(d[0])) < 1e-6


def test_entropy_is_defined_for_negative_distances() -> None:
    """The feature must not divide by, sqrt or log a distance."""
    from smartann.index.ecovector import _entropy

    d = np.array([NYT_REAL_NEGATIVE, 0.0, 0.3, 0.7])
    e = _entropy(d)
    assert np.isfinite(e) and e >= 0.0


# ---------------------------------------------------------------------------
# exhaustive equivalence -- acceptance check 1a, in miniature
# ---------------------------------------------------------------------------
def test_exhaustive_search_refuses_without_confirmation(
    built: tuple[EcoVectorIndex, np.ndarray],
) -> None:
    """It must not be reachable by accident from a normal run."""
    from smartann.index.validation import exhaustive_search

    idx, base = built
    with pytest.raises(ValueError, match="validation-only"):
        exhaustive_search(idx, base[0], k=5, confirm="")


def test_exhaustive_search_refuses_a_full_size_index(
    built: tuple[EcoVectorIndex, np.ndarray],
) -> None:
    """O(nc) file opens per query: 2000 on SIFT1M. The ceiling is deliberate."""
    from smartann.index.validation import CONFIRM, exhaustive_search

    idx, base = built
    with pytest.raises(ValueError, match="above the"):
        exhaustive_search(idx, base[0], k=5, confirm=CONFIRM, max_clusters=2)


def test_ann_config_cannot_request_exhaustive_search() -> None:
    """No flag on the config means no policy can set it and no YAML can carry it."""
    assert "exhaustive" not in AnnConfig.__dataclass_fields__


def test_exhaustive_ecovector_equals_brute_force(built: tuple[EcoVectorIndex, np.ndarray]) -> None:
    """Check 1a: with every cluster probed, the index is exact.

    A correctness proof for the id mapping, per-cluster storage, the flat and
    graph distance conventions and the merge -- everything except the
    approximation. Score equality rather than id equality is the criterion:
    duplicate vectors make two results interchangeable, and demanding identical
    ids would fail a correct implementation.
    """
    from smartann.index.validation import check_equivalence

    idx, base = built
    rep = check_equivalence(idx, base, base[:25], k=10)
    assert rep.ok, (
        f"{rep.n_score_equal}/{rep.n_queries} score-equal, worst {rep.max_distance_error:.3e}"
    )
    assert rep.max_distance_error < 1e-4


def test_exhaustive_equivalence_holds_for_ip_too(tmp_path: Path) -> None:
    """The convention that would invert silently gets its own pass."""
    from smartann.index.validation import check_equivalence

    base = _unit(_corpus(400, seed=12))
    out = IndexPaths(tmp_path / "ip")
    build_ecovector(
        base,
        dataset="s",
        space="ip",
        nc=20,
        m=8,
        ef_construction=64,
        min_graph_size=15,
        seed=3,
        config_hash="h",
        out=out,
        quiet=True,
    )
    idx = EcoVectorIndex.open(out.root)
    assert idx.meta.n_flat_clusters > 0 and idx.meta.n_graph_clusters > 0

    rep = check_equivalence(idx, base, base[:25], k=10)
    assert rep.ok, f"{rep.n_score_equal}/{rep.n_queries} score-equal"


def test_ivf_disk_variant_makes_every_cluster_flat(tmp_path: Path) -> None:
    """The ablation isolates the graphs by holding everything else identical."""
    from smartann.index.baselines import FLAT_EVERYTHING

    base = _corpus(300, seed=4)
    out = IndexPaths(tmp_path / "ivf")
    res = build_ecovector(
        base,
        dataset="s",
        space="l2",
        nc=10,
        m=8,
        ef_construction=64,
        min_graph_size=FLAT_EVERYTHING,
        seed=1,
        config_hash="h",
        out=out,
        index_variant="ivf_disk",
        quiet=True,
    )
    assert res.meta.n_graph_clusters == 0
    assert res.meta.n_flat_clusters + res.meta.n_empty_clusters == 10
    assert res.meta.index_variant == "ivf_disk"

    idx = EcoVectorIndex.open(out.root)
    r = idx.search(base[0], AnnConfig(nprobe=10, ef_centroid=16, ef_list=32, k=5, per_cluster_k=10))
    assert r.n_flat_clusters_scanned == r.n_cluster_loads
    assert r.n_dist_modeled == idx.probe(base[0], AnnConfig(10, 16, 32, 5, 10)).n_dist_modeled

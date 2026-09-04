"""The two index variants EcoVector is measured against.

``flat_hnsw`` is the ceiling: one graph over the whole corpus, entirely
resident, no I/O. It is what EcoVector gives up memory-for-CPU against, and on a
warm-cache desktop it will win on wall-clock. That is expected -- see the note
in :func:`compare_note`.

``ivf_disk`` is the ablation that isolates what the per-cluster *graphs*
contribute. It uses the same partition, the same files, the same load path, and
brute-forces inside each cluster instead of walking a graph. Because it is built
by the same function with ``min_graph_size`` set above the largest cluster, the
comparison holds everything else fixed by construction rather than by care.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import psutil

from smartann.index.ecovector import SearchResult, _anon_mb, _modelled_comparisons
from smartann.index.storage import IndexMeta, IndexPaths
from smartann.types import AnnConfig, QueryFeatures

__all__ = ["FLAT_EVERYTHING", "FlatHnswIndex", "build_flat_hnsw", "compare_note"]

# Any min_graph_size larger than the largest possible cluster makes every
# cluster flat, which is exactly the ivf_disk variant.
FLAT_EVERYTHING = 1 << 62


def compare_note() -> str:
    """The caveat that must travel with every EcoVector-vs-flat wall-clock number."""
    return (
        "flat_hnsw holds the whole corpus in RAM with no I/O; EcoVector trades CPU and "
        "I/O for a bounded memory footprint. A warm-cache desktop with 16 GB of RAM is "
        "the environment least favourable to that trade, so flat_hnsw is expected to win "
        "on wall-clock here. The ef-efficiency comparison is the one that is meaningful "
        "on this machine; the wall-clock comparison that matters happens on the phone "
        "in Phase 9."
    )


class FlatHnswIndex:
    """A single HNSW graph over the entire corpus, fully resident."""

    def __init__(self, root: str | Path) -> None:
        import hnswlib

        self.paths = IndexPaths(Path(root))
        self.meta = IndexMeta.from_json(self.paths.meta)
        self._proc = psutil.Process(os.getpid())
        self._graph = hnswlib.Index(space=self.meta.space, dim=self.meta.dim)
        self._graph.load_index(str(self.paths.centroid_graph), max_elements=self.meta.n_base)
        self._graph.set_num_threads(1)

    @classmethod
    def open(cls, root: str | Path) -> FlatHnswIndex:
        return cls(root)

    def search(self, query: np.ndarray, cfg: AnnConfig, token_len: int = 0) -> SearchResult:
        """Same return shape as EcoVector so the two are directly comparable.

        The probe-phase fields are zero rather than absent: a flat index has no
        centroid phase at all, and reporting 0.0 keeps the parquet schema
        identical across variants instead of introducing nulls that every
        downstream aggregate would then have to special-case.
        """
        q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        mem0 = self._proc.memory_info()

        t0 = time.perf_counter()
        self._graph.set_ef(max(cfg.ef_list, cfg.k))
        labels, dists = self._graph.knn_query(q, k=cfg.k)
        elapsed = (time.perf_counter() - t0) * 1000.0

        mem = self._proc.memory_info()
        return SearchResult(
            ids=labels[0].astype(np.int64),
            dists=dists[0].astype(np.float64),
            t_cluster_search=elapsed,
            t_total=elapsed,
            n_dist_modeled=_modelled_comparisons(
                max(cfg.ef_list, cfg.k), self.meta.m, self.meta.n_base
            ),
            anon_rss_mb=max(_anon_mb(mem0), _anon_mb(mem)),
            peak_rss_mb=max(mem0.rss, mem.rss) / 2**20,
            features=QueryFeatures(0.0, 0.0, 0.0, float(np.linalg.norm(q)), int(token_len)),
        )


def build_flat_hnsw(
    base: np.ndarray,
    *,
    dataset: str,
    space: str,
    m: int,
    ef_construction: int,
    seed: int,
    config_hash: str,
    out: IndexPaths,
    workers: int | None = None,
) -> IndexMeta:
    """Build the flat baseline, writing meta.json in the same shape as EcoVector.

    The graph is stored at ``centroid_graph.bin`` -- the name is wrong for this
    variant, but a shared layout means one loader, one manifest shape, and one
    place where a path can be got wrong.
    """
    import hnswlib

    from smartann.bench.manifest import library_versions

    proc = psutil.Process(os.getpid())
    rss0 = proc.memory_info().rss
    t0 = time.perf_counter()
    workers = workers or (os.cpu_count() or 1)
    n_base, dim = base.shape

    out.root.mkdir(parents=True, exist_ok=True)
    g = hnswlib.Index(space=space, dim=dim)
    g.init_index(max_elements=n_base, M=m, ef_construction=ef_construction, random_seed=seed)
    g.set_num_threads(workers)
    g.add_items(np.ascontiguousarray(base, dtype=np.float32), np.arange(n_base, dtype=np.int64))
    g.save_index(str(out.centroid_graph))
    peak = proc.memory_info().rss
    del g

    meta = IndexMeta(
        dataset=dataset,
        space=space,
        dim=int(dim),
        n_base=int(n_base),
        nc=0,
        m=int(m),
        ef_construction=int(ef_construction),
        min_graph_size=0,
        seed=int(seed),
        config_hash=config_hash,
        index_variant="flat_hnsw",
        built_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        build_seconds=time.perf_counter() - t0,
        build_peak_rss_mb=(peak - rss0) / 2**20,
        build_workers=int(workers),
        kmeans_algorithm="none",
        n_empty_clusters=0,
        n_flat_clusters=0,
        n_graph_clusters=1,
        largest_cluster=int(n_base),
        smallest_nonempty_cluster=int(n_base),
        n_duplicate_centroids=0,
        bytes_on_disk=out.centroid_graph.stat().st_size,
        has_per_cluster_graphs=False,
        library_versions=library_versions(),
    )
    meta.to_json(out.meta)
    return meta

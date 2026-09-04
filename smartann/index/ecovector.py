"""EcoVector search: a cheap RAM-resident probe, then per-cluster work on disk.

The two-phase split is the whole reason this project can have an adaptive
controller. ``probe()`` touches only the centroid graph, costs almost nothing,
and yields :class:`~smartann.types.QueryFeatures` -- signals about how hard the
query looks *before* the expensive part runs. ``finish()`` then spends whatever
effort the policy decided on, cluster by cluster, holding one at a time.

There is no exact-rerank stage (ADR-0019). hnswlib returns true distances to the
candidates it finds, so merging per-cluster results by those distances already
yields the exact best-k among everything examined.

**Distances may be negative.** For a duplicate vector the inner product of two
float32-normalised rows can exceed 1.0, so ``1 - ip`` lands just below zero --
measured at -1.918e-07 in the NYTimes ground truth, and hnswlib computes the
same expression in float32. Nothing here may assert non-negativity, take a
square root or a logarithm of a distance, or divide by one. Sorting handles
negatives correctly; almost nothing else does.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import psutil

from smartann.index.storage import (
    CLUSTER_EMPTY,
    CLUSTER_FLAT,
    IndexMeta,
    IndexPaths,
    cluster_kind,
)
from smartann.types import AnnConfig, QueryFeatures

__all__ = ["EcoVectorIndex", "ProbeResult", "SearchResult"]


@dataclass(frozen=True)
class ProbeResult:
    """Output of phase one: which clusters to visit, and how hard this looks."""

    cluster_ids: np.ndarray  # (nprobe,) int64, nearest first
    centroid_dists: np.ndarray  # (nprobe,) float64, same order
    features: QueryFeatures
    t_centroid_ms: float
    n_dist_modeled: int


@dataclass
class SearchResult:
    """Output of phase two, with the cost of producing it."""

    ids: np.ndarray
    dists: np.ndarray
    t_centroid: float = 0.0
    t_cluster_load: float = 0.0
    t_cluster_search: float = 0.0
    t_merge: float = 0.0
    t_total: float = 0.0
    n_cluster_loads: int = 0
    n_flat_clusters_scanned: int = 0
    n_dist_exact: int = 0
    n_dist_modeled: int = 0
    bytes_read: int = 0
    anon_rss_mb: float = 0.0
    peak_rss_mb: float = 0.0
    features: QueryFeatures | None = None
    per_cluster_found: list[int] = field(default_factory=list)

    @property
    def n_distance_computations(self) -> int:
        return self.n_dist_exact + self.n_dist_modeled


def _entropy(d: np.ndarray) -> float:
    """Shannon entropy of the softmax over negated centroid distances.

    Negated because these are distances: the nearest centroid should carry the
    most mass. Shifted by the max before exponentiating for stability, which is
    also why nothing here divides by a distance.
    """
    if d.size < 2:
        return 0.0
    w = np.exp(-(d - d.min()))
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return 0.0
    p = w / total
    nz = p[p > 0]
    return float(-np.sum(nz * np.log(nz)))


class EcoVectorIndex:
    """A built index, opened for search.

    Opening loads only the centroid graph, the centroid array and the
    assignment map. Cluster graphs are read per query and released.
    """

    def __init__(self, paths: IndexPaths, meta: IndexMeta) -> None:
        import hnswlib

        self.paths = paths
        self.meta = meta
        self._proc = psutil.Process(os.getpid())

        self.centroids: np.ndarray = np.load(paths.centroids)
        self.assignments: np.ndarray = np.load(paths.assignments, mmap_mode="r")
        self.sizes: np.ndarray = np.bincount(
            np.asarray(self.assignments), minlength=meta.nc
        ).astype(np.int64)
        self._kinds = [cluster_kind(int(s), meta.min_graph_size) for s in self.sizes]

        self._centroid_graph = hnswlib.Index(space=meta.space, dim=meta.dim)
        self._centroid_graph.load_index(
            str(paths.centroid_graph), max_elements=int(np.count_nonzero(self.sizes > 0))
        )
        self._centroid_graph.set_num_threads(1)

    @classmethod
    def open(cls, root: str | Path) -> EcoVectorIndex:
        paths = IndexPaths(Path(root))
        if not paths.meta.exists():
            raise FileNotFoundError(f"no index at {root} (meta.json missing)")
        return cls(paths, IndexMeta.from_json(paths.meta))

    # -- phase one ---------------------------------------------------------
    def probe(self, query: np.ndarray, cfg: AnnConfig, token_len: int = 0) -> ProbeResult:
        """Search the resident centroid graph. Cheap, and the only input a
        policy gets before committing to an effort level."""
        q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        n_live = int(np.count_nonzero(self.sizes > 0))
        nprobe = min(cfg.nprobe, n_live)

        t0 = time.perf_counter()
        self._centroid_graph.set_ef(max(cfg.ef_centroid, nprobe))
        labels, dists = self._centroid_graph.knn_query(q, k=nprobe)
        t_centroid_ms = (time.perf_counter() - t0) * 1000.0

        cids = labels[0].astype(np.int64)
        cd = dists[0].astype(np.float64)

        # d2/d1 as a ratio would divide by a distance that can be zero or
        # negative (duplicate vectors). The difference carries the same
        # "is this query ambiguous" signal and is always defined.
        gap = float(cd[1] - cd[0]) if cd.size > 1 else 0.0
        return ProbeResult(
            cluster_ids=cids,
            centroid_dists=cd,
            features=QueryFeatures(
                centroid_top1_dist=float(cd[0]),
                centroid_dist_gap=gap,
                centroid_dist_entropy=_entropy(cd),
                query_norm=float(np.linalg.norm(q)),
                query_token_len=int(token_len),
            ),
            t_centroid_ms=t_centroid_ms,
            n_dist_modeled=max(cfg.ef_centroid, nprobe) * self.meta.m,
        )

    # -- phase two ---------------------------------------------------------
    def finish(self, query: np.ndarray, probe: ProbeResult, cfg: AnnConfig) -> SearchResult:
        """Visit the probed clusters, then merge. One cluster resident at a time."""
        import hnswlib

        q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        res = SearchResult(
            ids=np.empty(0, dtype=np.int64),
            dists=np.empty(0, dtype=np.float64),
            features=probe.features,
        )
        res.t_centroid = probe.t_centroid_ms
        res.n_dist_modeled = probe.n_dist_modeled

        t_start = time.perf_counter()
        mem0 = self._proc.memory_info()
        anon_peak = _anon_mb(mem0)
        rss_peak = mem0.rss / 2**20

        found_ids: list[np.ndarray] = []
        found_dists: list[np.ndarray] = []

        for cid in probe.cluster_ids:
            kind = self._kinds[int(cid)]
            if kind == CLUSTER_EMPTY:
                # Unreachable via the centroid graph, which only indexes live
                # clusters -- but a stale probe list must not crash the search.
                continue
            size = int(self.sizes[int(cid)])
            path = self.paths.cluster_file(int(cid), kind)

            t0 = time.perf_counter()
            if kind == CLUSTER_FLAT:
                with np.load(path) as z:
                    ids_c = np.asarray(z["ids"], dtype=np.int64)
                    vecs = np.asarray(z["vectors"], dtype=np.float32)
                res.t_cluster_load += (time.perf_counter() - t0) * 1000.0

                t1 = time.perf_counter()
                d = _brute_force(q[0], vecs, self.meta.space)
                take = min(cfg.per_cluster_k, d.size)
                sel = np.argpartition(d, take - 1)[:take] if take < d.size else np.arange(d.size)
                sel = sel[np.argsort(d[sel], kind="stable")]
                got_i, got_d = ids_c[sel], d[sel]
                res.t_cluster_search += (time.perf_counter() - t1) * 1000.0
                res.n_flat_clusters_scanned += 1
                res.n_dist_exact += size  # brute force: exactly this many, counted
            else:
                g = hnswlib.Index(space=self.meta.space, dim=self.meta.dim)
                g.load_index(str(path), max_elements=size)
                g.set_num_threads(1)
                res.t_cluster_load += (time.perf_counter() - t0) * 1000.0

                t1 = time.perf_counter()
                g.set_ef(max(cfg.ef_list, cfg.per_cluster_k))
                # hnswlib raises when k exceeds the element count; a cluster can
                # be smaller than per_cluster_k without being flat.
                labels, dists = g.knn_query(q, k=min(cfg.per_cluster_k, size))
                got_i = labels[0].astype(np.int64)
                got_d = dists[0].astype(np.float64)
                res.t_cluster_search += (time.perf_counter() - t1) * 1000.0
                res.n_dist_modeled += max(cfg.ef_list, cfg.per_cluster_k) * self.meta.m
                del g  # release before opening the next one

            found_ids.append(got_i)
            found_dists.append(got_d)
            res.per_cluster_found.append(int(got_i.size))
            res.n_cluster_loads += 1
            res.bytes_read += path.stat().st_size

            mem = self._proc.memory_info()
            anon_peak = max(anon_peak, _anon_mb(mem))
            rss_peak = max(rss_peak, mem.rss / 2**20)

        t_merge = time.perf_counter()
        if found_ids:
            all_i = np.concatenate(found_ids)
            all_d = np.concatenate(found_dists)
            # lexsort's last key is primary; ascending id breaks distance ties
            # deterministically, matching the ground-truth generator.
            order = np.lexsort((all_i, all_d))[: cfg.k]
            res.ids, res.dists = all_i[order], all_d[order]
        res.t_merge = (time.perf_counter() - t_merge) * 1000.0
        res.t_total = (time.perf_counter() - t_start) * 1000.0 + probe.t_centroid_ms
        res.anon_rss_mb = anon_peak
        res.peak_rss_mb = rss_peak
        return res

    def search(self, query: np.ndarray, cfg: AnnConfig, token_len: int = 0) -> SearchResult:
        return self.finish(query, self.probe(query, cfg, token_len), cfg)


def _brute_force(q: np.ndarray, vecs: np.ndarray, space: str) -> np.ndarray:
    """Exact distances in hnswlib's convention, so a flat cluster merges with a
    graph cluster without any special handling.

    This is the trap ADR-0019 flagged: a flat cluster returning raw L2 while a
    graph cluster returns squared L2 would sort wrong across cluster types, with
    no symptom other than slightly worse recall.
    """
    if space == "l2":
        diff = vecs - q[None, :]
        out = np.einsum("ij,ij->i", diff, diff)
    else:
        out = 1.0 - vecs @ q
    return np.asarray(out, dtype=np.float64)


def _anon_mb(mem: object) -> float:
    """Anonymous (non-file-backed) resident memory, in MB.

    On Windows psutil has no ``uss``/``shared`` split on ``memory_info``, so we
    take the private working set when available and fall back to RSS. The
    fallback is an over-estimate, never an under-estimate, so a bound computed
    from it stays honest.
    """
    for attr in ("private", "uss"):
        value = getattr(mem, attr, None)
        if value:
            return float(value) / 2**20
    rss = float(getattr(mem, "rss", 0.0))
    shared = float(getattr(mem, "shared", 0.0))
    return (rss - shared) / 2**20

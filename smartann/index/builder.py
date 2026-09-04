"""Building an EcoVector index: partition, centroid graph, per-cluster graphs.

The build is the only place the corpus is fully resident, and even here the
per-cluster graphs are written and released one at a time.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil

from smartann.data.readers import duplicate_report
from smartann.index.storage import (
    CLUSTER_EMPTY,
    CLUSTER_FLAT,
    IndexMeta,
    IndexPaths,
    cluster_kind,
)

__all__ = [
    "BuildResult",
    "KMeansResult",
    "assign_to_centroids",
    "build_ecovector",
    "fit_partition",
    "group_by_cluster",
]

# MiniBatch rather than full Lloyd, and it is not a shortcut: one Lloyd
# iteration over SIFT1M with nc=2000 is 1e6 x 2000 x 128 = 2.6e11 multiply-adds,
# so a 100-iteration fit is days. MiniBatchKMeans reaches a comparable partition
# in minutes and is deterministic under a fixed random_state. Recorded in
# meta.json as kmeans_algorithm so no result can be read as full k-means.
KMEANS_ALGORITHM = "MiniBatchKMeans"


@dataclass(frozen=True)
class KMeansResult:
    centroids: np.ndarray  # (nc, dim) float32; unit-norm when space is ip
    assignments: np.ndarray  # (n_base,) int32
    sizes: np.ndarray  # (nc,) int64
    n_empty: int
    n_duplicate_centroids: int
    largest_duplicate_group: int
    seconds: float


@dataclass(frozen=True)
class BuildResult:
    meta: IndexMeta
    paths: IndexPaths


def fit_partition(
    base: np.ndarray,
    nc: int,
    space: str,
    seed: int,
    batch_size: int = 4096,
    max_iter: int = 100,
) -> KMeansResult:
    """Partition *base* into *nc* clusters, spherically when the space is ip.

    Two properties are measured rather than assumed, both because NYTimes is
    one-seventh byte-identical duplicate rows (ADR-0014 amendment):

    * **Empty clusters.** Heavy duplicate concentration makes them likelier, and
      an empty cluster is a distinct case from a small one -- it has no file and
      must never be probed.
    * **Duplicate centroids.** Two byte-identical centroids let probe() spend two
      of its nprobe slots reaching different clusters through indistinguishable
      entries. Not incorrect, but wasted budget.
    """
    from sklearn.cluster import MiniBatchKMeans

    if nc < 1 or nc > base.shape[0]:
        raise ValueError(f"nc={nc} is not usable for {base.shape[0]} vectors")

    t0 = time.perf_counter()
    x = np.ascontiguousarray(base, dtype=np.float32)
    km = MiniBatchKMeans(
        n_clusters=nc,
        random_state=seed,
        batch_size=batch_size,
        max_iter=max_iter,
        n_init=3,
        verbose=0,
    )
    km.fit(x)
    centroids = np.ascontiguousarray(km.cluster_centers_, dtype=np.float32)

    if space in ("ip", "cosine"):
        # Spherical k-means: the mean of unit vectors is not itself a unit
        # vector, and an unnormalised centroid compares wrongly under inner
        # product. A zero-norm centroid cannot arise from unit-norm members, but
        # the guard is free and a silent NaN row would be very expensive.
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        if not np.all(norms > 0):
            raise ValueError(f"{int(np.sum(norms <= 0))} centroid(s) collapsed to zero norm")
        centroids = (centroids / norms).astype(np.float32)

    assignments = assign_to_centroids(x, centroids, space)
    sizes = np.bincount(assignments, minlength=nc).astype(np.int64)
    dup = duplicate_report(centroids)

    return KMeansResult(
        centroids=centroids,
        assignments=assignments,
        sizes=sizes,
        n_empty=int(np.count_nonzero(sizes == 0)),
        n_duplicate_centroids=dup.n_repeat_rows,
        largest_duplicate_group=dup.largest_group,
        seconds=time.perf_counter() - t0,
    )


def assign_to_centroids(
    x: np.ndarray, centroids: np.ndarray, space: str, block: int = 50_000
) -> np.ndarray:
    """Nearest centroid for every row, in the index's own metric.

    Done explicitly rather than through km.predict because sklearn always uses
    euclidean distance. For unit vectors that happens to agree with maximum
    inner product, but relying on the coincidence would break the moment a
    non-normalised ip dataset appeared.
    """
    n = x.shape[0]
    out = np.empty(n, dtype=np.int32)
    c = np.ascontiguousarray(centroids, dtype=np.float32)
    c_sq = (c * c).sum(1)
    for lo in range(0, n, block):
        xb = x[lo : lo + block]
        if space in ("ip", "cosine"):
            out[lo : lo + block] = np.argmax(xb @ c.T, axis=1)
        else:
            out[lo : lo + block] = np.argmin(c_sq[None, :] - 2.0 * (xb @ c.T), axis=1)
    return out


def group_by_cluster(assignments: np.ndarray, nc: int) -> tuple[np.ndarray, np.ndarray]:
    """Member ids per cluster, as (sorted_ids, offsets).

    One stable argsort over the whole assignment array. The obvious
    flatnonzero(assignments == cid) per cluster is O(nc * n_base) -- two billion
    comparisons on SIFT1M with nc=2000, for a result one sort already gives.
    """
    order = np.argsort(assignments, kind="stable").astype(np.int64)
    sizes = np.bincount(assignments, minlength=nc)
    offsets = np.zeros(nc + 1, dtype=np.int64)
    np.cumsum(sizes, out=offsets[1:])
    return order, offsets


def build_ecovector(
    base: np.ndarray,
    *,
    dataset: str,
    space: str,
    nc: int,
    m: int,
    ef_construction: int,
    min_graph_size: int,
    seed: int,
    config_hash: str,
    out: IndexPaths,
    index_variant: str = "ecovector",
    workers: int | None = None,
    quiet: bool = False,
) -> BuildResult:
    """Build and write a complete EcoVector index."""
    import hnswlib

    from smartann.bench.manifest import library_versions

    proc = psutil.Process(os.getpid())
    rss0 = proc.memory_info().rss
    peak = rss0
    t_start = time.perf_counter()
    workers = workers or (os.cpu_count() or 1)

    n_base, dim = base.shape
    part = fit_partition(base, nc, space, seed)
    peak = max(peak, proc.memory_info().rss)
    if not quiet:
        print(
            f"  k-means: {part.seconds:.1f}s, {part.n_empty} empty cluster(s), "
            f"{part.n_duplicate_centroids} duplicate centroid(s) "
            f"(largest group {part.largest_duplicate_group})"
        )

    out.clusters.mkdir(parents=True, exist_ok=True)
    order, offsets = group_by_cluster(part.assignments, nc)
    kinds = [cluster_kind(int(s), min_graph_size) for s in part.sizes]

    for cid in range(nc):
        kind = kinds[cid]
        if kind == CLUSTER_EMPTY:
            continue
        ids = order[offsets[cid] : offsets[cid + 1]]
        vecs = np.ascontiguousarray(base[ids], dtype=np.float32)
        path = out.cluster_file(cid, kind)
        if kind == CLUSTER_FLAT:
            # Too few members for a graph to help, and hnswlib raises outright
            # when k exceeds the element count (test_hnswlib_contract). Brute
            # force over a handful of vectors is exact and faster anyway.
            np.savez(path, ids=ids.astype(np.int32), vectors=vecs)
        else:
            g = hnswlib.Index(space=space, dim=dim)
            g.init_index(
                max_elements=len(ids), M=m, ef_construction=ef_construction, random_seed=seed
            )
            g.set_num_threads(workers)
            g.add_items(vecs, ids.astype(np.int64))
            g.save_index(str(path))
            del g
        peak = max(peak, proc.memory_info().rss)

    # Centroid graph over NON-EMPTY clusters only, so probe() cannot return a
    # cluster that has no file at all. Labels are cluster ids, not row offsets.
    live = np.flatnonzero(part.sizes > 0).astype(np.int64)
    cg = hnswlib.Index(space=space, dim=dim)
    cg.init_index(max_elements=len(live), M=m, ef_construction=ef_construction, random_seed=seed)
    cg.set_num_threads(workers)
    cg.add_items(np.ascontiguousarray(part.centroids[live]), live)
    cg.save_index(str(out.centroid_graph))
    del cg

    np.save(out.centroids, part.centroids)
    np.save(out.assignments, part.assignments)
    peak = max(peak, proc.memory_info().rss)

    nonempty = part.sizes[part.sizes > 0]
    meta = IndexMeta(
        dataset=dataset,
        space=space,
        dim=int(dim),
        n_base=int(n_base),
        nc=int(nc),
        m=int(m),
        ef_construction=int(ef_construction),
        min_graph_size=int(min_graph_size),
        seed=int(seed),
        config_hash=config_hash,
        index_variant=index_variant,
        built_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        build_seconds=time.perf_counter() - t_start,
        build_peak_rss_mb=(peak - rss0) / 2**20,
        build_workers=int(workers),
        kmeans_algorithm=KMEANS_ALGORITHM,
        n_empty_clusters=int(part.n_empty),
        n_flat_clusters=kinds.count(CLUSTER_FLAT),
        n_graph_clusters=len(kinds) - kinds.count(CLUSTER_FLAT) - part.n_empty,
        largest_cluster=int(part.sizes.max()),
        smallest_nonempty_cluster=int(nonempty.min()) if nonempty.size else 0,
        n_duplicate_centroids=int(part.n_duplicate_centroids),
        bytes_on_disk=_dir_bytes(out.root),
        library_versions=library_versions(),
    )
    meta.to_json(out.meta)
    return BuildResult(meta=meta, paths=out)


def _dir_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())

"""On-disk layout for an EcoVector index, and the memory model it implies.

    <data_root>/index/<dataset>/<config_hash>/
        meta.json            build parameters, versions, and the build report
        centroids.npy        (nc, dim) float32 -- resident
        centroid_graph.bin   hnswlib graph over centroids, labels are cluster ids
        assignments.npy      (n_base,) int32 -- resident, the 8N term
        clusters/NNNNN.bin   one hnswlib graph per cluster, labels are GLOBAL ids
        clusters/NNNNN.npz   ...or ids+vectors for clusters too small for a graph

Only the first three are held in RAM. Cluster files are opened, searched and
released one at a time, which is the whole point of the design: peak resident
memory is bounded by the largest single cluster rather than by the corpus.

Addressed by config hash so that two builds with different parameters cannot
overwrite each other, and so a parquet row's ``index_config_hash`` names the
artifact that produced it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "CLUSTER_EMPTY",
    "CLUSTER_FLAT",
    "CLUSTER_GRAPH",
    "IndexBuildConfig",
    "IndexMeta",
    "IndexPaths",
    "cluster_kind",
    "memory_model_mb",
    "resolve_index_root",
]

CLUSTER_EMPTY = "empty"
CLUSTER_FLAT = "flat"
CLUSTER_GRAPH = "graph"


def cluster_kind(size: int, min_graph_size: int) -> str:
    """How a cluster of *size* members is stored.

    Three cases, not two. An empty cluster is not a very small cluster: it has
    no file at all and must never be probed, whereas a flat cluster has a file
    and is searched by brute force. k-means on a corpus that is one-seventh
    duplicate vectors (NYTimes, ADR-0014 amendment) produces both.
    """
    if size <= 0:
        return CLUSTER_EMPTY
    return CLUSTER_FLAT if size < min_graph_size else CLUSTER_GRAPH


@dataclass(frozen=True)
class IndexBuildConfig:
    """What a build is parameterised by. Hashed to address the artifact.

    There is deliberately no ``space`` field. The metric belongs to the dataset
    and is read from the registry at build time; declaring it here would be a
    second source of truth for the one thing that silently halves recall when
    the two disagree.
    """

    dataset: str
    nc: int | str = "auto"  # "auto" resolves to round(sqrt(n_base)) via the registry
    m: int = 32
    ef_construction: int = 200
    # Default is 'every cluster flat'. Measurement A found no M at which the
    # per-cluster graphs beat brute force at matched recall (ADR-0028); a
    # smaller value restores them for the ablation.
    min_graph_size: int = 1 << 62
    seed: int = 42
    index_variant: str = "ecovector"  # ecovector | flat_hnsw | ivf_disk
    # k-means settings live here, and therefore in the config hash, because they
    # change the artifact. Leaving them as function defaults would let two
    # materially different partitions share one hash -- exactly the collision
    # the hash exists to prevent (Section 3 rule 4).
    kmeans_batch_size: int | str = "auto"  # "auto" -> 10 * nc, per sklearn guidance
    kmeans_n_init: int = 1


@dataclass(frozen=True)
class IndexPaths:
    """Where every artifact of one built index lives."""

    root: Path

    @property
    def meta(self) -> Path:
        return self.root / "meta.json"

    @property
    def centroids(self) -> Path:
        return self.root / "centroids.npy"

    @property
    def centroid_graph(self) -> Path:
        return self.root / "centroid_graph.bin"

    @property
    def assignments(self) -> Path:
        return self.root / "assignments.npy"

    @property
    def clusters(self) -> Path:
        return self.root / "clusters"

    def cluster_file(self, cid: int, kind: str) -> Path:
        suffix = {CLUSTER_GRAPH: ".bin", CLUSTER_FLAT: ".npy"}.get(kind)
        if suffix is None:
            raise ValueError(f"cluster kind {kind!r} has no file on disk")
        return self.clusters / f"{cid:05d}{suffix}"

    def cluster_ids_file(self, cid: int) -> Path:
        """Global ids for a flat cluster, beside its vectors.

        Two plain .npy files rather than one .npz, because np.load can
        memory-map a .npy and cannot map inside a zip archive. Mapping is the
        whole point: it gives zero-copy access with no reconstruction, which is
        most of what a cluster cache would have bought without introducing a
        cache (ADR-0029).
        """
        return self.clusters / f"{cid:05d}.ids.npy"

    @classmethod
    def for_build(cls, index_root: Path, dataset: str, config_hash: str) -> IndexPaths:
        return cls(Path(index_root) / dataset / config_hash)


@dataclass(frozen=True)
class IndexMeta:
    """Everything needed to reopen an index and to trust what it contains."""

    dataset: str
    space: str  # read from the dataset registry, never declared independently
    dim: int
    n_base: int
    nc: int
    m: int
    ef_construction: int
    min_graph_size: int
    seed: int
    config_hash: str
    index_variant: str
    built_at: str
    build_seconds: float
    build_peak_rss_mb: float
    build_workers: int
    kmeans_algorithm: str
    # build report -- measured, not assumed
    n_empty_clusters: int
    n_flat_clusters: int
    n_graph_clusters: int
    largest_cluster: int
    smallest_nonempty_cluster: int
    n_duplicate_centroids: int
    bytes_on_disk: int
    # EcoVector's defining component. False means this artifact is IVF-HNSW with
    # mmap'd flat clusters, NOT EcoVector -- the name on index_variant is kept
    # for continuity but does not describe what was built (ADR-0028).
    has_per_cluster_graphs: bool = False
    library_versions: dict[str, str] = field(default_factory=dict)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", "utf-8")

    @classmethod
    def from_json(cls, path: Path) -> IndexMeta:
        data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"{path} has unknown keys {unknown}; built by a different version?")
        return cls(**data)


def memory_model_mb(
    nc: int, dim: int, n_base: int, largest_cluster: int, m: int, itemsize: int = 4
) -> dict[str, float]:
    """Predicted resident footprint, term by term (Section 3.4).

    Terms, all in bytes before conversion:

    * ``centroids``   ``nc * dim * itemsize`` -- always resident
    * ``centroid_graph`` roughly ``nc * M * 2 * 4`` for the link lists
    * ``assignments`` ``n_base * 4`` -- the id-to-cluster map, the ``8N`` term
      in MobileRAG's expression at 8 bytes; we store int32, so 4
    * ``cluster_peak`` the largest single cluster's vectors plus its links,
      because exactly one cluster is open at a time

    Returned separately rather than summed: check 9 compares the prediction to
    a measurement, and a single number would hide which term was wrong.
    """
    centroids = nc * dim * itemsize
    centroid_graph = nc * m * 2 * 4
    assignments = n_base * 4
    cluster_peak = largest_cluster * (dim * itemsize + m * 2 * 4)
    terms = {
        "centroids": centroids,
        "centroid_graph": centroid_graph,
        "assignments": assignments,
        "cluster_peak": cluster_peak,
    }
    out = {k: v / 2**20 for k, v in terms.items()}
    out["total"] = sum(out.values())
    return out


def resolve_index_root(
    dataset: str,
    variant: str = "ecovector",
    config_dir: Path | None = None,
) -> IndexPaths | None:
    """The index built from `configs/index/<dataset>_<variant>.yaml`, or None.

    Derived from the config rather than discovered on disk. Picking the
    lexicographically largest hash, or even the newest mtime, means a caller
    silently describes whichever artifact happened to sort or land last, which
    is fine until the first `--force` rebuild and wrong thereafter with no
    symptom. The config file is what the build was told to do, so it is what
    resolution should ask about.

    Lives here rather than in `scripts/` because both the acceptance scripts and
    the library's own query path need it, and two resolvers would eventually
    disagree about which artifact a result came from.

    Returns None only when the index is genuinely not built. A missing config
    *directory* is a broken checkout rather than a missing index, and raises.
    """
    from smartann.config import config_hash, load_yaml, resolve_paths

    cfg_dir = config_dir or Path(__file__).resolve().parent.parent.parent / "configs" / "index"
    if not cfg_dir.is_dir():
        raise FileNotFoundError(
            f"{cfg_dir} does not exist. Without it every caller would report "
            "'no index built', which is a misreport rather than a result."
        )
    cfg_path = cfg_dir / f"{dataset}_{variant}.yaml"
    if not cfg_path.exists():
        return None
    cfg = load_yaml(cfg_path, IndexBuildConfig)
    root = resolve_paths().data_root / "index" / dataset / config_hash(cfg)
    return IndexPaths(root) if (root / "meta.json").exists() else None

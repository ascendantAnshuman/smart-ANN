"""Vector-file readers: TEXMEX ``.fvecs``/``.ivecs`` and ANN-Benchmarks HDF5.

Everything returns ``float32`` and prefers memory-mapped views. SIFT1M's base
set is 512 MB; loading it twice in one process must not cost 1 GB of RSS, so the
processed loaders memory-map by default and callers copy only when they need to
mutate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from smartann.data.registry import DatasetSpec

__all__ = [
    "AnnDataset",
    "DISTANCE_CONVENTIONS",
    "DuplicateReport",
    "GroundTruth",
    "ZeroNormError",
    "ZeroNormReport",
    "drop_zero_norm_rows",
    "duplicate_report",
    "l2_normalize",
    "load_ground_truth",
    "load_processed",
    "read_ann_benchmarks_hdf5",
    "read_fvecs",
    "read_ivecs",
    "remap_neighbors",
]


class ZeroNormError(ValueError):
    """A vector set contains zero-norm rows, which cannot be normalised."""


@dataclass(frozen=True)
class AnnDataset:
    """One ANN-Benchmarks HDF5 file."""

    train: np.ndarray  # (n_base, dim) float32
    test: np.ndarray  # (n_queries, dim) float32
    neighbors: np.ndarray  # (n_queries, 100) int32 -- exact top-100
    distances: np.ndarray  # (n_queries, 100) float32
    distance_attr: str  # "euclidean" | "angular"


# ---------------------------------------------------------------------------
# TEXMEX
# ---------------------------------------------------------------------------
def _read_xvecs(path: str | Path, dtype: type[np.generic]) -> np.ndarray:
    """Shared TEXMEX reader.

    Layout: each record is a little-endian int32 dimension followed by that many
    values. The dimension is repeated on every record, and this checks that they
    all agree -- a truncated or concatenated file otherwise reads as a plausible
    array of the wrong shape.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    as_int = np.memmap(p, dtype=np.int32, mode="r")
    if as_int.size == 0:
        raise ValueError(f"{p} is empty")

    dim = int(as_int[0])
    if dim <= 0:
        raise ValueError(f"{p}: first record declares a non-positive dimension {dim}")
    stride = dim + 1
    if as_int.size % stride:
        raise ValueError(
            f"{p}: size {as_int.size} int32s is not a multiple of the record stride "
            f"{stride} (dim={dim}); the file is truncated or not TEXMEX format"
        )

    n = as_int.size // stride
    dims = as_int.reshape(n, stride)[:, 0]
    bad = int(np.count_nonzero(dims != dim))
    if bad:
        first = int(np.argmax(dims != dim))
        raise ValueError(
            f"{p}: {bad} record(s) declare a dimension other than {dim}; "
            f"first at index {first} declares {int(dims[first])}"
        )

    # Two views of the same buffer: dimensions read as int32, payload as `dtype`.
    payload = np.memmap(p, dtype=dtype, mode="r").reshape(n, stride)[:, 1:]
    return payload


def read_fvecs(path: str | Path) -> np.ndarray:
    """Read a TEXMEX ``.fvecs`` file as a memory-mapped ``(n, dim)`` float32 view."""
    return _read_xvecs(path, np.float32)


def read_ivecs(path: str | Path) -> np.ndarray:
    """Read a TEXMEX ``.ivecs`` file as a memory-mapped ``(n, dim)`` int32 view."""
    return _read_xvecs(path, np.int32)


# ---------------------------------------------------------------------------
# ANN-Benchmarks HDF5
# ---------------------------------------------------------------------------
def read_ann_benchmarks_hdf5(path: str | Path, spec: DatasetSpec) -> AnnDataset:
    """Read an ANN-Benchmarks HDF5 file, asserting its metric against *spec*.

    The ``distance`` attribute is checked against ``spec.space`` before anything
    else. A metric mismatch is the single most expensive silent bug available
    here, so it fails immediately rather than producing quietly wrong recall.
    """
    import h5py

    p = Path(path)
    with h5py.File(p, "r") as f:
        missing = [k for k in ("train", "test", "neighbors", "distances") if k not in f]
        if missing:
            raise ValueError(f"{p}: missing dataset(s) {missing}; present: {list(f.keys())}")

        raw_attr = f.attrs.get("distance")
        if raw_attr is None:
            raise ValueError(f"{p}: no 'distance' attribute; cannot verify the metric")
        distance_attr = raw_attr.decode() if isinstance(raw_attr, bytes) else str(raw_attr)
        spec.assert_space_matches(distance_attr)

        train = np.asarray(f["train"], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)
        neighbors = np.asarray(f["neighbors"], dtype=np.int32)
        distances = np.asarray(f["distances"], dtype=np.float32)

    if train.shape[1] != spec.dim:
        raise ValueError(f"{p}: train dim {train.shape[1]} != registry dim {spec.dim}")
    if test.shape[1] != spec.dim:
        raise ValueError(f"{p}: test dim {test.shape[1]} != registry dim {spec.dim}")
    if spec.expected_n_base is not None and train.shape[0] != spec.expected_n_base:
        raise ValueError(
            f"{p}: {train.shape[0]} base vectors, registry expects {spec.expected_n_base}"
        )
    if test.shape[0] != spec.expected_n_queries:
        raise ValueError(
            f"{p}: {test.shape[0]} query vectors, registry expects {spec.expected_n_queries}"
        )

    return AnnDataset(
        train=train,
        test=test,
        neighbors=neighbors,
        distances=distances,
        distance_attr=distance_attr,
    )


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ZeroNormReport:
    """What ``drop_zero_norm_rows`` removed, so preparation can record it."""

    kept: np.ndarray  # (n_kept, dim) the surviving vectors
    keep_mask: np.ndarray  # (n,) bool
    dropped_indices: np.ndarray  # (n_dropped,) int64, into the ORIGINAL array
    old_to_new: np.ndarray  # (n,) int64; -1 where dropped

    @property
    def n_dropped(self) -> int:
        return int(self.dropped_indices.size)


def drop_zero_norm_rows(x: np.ndarray) -> ZeroNormReport:
    """Remove zero-norm rows and return the mapping needed to remap references.

    A zero vector has no direction, so under cosine it is similar to nothing and
    its distance to everything is undefined. Worse, under an inner-product
    search it scores exactly 0.0, which ranks it *above* every genuinely
    anti-correlated vector -- so leaving it in the index does not merely waste a
    slot, it displaces correct results.

    NYTimes ships 239 such rows in its base set and 9 in its query set. Dropping
    them is an explicit, recorded decision made here; :func:`l2_normalize`
    deliberately still refuses to normalise them, so nothing can remove them by
    accident.
    """
    norms = np.linalg.norm(np.asarray(x, dtype=np.float32), axis=1)
    keep_mask = norms > 0.0
    dropped = np.flatnonzero(~keep_mask).astype(np.int64)

    old_to_new = np.full(x.shape[0], -1, dtype=np.int64)
    old_to_new[keep_mask] = np.arange(int(keep_mask.sum()), dtype=np.int64)

    return ZeroNormReport(
        kept=np.ascontiguousarray(x[keep_mask], dtype=np.float32),
        keep_mask=keep_mask,
        dropped_indices=dropped,
        old_to_new=old_to_new,
    )


@dataclass(frozen=True)
class DuplicateReport:
    """How many rows of a matrix are byte-identical repeats of another row."""

    n_rows: int
    n_distinct: int
    n_repeat_rows: int  # rows that are not the first member of their group
    n_groups: int  # groups with more than one member
    largest_group: int

    @property
    def repeat_fraction(self) -> float:
        return self.n_repeat_rows / self.n_rows if self.n_rows else 0.0


def duplicate_report(x: np.ndarray) -> DuplicateReport:
    """Exact duplicate detection by raw bytes, not by distance.

    Distance-based near-duplicate detection would beg the question this exists
    to answer. ``1 - cos`` suffers catastrophic cancellation as ``cos -> 1``:
    in float32, eps near 1.0 is 1.19e-07, so any two vectors whose cosine is
    within that of 1.0 collapse to the same stored distance whether or not they
    are the same vector. Comparing bytes is immune to that entirely -- it either
    is the same vector or it is not.
    """
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {x.shape}")
    rows = np.ascontiguousarray(x)
    packed = rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).ravel()
    _, counts = np.unique(packed, return_counts=True)
    return DuplicateReport(
        n_rows=int(rows.shape[0]),
        n_distinct=int(counts.shape[0]),
        n_repeat_rows=int(rows.shape[0] - counts.shape[0]),
        n_groups=int(np.sum(counts > 1)),
        largest_group=int(counts.max()) if counts.size else 0,
    )


def remap_neighbors(neighbors: np.ndarray, old_to_new: np.ndarray) -> np.ndarray:
    """Rewrite neighbour indices into a compacted base set.

    Raises if any neighbour referred to a dropped row: that would make the
    ground truth ragged, and silently truncating it would overstate recall.
    """
    remapped = old_to_new[neighbors]
    if np.any(remapped < 0):
        n_bad = int(np.count_nonzero(remapped < 0))
        raise ValueError(
            f"ground truth references {n_bad} dropped row(s); compacting the base set "
            "would make the neighbour lists ragged. Refusing to truncate them."
        )
    out: np.ndarray = remapped.astype(np.int32)
    return out


def l2_normalize(x: np.ndarray, name: str = "vectors") -> np.ndarray:
    """L2-normalise rows, refusing to invent a direction for zero vectors.

    Zero-norm rows cannot be normalised: dividing gives NaN, and substituting an
    arbitrary unit vector would fabricate data. They are reported instead.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1)
    zeros = int(np.count_nonzero(norms == 0.0))
    if zeros:
        idx = np.flatnonzero(norms == 0.0)[:5].tolist()
        raise ZeroNormError(
            f"{name}: {zeros} zero-norm row(s) cannot be L2-normalised "
            f"(first indices: {idx}). Refusing to fabricate a direction."
        )
    normalized: np.ndarray = (x / norms[:, None]).astype(np.float32)
    return normalized


# ---------------------------------------------------------------------------
# Processed artifacts
# ---------------------------------------------------------------------------
def load_processed(processed_dir: str | Path, split: str, mmap: bool = True) -> np.ndarray:
    """Load ``base.npy`` / ``queries.npy`` / ``groundtruth_top100.npy``.

    Memory-mapped by default: SIFT1M's base array is 512 MB and is opened by
    several call sites in one process.
    """
    filename = {
        "base": "base.npy",
        "queries": "queries.npy",
        "groundtruth": "groundtruth_top100.npy",
        "groundtruth_dists": "groundtruth_dists_top100.npy",
    }.get(split)
    if filename is None:
        raise KeyError(f"unknown split {split!r}; expected base, queries or groundtruth")

    path = Path(processed_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not prepared yet")
    arr: np.ndarray = np.load(path, mmap_mode="r" if mmap else None)
    return arr


@dataclass(frozen=True)
class GroundTruth:
    """Neighbour ids and their distances, with the units they are stored in."""

    ids: np.ndarray  # (n_queries, depth) int32
    dists: np.ndarray  # (n_queries, depth) float64, smaller is better
    convention: str
    depth: int


# The distance conventions smartann.bench.metrics can interpret. Both are
# "smaller is better"; the metrics module has no space parameter precisely
# because that is uniform (ADR-0018), so this is where the assumption is
# checked rather than assumed.
DISTANCE_CONVENTIONS = frozenset({"squared_l2", "one_minus_ip"})


def load_ground_truth(
    processed_dir: str | Path, data_root: str | Path, mmap: bool = True
) -> GroundTruth:
    """Load ground truth and *verify* the convention its distances are in.

    The metrics take no ``space`` argument, which is only safe while every
    dataset stores distances the same way. That holds today and is asserted
    here rather than assumed, because the stretch datasets (HotpotQA, TriviaQA)
    have not been prepared and a future reader that stored raw similarities
    would invert every ranking with no symptom.
    """
    from smartann.data.manifest import DataManifest

    processed_dir = Path(processed_dir)
    ids = load_processed(processed_dir, "groundtruth", mmap=mmap)
    dists = load_processed(processed_dir, "groundtruth_dists", mmap=mmap)
    if ids.shape != dists.shape:
        raise ValueError(f"ground truth ids {ids.shape} and distances {dists.shape} disagree")

    entry = DataManifest(data_root).get(processed_dir / "groundtruth_dists_top100.npy")
    if entry is None:
        raise ValueError(
            f"{processed_dir / 'groundtruth_dists_top100.npy'} has no manifest entry, so its "
            "distance convention is unknown. Run `smartann data gt-dists <dataset>`."
        )
    convention = entry.get("extra", {}).get("dists_convention")
    if convention not in DISTANCE_CONVENTIONS:
        raise ValueError(
            f"ground truth is recorded as {convention!r}, which is not one of "
            f"{sorted(DISTANCE_CONVENTIONS)}. smartann.bench.metrics assumes distances where "
            "smaller is better; a similarity convention would invert every ranking silently."
        )

    return GroundTruth(ids=ids, dists=dists, convention=str(convention), depth=int(ids.shape[1]))

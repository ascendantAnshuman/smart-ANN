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
    "ZeroNormError",
    "l2_normalize",
    "load_processed",
    "read_ann_benchmarks_hdf5",
    "read_fvecs",
    "read_ivecs",
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
def load_processed(
    processed_dir: str | Path, split: str, mmap: bool = True
) -> np.ndarray:
    """Load ``base.npy`` / ``queries.npy`` / ``groundtruth_top100.npy``.

    Memory-mapped by default: SIFT1M's base array is 512 MB and is opened by
    several call sites in one process.
    """
    filename = {
        "base": "base.npy",
        "queries": "queries.npy",
        "groundtruth": "groundtruth_top100.npy",
    }.get(split)
    if filename is None:
        raise KeyError(f"unknown split {split!r}; expected base, queries or groundtruth")

    path = Path(processed_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not prepared yet")
    arr: np.ndarray = np.load(path, mmap_mode="r" if mmap else None)
    return arr

"""Phase 1: TEXMEX readers, normalisation, and the metric assertion."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from smartann.data.readers import (
    ZeroNormError,
    l2_normalize,
    load_processed,
    read_ann_benchmarks_hdf5,
    read_fvecs,
    read_ivecs,
)
from smartann.data.registry import spec_for


def write_fvecs(path: Path, arr: np.ndarray) -> None:
    """Write TEXMEX .fvecs: per record, int32 dim then dim float32."""
    with path.open("wb") as fh:
        for row in arr:
            fh.write(struct.pack("<i", len(row)))
            fh.write(np.asarray(row, dtype="<f4").tobytes())


def write_ivecs(path: Path, arr: np.ndarray) -> None:
    with path.open("wb") as fh:
        for row in arr:
            fh.write(struct.pack("<i", len(row)))
            fh.write(np.asarray(row, dtype="<i4").tobytes())


# ---------------------------------------------------------------------------
# TEXMEX
# ---------------------------------------------------------------------------
def test_fvecs_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((37, 12)).astype(np.float32)
    p = tmp_path / "a.fvecs"
    write_fvecs(p, arr)
    np.testing.assert_array_equal(np.asarray(read_fvecs(p)), arr)


def test_ivecs_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 1000, size=(19, 8)).astype(np.int32)
    p = tmp_path / "a.ivecs"
    write_ivecs(p, arr)
    np.testing.assert_array_equal(np.asarray(read_ivecs(p)), arr)


def test_fvecs_rejects_ragged_dimensions(tmp_path: Path) -> None:
    """A record whose declared dimension differs must not read as valid.

    The size is left a clean multiple of the stride so that this exercises the
    per-record dimension check specifically, rather than the stride check.
    """
    p = tmp_path / "ragged.fvecs"
    write_fvecs(p, np.ones((3, 4), dtype=np.float32))

    raw = bytearray(p.read_bytes())
    raw[5 * 4 : 6 * 4] = struct.pack("<i", 9)  # corrupt record 1's dimension field
    p.write_bytes(bytes(raw))

    with pytest.raises(ValueError, match="declare a dimension other than 4"):
        read_fvecs(p)


def test_fvecs_rejects_truncated_file(tmp_path: Path) -> None:
    p = tmp_path / "trunc.fvecs"
    raw = tmp_path / "full.fvecs"
    write_fvecs(raw, np.ones((4, 6), dtype=np.float32))
    p.write_bytes(raw.read_bytes()[:-9])  # lop off part of the last record
    with pytest.raises(ValueError, match=r"truncated|multiple"):
        read_fvecs(p)


def test_fvecs_rejects_empty(tmp_path: Path) -> None:
    p = tmp_path / "empty.fvecs"
    p.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        read_fvecs(p)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def test_l2_normalize_makes_unit_rows() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal((50, 16)).astype(np.float32) * 7.0
    out = l2_normalize(x)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)


def test_l2_normalize_preserves_cosine_ordering() -> None:
    """The reason normalising is safe: it cannot reorder cosine neighbours."""
    rng = np.random.default_rng(3)
    base = rng.standard_normal((200, 16)).astype(np.float32)
    q = rng.standard_normal(16).astype(np.float32)

    raw_cos = (base @ q) / (np.linalg.norm(base, axis=1) * np.linalg.norm(q))
    normed = l2_normalize(base)
    ip = normed @ (q / np.linalg.norm(q))
    np.testing.assert_array_equal(np.argsort(-raw_cos), np.argsort(-ip))


def test_l2_normalize_refuses_zero_rows() -> None:
    x = np.ones((5, 4), dtype=np.float32)
    x[2] = 0.0
    with pytest.raises(ZeroNormError, match="zero-norm"):
        l2_normalize(x, "test-vectors")


# ---------------------------------------------------------------------------
# HDF5 metric assertion
# ---------------------------------------------------------------------------
def _write_hdf5(path: Path, distance: str, dim: int, n_base: int, n_q: int) -> None:
    import h5py

    rng = np.random.default_rng(4)
    with h5py.File(path, "w") as f:
        f.attrs["distance"] = distance
        f.create_dataset("train", data=rng.standard_normal((n_base, dim)).astype(np.float32))
        f.create_dataset("test", data=rng.standard_normal((n_q, dim)).astype(np.float32))
        f.create_dataset("neighbors", data=rng.integers(0, n_base, (n_q, 100)).astype(np.int32))
        f.create_dataset("distances", data=rng.random((n_q, 100)).astype(np.float32))


def test_hdf5_metric_mismatch_is_fatal(tmp_path: Path) -> None:
    """sift1m is registered as l2; a file claiming angular must not load."""
    spec = spec_for("sift1m")
    p = tmp_path / "wrong.hdf5"
    _write_hdf5(p, "angular", spec.dim, 10, 5)
    with pytest.raises(ValueError, match="metric mismatch"):
        read_ann_benchmarks_hdf5(p, spec)


def test_hdf5_unknown_distance_is_fatal(tmp_path: Path) -> None:
    spec = spec_for("sift1m")
    p = tmp_path / "weird.hdf5"
    _write_hdf5(p, "manhattan", spec.dim, 10, 5)
    with pytest.raises(ValueError, match="unrecognised"):
        read_ann_benchmarks_hdf5(p, spec)


def test_hdf5_wrong_count_is_fatal(tmp_path: Path) -> None:
    """The registry's expected counts are asserted, not assumed."""
    spec = spec_for("sift1m")
    p = tmp_path / "short.hdf5"
    _write_hdf5(p, "euclidean", spec.dim, 10, 5)
    with pytest.raises(ValueError, match=r"base vectors|query vectors"):
        read_ann_benchmarks_hdf5(p, spec)


# ---------------------------------------------------------------------------
# Processed loading
# ---------------------------------------------------------------------------
def test_load_processed_is_memmapped(tmp_path: Path) -> None:
    arr = np.arange(1000, dtype=np.float32).reshape(100, 10)
    np.save(tmp_path / "base.npy", arr)
    loaded = load_processed(tmp_path, "base")
    assert isinstance(loaded, np.memmap)
    np.testing.assert_array_equal(np.asarray(loaded), arr)


def test_load_processed_rejects_unknown_split(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="unknown split"):
        load_processed(tmp_path, "nonsense")


def test_load_processed_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_processed(tmp_path, "base")

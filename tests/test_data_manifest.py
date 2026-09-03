"""Phase 1: MANIFEST.json provenance (acceptance check 7)."""

from __future__ import annotations

import json
from pathlib import Path

from smartann.data.manifest import DataManifest


def _touch(path: Path, content: bytes = b"x" * 10) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_record_and_write(tmp_path: Path) -> None:
    f = _touch(tmp_path / "raw" / "sift1m" / "a.hdf5")
    m = DataManifest(tmp_path)
    rec = m.record(f, dataset="sift1m", stage="raw", sha256="a" * 64, source_url="https://x/a")

    assert rec.path == "raw/sift1m/a.hdf5"
    assert rec.n_bytes == 10
    out = json.loads(m.write().read_text(encoding="utf-8"))
    assert out["artifacts"]["raw/sift1m/a.hdf5"]["source_url"] == "https://x/a"


def test_manifest_survives_a_reload(tmp_path: Path) -> None:
    f = _touch(tmp_path / "raw" / "a.bin")
    m = DataManifest(tmp_path)
    m.record(f, dataset="d", stage="raw", sha256="b" * 64)
    m.write()

    reloaded = DataManifest(tmp_path)
    assert reloaded.get(f) is not None
    assert reloaded.total_bytes() == 10


def test_missing_files_detects_a_stale_entry(tmp_path: Path) -> None:
    """A manifest that describes files which are gone is a lie, and says so."""
    f = _touch(tmp_path / "raw" / "gone.bin")
    m = DataManifest(tmp_path)
    m.record(f, dataset="d", stage="raw", sha256="c" * 64)
    f.unlink()
    assert m.missing_files() == ["raw/gone.bin"]


def test_unrecorded_files_detects_untracked_artifacts(tmp_path: Path) -> None:
    _touch(tmp_path / "processed" / "orphan.npy")
    m = DataManifest(tmp_path)
    assert m.unrecorded_files() == ["processed/orphan.npy"]


def test_working_files_are_not_treated_as_artifacts(tmp_path: Path) -> None:
    """Partial downloads, checkpoints and the embedding cache are not results."""
    _touch(tmp_path / "raw" / "big.hdf5.part")
    _touch(tmp_path / "processed" / "squad" / "groundtruth.ckpt.npz")
    _touch(tmp_path / "cache" / "embeddings.sqlite")
    assert DataManifest(tmp_path).unrecorded_files() == []


def test_per_dataset_accounting(tmp_path: Path) -> None:
    m = DataManifest(tmp_path)
    m.record(_touch(tmp_path / "a.bin", b"1" * 100), dataset="x", stage="raw", sha256="d" * 64)
    m.record(_touch(tmp_path / "b.bin", b"1" * 50), dataset="y", stage="raw", sha256="e" * 64)

    assert m.total_bytes("x") == 100
    assert m.total_bytes("y") == 50
    assert m.total_bytes() == 150
    assert len(m.for_dataset("x")) == 1

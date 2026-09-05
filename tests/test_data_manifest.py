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


def test_concurrent_writers_do_not_clobber_each_other(tmp_path: Path) -> None:
    """The NYTimes/SQuAD lost-update race, reproduced.

    Two preparations running in parallel each build a manifest, add their own
    artifacts, and write at the end. Without merge-on-write the later writer
    erases the earlier one's entries -- which is exactly what removed NYTimes'
    four artifacts from the real manifest.
    """
    a_file = _touch(tmp_path / "raw" / "a.bin")
    b_file = _touch(tmp_path / "raw" / "b.bin")

    first = DataManifest(tmp_path)
    second = DataManifest(tmp_path)  # constructed before either has written

    first.record(a_file, dataset="alpha", stage="raw", sha256="a" * 64)
    second.record(b_file, dataset="beta", stage="raw", sha256="b" * 64)

    first.write()
    second.write()  # must not erase alpha

    final = DataManifest(tmp_path)
    assert final.get(a_file) is not None, "the earlier writer's entry was clobbered"
    assert final.get(b_file) is not None
    assert final.unrecorded_files() == []


def test_write_recovers_from_a_corrupt_manifest(tmp_path: Path) -> None:
    (tmp_path / DataManifest.FILENAME).write_text("{not json", encoding="utf-8")
    f = _touch(tmp_path / "raw" / "a.bin")

    m = DataManifest(tmp_path)
    m.record(f, dataset="x", stage="raw", sha256="c" * 64)
    m.write()

    assert DataManifest(tmp_path).get(f) is not None


def test_write_leaves_no_temp_file_behind(tmp_path: Path) -> None:
    f = _touch(tmp_path / "raw" / "a.bin")
    m = DataManifest(tmp_path)
    m.record(f, dataset="x", stage="raw", sha256="d" * 64)
    m.write()

    assert not (tmp_path / (DataManifest.FILENAME + ".tmp")).exists()
    assert m.unrecorded_files() == []


class TestIndexRootsAreSelfDescribing:
    """Index roots carry their own provenance, so they are excluded conditionally.

    Discovered when ``data status`` -- run for the first time through a working
    entry point (ADR-0042) -- reported 13,154 unrecorded files, all of them
    inside built index directories. Indexes are build outputs reproducible from
    ``(config.yaml, seed)``, and each root already stores a ``meta.json``. The
    exclusion is conditional on that file existing, so a half-written root is
    still reported.
    """

    def test_described_index_root_is_not_reported(self, tmp_path: Path) -> None:
        root = tmp_path / "index" / "sift1m" / "abc123"
        (root / "clusters").mkdir(parents=True)
        (root / "meta.json").write_text("{}", encoding="utf-8")
        (root / "centroids.npy").write_bytes(b"x")
        (root / "clusters" / "00000.npy").write_bytes(b"y")

        assert DataManifest(tmp_path).unrecorded_files() == []

    def test_index_root_without_meta_is_still_reported(self, tmp_path: Path) -> None:
        """A root with no ``meta.json`` is a half-written build, not an artifact."""
        root = tmp_path / "index" / "sift1m" / "abc123"
        root.mkdir(parents=True)
        (root / "centroids.npy").write_bytes(b"x")

        assert DataManifest(tmp_path).unrecorded_files() == ["index/sift1m/abc123/centroids.npy"]

    def test_stray_file_outside_index_still_reported(self, tmp_path: Path) -> None:
        """The exclusion must not widen beyond ``index/``."""
        described = tmp_path / "index" / "sift1m" / "abc123"
        described.mkdir(parents=True)
        (described / "meta.json").write_text("{}", encoding="utf-8")
        (tmp_path / "processed").mkdir()
        (tmp_path / "processed" / "stray.npy").write_bytes(b"z")

        assert DataManifest(tmp_path).unrecorded_files() == ["processed/stray.npy"]

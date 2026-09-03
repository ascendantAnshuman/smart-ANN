"""Phase 1: the preparation pipeline end to end, against synthetic HDF5 files.

Uses tiny synthetic datasets rather than the real ones so the whole suite stays
fast and offline. The real-data checks live in scripts/phase1_acceptance.py.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from smartann.config import resolve_paths
from smartann.data.manifest import DataManifest
from smartann.data.prepare import _process_ann, status_for
from smartann.data.registry import DatasetArtifact, DatasetSpec


def _spec(normalized: bool, n_base: int = 60, n_q: int = 10, dim: int = 8) -> DatasetSpec:
    return DatasetSpec(
        name="synth",
        kind="ann",
        dim=dim,
        space="cosine" if normalized else "l2",
        normalized=normalized,
        expected_n_base=n_base,
        expected_n_queries=n_q,
        artifacts=(DatasetArtifact(key="hdf5", filename="s.hdf5", urls=("https://x/s.hdf5",)),),
        ships_groundtruth=True,
        default_nc="auto",
        notes="synthetic fixture",
    )


def _write_hdf5(
    path: Path,
    spec: DatasetSpec,
    zero_base: list[int] | None = None,
    zero_queries: list[int] | None = None,
    seed: int = 0,
    gt_depth: int = 10,
) -> np.ndarray:
    """Write a synthetic ANN-Benchmarks file. Returns the neighbours written.

    The real files carry top-100; these fixtures carry top-``gt_depth`` so the
    base set can stay tiny and the suite fast. Nothing in the remapping logic
    depends on the depth.
    """
    import h5py

    rng = np.random.default_rng(seed)
    n_base = spec.expected_n_base or 0
    train = rng.standard_normal((n_base, spec.dim)).astype(np.float32)
    test = rng.standard_normal((spec.expected_n_queries, spec.dim)).astype(np.float32)
    if zero_base:
        train[zero_base] = 0.0
    if zero_queries:
        test[zero_queries] = 0.0

    # Neighbours that deliberately avoid the zero rows, as the real files do.
    allowed = [i for i in range(n_base) if i not in set(zero_base or [])]
    neighbors = np.array(
        [rng.choice(allowed, size=gt_depth, replace=False) for _ in range(spec.expected_n_queries)],
        dtype=np.int32,
    )

    with h5py.File(path, "w") as f:
        f.attrs["distance"] = "angular" if spec.normalized else "euclidean"
        f.create_dataset("train", data=train)
        f.create_dataset("test", data=test)
        f.create_dataset("neighbors", data=neighbors)
        f.create_dataset(
            "distances", data=rng.random((spec.expected_n_queries, gt_depth)).astype(np.float32)
        )
    return neighbors


def test_process_ann_writes_all_three_artifacts(tmp_path: Path) -> None:
    spec = _spec(normalized=False)
    raw = tmp_path / "s.hdf5"
    _write_hdf5(raw, spec)
    out = tmp_path / "processed"
    manifest = DataManifest(tmp_path)

    result = _process_ann(spec, {"hdf5": raw}, out, manifest, quiet=True)

    assert (out / "base.npy").exists()
    assert (out / "queries.npy").exists()
    assert (out / "groundtruth_top100.npy").exists()
    assert result.n_base == 60
    assert result.n_queries == 10
    assert result.search_space == "l2"


def test_unnormalised_dataset_is_left_alone(tmp_path: Path) -> None:
    spec = _spec(normalized=False)
    raw = tmp_path / "s.hdf5"
    _write_hdf5(raw, spec)
    out = tmp_path / "processed"

    result = _process_ann(spec, {"hdf5": raw}, out, DataManifest(tmp_path), quiet=True)
    base = np.load(out / "base.npy")

    assert result.details["dropped_zero_norm_base"] == 0
    assert not np.allclose(np.linalg.norm(base, axis=1), 1.0), "l2 data must not be normalised"


def test_normalised_dataset_gets_unit_rows(tmp_path: Path) -> None:
    spec = _spec(normalized=True)
    raw = tmp_path / "s.hdf5"
    _write_hdf5(raw, spec)
    out = tmp_path / "processed"

    result = _process_ann(spec, {"hdf5": raw}, out, DataManifest(tmp_path), quiet=True)
    base = np.load(out / "base.npy")

    np.testing.assert_allclose(np.linalg.norm(base, axis=1), 1.0, atol=1e-6)
    assert result.search_space == "ip"


def test_zero_norm_rows_are_dropped_and_reported(tmp_path: Path) -> None:
    """The NYTimes case, in miniature."""
    spec = _spec(normalized=True)
    raw = tmp_path / "s.hdf5"
    _write_hdf5(raw, spec, zero_base=[3, 17, 41], zero_queries=[2, 5])
    out = tmp_path / "processed"

    result = _process_ann(spec, {"hdf5": raw}, out, DataManifest(tmp_path), quiet=True)

    assert result.details["dropped_zero_norm_base"] == 3
    assert result.details["dropped_zero_norm_queries"] == 2
    assert result.details["file_n_base"] == 60
    assert result.details["file_n_queries"] == 10
    assert result.n_base == 57
    assert result.n_queries == 8


def test_ground_truth_is_remapped_and_stays_in_range(tmp_path: Path) -> None:
    spec = _spec(normalized=True)
    raw = tmp_path / "s.hdf5"
    _write_hdf5(raw, spec, zero_base=[3, 17, 41], zero_queries=[2, 5])
    out = tmp_path / "processed"

    _process_ann(spec, {"hdf5": raw}, out, DataManifest(tmp_path), quiet=True)
    gt = np.load(out / "groundtruth_top100.npy")
    base = np.load(out / "base.npy")

    assert gt.shape[0] == 8, "ground-truth rows for dropped queries must go too"
    assert int(gt.min()) >= 0
    assert int(gt.max()) < base.shape[0], "remapped indices must address the compacted base"


def test_remapped_ground_truth_points_at_the_same_vectors(tmp_path: Path) -> None:
    """The claim ADR-0014 rests on: identity preserved, only indices move."""
    import h5py

    spec = _spec(normalized=True)
    raw = tmp_path / "s.hdf5"
    _write_hdf5(raw, spec, zero_base=[3, 17, 41], zero_queries=[2, 5])
    out = tmp_path / "processed"
    _process_ann(spec, {"hdf5": raw}, out, DataManifest(tmp_path), quiet=True)

    with h5py.File(raw, "r") as f:
        orig_train = np.asarray(f["train"], dtype=np.float32)
        orig_nbrs = np.asarray(f["neighbors"], dtype=np.int32)
    kept_q = [i for i in range(10) if i not in (2, 5)]

    base = np.load(out / "base.npy")
    gt = np.load(out / "groundtruth_top100.npy")

    for new_q, old_q in enumerate(kept_q):
        for new_i, old_i in zip(gt[new_q, :5], orig_nbrs[old_q, :5], strict=True):
            want = orig_train[old_i] / np.linalg.norm(orig_train[old_i])
            np.testing.assert_allclose(base[new_i], want, atol=1e-6)


def test_ground_truth_referencing_a_dropped_row_is_fatal(tmp_path: Path) -> None:
    """Silently truncating a neighbour list would overstate recall."""
    import h5py

    spec = _spec(normalized=True)
    raw = tmp_path / "s.hdf5"
    _write_hdf5(raw, spec, zero_base=[3])
    with h5py.File(raw, "r+") as f:
        f["neighbors"][0, 0] = 3  # point at the dropped row

    with pytest.raises(ValueError, match="dropped row"):
        _process_ann(spec, {"hdf5": raw}, tmp_path / "p", DataManifest(tmp_path), quiet=True)


def test_resolved_nc_uses_the_post_drop_count(tmp_path: Path) -> None:
    """Nc must be derived from the vectors actually indexed."""
    spec = _spec(normalized=True, n_base=100)
    raw = tmp_path / "s.hdf5"
    _write_hdf5(raw, spec, zero_base=list(range(19)))
    out = tmp_path / "processed"

    result = _process_ann(spec, {"hdf5": raw}, out, DataManifest(tmp_path), quiet=True)
    assert result.n_base == 81
    assert result.resolved_nc == 9  # round(sqrt(81)), not round(sqrt(100))


def test_status_reports_unprepared_dataset() -> None:
    """conftest isolates the data root, so nothing is prepared here."""
    s = status_for("sift1m", resolve_paths())
    assert not s.downloaded
    assert not s.processed
    assert s.n_base is None
    assert s.dim == 128


def test_prepare_result_is_frozen() -> None:
    """A recorded preparation outcome must not be editable after the fact."""
    from smartann.data.prepare import PrepareResult

    assert dataclasses.fields(PrepareResult)
    assert PrepareResult.__dataclass_params__.frozen is True  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# completeness is an exit code, not a table to read
# ---------------------------------------------------------------------------
def test_status_of_an_unprepared_dataset_is_incomplete() -> None:
    s = status_for("sift1m", resolve_paths())
    assert not s.complete
    assert "downloaded" in s.missing()
    assert "groundtruth_dists_top100.npy" in s.missing()


def test_data_status_exits_non_zero_when_a_dataset_is_incomplete() -> None:
    """The Phase 1 miss, as a test: SQuAD had no ground truth and Phase 1 passed."""
    from typer.testing import CliRunner

    from smartann.cli import app

    result = CliRunner().invoke(app, ["data", "status"])
    assert result.exit_code == 1, "an empty data root must fail, not merely print red"
    assert "incomplete" in result.stdout


def test_duplicate_report_counts_repeats_not_near_neighbours() -> None:
    """Bytes, not distances -- distance-based detection begs the question."""
    from smartann.data.readers import duplicate_report

    rows = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    rep = duplicate_report(rows)
    assert rep.n_rows == 5
    assert rep.n_distinct == 3
    assert rep.n_repeat_rows == 2
    assert rep.n_groups == 1
    assert rep.largest_group == 3
    assert rep.repeat_fraction == pytest.approx(0.4)


def test_duplicate_report_does_not_merge_merely_close_rows() -> None:
    from smartann.data.readers import duplicate_report

    rows = np.array([[1.0, 2.0], [1.0, 2.0000002]], dtype=np.float32)
    assert duplicate_report(rows).n_repeat_rows == 0

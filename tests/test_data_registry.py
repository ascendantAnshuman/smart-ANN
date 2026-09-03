"""Phase 1: the registry is the single source of truth for dataset facts."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from smartann.data.registry import (
    HDF5_DISTANCE_TO_SPACE,
    load_registry,
    record_checksum,
    spec_for,
)

# The Phase 1 table, asserted rather than assumed.
EXPECTED = {
    "sift1m": {"kind": "ann", "dim": 128, "space": "l2", "n_base": 1_000_000, "n_q": 10_000},
    "nytimes": {"kind": "ann", "dim": 256, "space": "cosine", "n_base": 290_000, "n_q": 10_000},
    "squad": {"kind": "qa", "dim": 384, "space": "cosine", "n_base": None, "n_q": 1_000},
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_registry_matches_the_specified_table(name: str) -> None:
    spec = spec_for(name)
    want = EXPECTED[name]
    assert spec.kind == want["kind"]
    assert spec.dim == want["dim"]
    assert spec.space == want["space"]
    assert spec.expected_n_base == want["n_base"]
    assert spec.expected_n_queries == want["n_q"]


def test_search_space_is_derived_not_stored() -> None:
    """Normalised cosine datasets are searched as inner product (ADR-0010)."""
    assert spec_for("sift1m").search_space == "l2"  # not normalised
    assert spec_for("nytimes").search_space == "ip"  # cosine + normalised
    assert spec_for("squad").search_space == "ip"


def test_nc_pinned_for_sift_and_auto_elsewhere() -> None:
    sift = spec_for("sift1m")
    assert sift.default_nc == 2000
    assert sift.resolve_nc(1_000_000) == 2000, "a pinned Nc must ignore N"

    ny = spec_for("nytimes")
    assert ny.default_nc == "auto"
    assert ny.resolve_nc(290_000) == round(math.sqrt(290_000)) == 539


def test_resolve_nc_rejects_empty_corpus() -> None:
    with pytest.raises(ValueError, match="n_base"):
        spec_for("squad").resolve_nc(0)


@pytest.mark.parametrize(("attr", "space"), sorted(HDF5_DISTANCE_TO_SPACE.items()))
def test_distance_attr_mapping(attr: str, space: str) -> None:
    assert HDF5_DISTANCE_TO_SPACE[attr] == space


def test_assert_space_matches_accepts_correct_metric() -> None:
    spec_for("sift1m").assert_space_matches("euclidean")
    spec_for("nytimes").assert_space_matches("angular")


def test_assert_space_matches_rejects_wrong_metric() -> None:
    with pytest.raises(ValueError, match="metric mismatch"):
        spec_for("sift1m").assert_space_matches("angular")


def test_squad_has_two_artifacts() -> None:
    """Multi-file datasets must not need special handling (ADR-0013)."""
    spec = spec_for("squad")
    assert {a.key for a in spec.artifacts} == {"train", "dev"}
    assert spec.sha256 is None, "a multi-artifact dataset has no single checksum"


def test_artifact_lookup_error_lists_known_keys() -> None:
    with pytest.raises(KeyError, match=r"train, dev|dev, train"):
        spec_for("squad").artifact("nope")


def test_all_urls_are_https_or_ftp() -> None:
    for spec in load_registry().values():
        for url in spec.source_urls:
            assert url.startswith(("https://", "ftp://")), f"{spec.name}: insecure URL {url}"


# ---------------------------------------------------------------------------
# Checksum recording
# ---------------------------------------------------------------------------
def test_record_checksum_writes_and_overlays(tmp_path: Path) -> None:
    cfg = tmp_path / "datasets.yaml"
    record_checksum("sift1m", "hdf5", "a" * 64, 123, "https://example/x", checksum_path=cfg)

    written = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert written["sift1m"]["hdf5"]["sha256"] == "a" * 64
    assert written["sift1m"]["hdf5"]["n_bytes"] == 123

    overlaid = load_registry(checksum_path=cfg)["sift1m"]
    assert overlaid.artifact("hdf5").sha256 == "a" * 64
    assert overlaid.sha256 == "a" * 64  # single-artifact convenience


def test_record_checksum_refuses_to_overwrite_a_different_value(tmp_path: Path) -> None:
    """A changed checksum is evidence, not a thing to silently replace."""
    cfg = tmp_path / "datasets.yaml"
    record_checksum("sift1m", "hdf5", "a" * 64, 1, "https://example/x", checksum_path=cfg)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        record_checksum("sift1m", "hdf5", "b" * 64, 1, "https://example/x", checksum_path=cfg)


def test_record_checksum_is_idempotent(tmp_path: Path) -> None:
    cfg = tmp_path / "datasets.yaml"
    record_checksum("sift1m", "hdf5", "a" * 64, 1, "https://example/x", checksum_path=cfg)
    record_checksum("sift1m", "hdf5", "a" * 64, 1, "https://example/x", checksum_path=cfg)
    assert load_registry(checksum_path=cfg)["sift1m"].sha256 == "a" * 64


def test_missing_checksum_file_yields_none(tmp_path: Path) -> None:
    reg = load_registry(checksum_path=tmp_path / "absent.yaml")
    assert all(a.sha256 is None for s in reg.values() for a in s.artifacts)

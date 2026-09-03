"""Phase 0 acceptance tests: the reproducibility record (Section 3, rule 4)."""

from __future__ import annotations

import json
from pathlib import Path

from smartann.bench.manifest import Manifest, git_state, library_versions
from smartann.config import config_hash
from smartann.types import AnnConfig, PipelineConfig, ScrConfig

REQUIRED_KEYS = {
    "experiment_id",
    "run_id",
    "seed",
    "config",
    "config_hash",
    "git",
    "hostname",
    "python_version",
    "platform",
    "libraries",
    "wall_clock_start",
    "wall_clock_end",
}


def _cfg() -> PipelineConfig:
    return PipelineConfig(
        ann=AnnConfig(nprobe=4, ef_centroid=64, ef_list=64, k=5, per_cluster_k=20),
        scr=ScrConfig(
            sliding_window_size=3,
            overlap_size=2,
            context_extension_size=1,
            top_windows=1,
            similarity_threshold=0.0,
        ),
    )


def test_manifest_contains_every_required_key(tmp_path: Path) -> None:
    m = Manifest.start(experiment_id="E0", run_id="r0", seed=7, config=_cfg()).finish()
    path = m.write(tmp_path)
    written = json.loads(path.read_text(encoding="utf-8"))
    missing = REQUIRED_KEYS - set(written)
    assert not missing, f"manifest.json is missing required keys: {sorted(missing)}"


def test_manifest_config_hash_matches_config(tmp_path: Path) -> None:
    cfg = _cfg()
    m = Manifest.start(experiment_id="E0", run_id="r0", seed=0, config=cfg).finish()
    assert m.config_hash == config_hash(cfg)


def test_manifest_records_duration(tmp_path: Path) -> None:
    m = Manifest.start(experiment_id="E0", run_id="r0", seed=0, config=_cfg())
    assert m.wall_clock_end is None
    m.finish()
    assert m.wall_clock_end is not None
    assert m.duration_s is not None
    assert m.duration_s >= 0.0


def test_manifest_is_json_round_trippable(tmp_path: Path) -> None:
    m = Manifest.start(experiment_id="E0", run_id="r0", seed=1, config=_cfg()).finish()
    path = m.write(tmp_path / "nested" / "dir")
    assert path.exists()
    reparsed = json.loads(path.read_text(encoding="utf-8"))
    assert reparsed["seed"] == 1
    assert reparsed["experiment_id"] == "E0"


def test_git_state_reports_availability() -> None:
    g = git_state()
    assert set(g) == {"sha", "branch", "dirty", "available"}
    if g["available"]:
        assert isinstance(g["sha"], str)
        assert len(g["sha"]) == 40


def test_library_versions_marks_absent_packages() -> None:
    v = library_versions(("numpy", "definitely-not-a-real-package"))
    assert v["definitely-not-a-real-package"] == "not-installed"

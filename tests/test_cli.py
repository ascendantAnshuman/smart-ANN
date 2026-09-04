"""Phase 0 acceptance tests: the CLI surface.

These lock in the acceptance check that ``smartann --help`` lists the
subcommands, and that unimplemented commands fail loudly rather than returning
a plausible value (Section 3, rule 2).
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from smartann import __version__
from smartann.cli import app

runner = CliRunner()

EXPECTED_SUBCOMMANDS = ["version", "paths", "data", "index", "config", "run", "figures"]


# Commands that must raise NotImplementedError until their phase lands.
#
# `data status`, `data prepare`, `data verify` and `data groundtruth` were on
# this list until Phase 1 implemented them. Leaving them here did real damage:
# `data prepare squad` stopped raising and instead performed a live 35 MB
# download and began writing a chunk database into the developer's data root.
# Anything removed from this list must gain real coverage below, and
# tests/conftest.py now isolates the data roots regardless.
def test_help_lists_every_subcommand() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in EXPECTED_SUBCOMMANDS:
        assert name in result.output, f"{name} missing from --help"


def test_version_matches_package() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == __version__


def test_paths_reports_both_roots() -> None:
    result = runner.invoke(app, ["paths"])
    assert result.exit_code == 0
    assert "data_root" in result.output
    assert "results_root" in result.output


def test_config_hash_is_stable_across_invocations(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "nprobe: 4\nef_centroid: 64\nef_list: 64\nk: 5\nper_cluster_k: 20\n",
        encoding="utf-8",
    )
    first = runner.invoke(app, ["config", "hash", str(cfg)])
    second = runner.invoke(app, ["config", "hash", str(cfg)])
    assert first.exit_code == 0
    assert first.output.strip() == second.output.strip()
    assert len(first.output.strip()) == 12


def test_config_hash_ignores_key_order(tmp_path: Path) -> None:
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("nprobe: 4\nk: 5\n", encoding="utf-8")
    b.write_text("k: 5\nnprobe: 4\n", encoding="utf-8")
    ra = runner.invoke(app, ["config", "hash", str(a)])
    rb = runner.invoke(app, ["config", "hash", str(b)])
    assert ra.output.strip() == rb.output.strip()


def test_config_show_json(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("nprobe: 4\nk: 5\n", encoding="utf-8")
    result = runner.invoke(app, ["config", "show", str(cfg), "--json"])
    assert result.exit_code == 0
    assert '"nprobe": 4' in result.output


def test_run_experiment_rejects_an_unknown_name() -> None:
    """Every command is implemented now, so the contract is a real argument check."""
    result = runner.invoke(app, ["run", "experiment", "not-an-experiment"])
    assert result.exit_code != 0
    assert "unknown experiment" in result.output


def test_run_experiment_lists_the_experiments_it_knows() -> None:
    result = runner.invoke(app, ["run", "experiment", "nope"])
    for name in ("contention", "target-step", "matrix", "profile"):
        assert name in result.output


def test_figures_make_fails_loudly_on_a_missing_run() -> None:
    """Section 3 rule 1: a missing input must fail, never draw something plausible."""
    result = runner.invoke(app, ["figures", "make", "definitely-not-a-run-dir"])
    assert result.exit_code != 0


def test_no_command_raises_not_implemented_any_more() -> None:
    """Phase 8 wired the last two; a stub left behind would be a silent gap."""

    from smartann import cli

    src = Path(cli.__file__).read_text(encoding="utf-8")
    assert "NotImplementedError" not in src


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output


# ---------------------------------------------------------------------------
# data subcommands (implemented in Phase 1)
# ---------------------------------------------------------------------------
def test_data_status_runs_on_an_empty_root() -> None:
    """Status must render before anything is downloaded, not crash.

    It exits 1 because nothing is prepared -- completeness is an exit code now,
    not a table to read past (ADR-0020). The point of this test is that the
    table still renders and no exception escapes.
    """
    result = runner.invoke(app, ["data", "status"])
    assert result.exception is None or isinstance(result.exception, SystemExit), result.output
    assert result.exit_code == 1, result.output
    for name in ("sift1m", "nytimes", "squad"):
        assert name in result.output


def test_data_status_reports_registry_facts() -> None:
    result = runner.invoke(app, ["data", "status"])
    assert "128" in result.output  # sift dim
    assert "256" in result.output  # nytimes dim
    assert "384" in result.output  # squad dim
    assert "cosine" in result.output
    assert "l2" in result.output


def test_data_verify_reports_missing_files() -> None:
    """With an isolated empty data root, every artifact is missing."""
    result = runner.invoke(app, ["data", "verify", "sift1m"])
    assert result.exit_code == 1
    assert "MISSING" in result.output


def test_data_groundtruth_declines_for_ann_datasets() -> None:
    """sift1m ships its own; recomputing would waste an hour for nothing."""
    result = runner.invoke(app, ["data", "groundtruth", "sift1m"])
    assert result.exit_code == 0
    assert "ships exact top-100" in result.output


def test_data_prepare_rejects_an_unknown_dataset() -> None:
    result = runner.invoke(app, ["data", "prepare", "not-a-dataset"])
    assert result.exit_code != 0
    assert isinstance(result.exception, KeyError)


# ---------------------------------------------------------------------------
# index subcommands (implemented in Phase 2)
# ---------------------------------------------------------------------------
def test_index_info_rejects_a_directory_with_no_index(tmp_path) -> None:  # type: ignore[no-untyped-def]
    result = runner.invoke(app, ["index", "info", str(tmp_path)])
    assert result.exit_code != 0
    assert "meta.json missing" in result.output


def test_index_build_rejects_an_unknown_dataset(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("dataset: not_a_dataset\n", encoding="utf-8")
    result = runner.invoke(app, ["index", "build", str(cfg)])
    assert result.exit_code != 0


def test_index_build_config_has_no_space_key(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Declaring the metric beside the build parameters is the two-sources bug."""
    cfg = tmp_path / "withspace.yaml"
    cfg.write_text("dataset: sift1m\nspace: l2\n", encoding="utf-8")
    result = runner.invoke(app, ["index", "build", str(cfg)])
    assert result.exit_code != 0

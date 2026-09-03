"""Phase 0 acceptance tests: the CLI surface.

These lock in the acceptance check that ``smartann --help`` lists the
subcommands, and that unimplemented commands fail loudly rather than returning
a plausible value (Section 3, rule 2).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from smartann import __version__
from smartann.cli import app

runner = CliRunner()

EXPECTED_SUBCOMMANDS = ["version", "paths", "data", "index", "config", "run", "figures"]

# Commands that must raise NotImplementedError until their phase lands.
STUBS = [
    ["data", "status"],
    ["data", "prepare", "squad"],
    ["data", "groundtruth", "sift1m"],
    ["index", "build", "x.yaml"],
    ["index", "info", "sift1m"],
    ["run", "experiment", "E1.yaml"],
    ["figures", "make", "E1"],
]


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
        "nprobe: 4\nef_centroid: 64\nef_list: 64\nk: 5\nrerank_candidates: 20\n",
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


@pytest.mark.parametrize("argv", STUBS, ids=lambda a: " ".join(a))
def test_stub_raises_not_implemented(argv: list[str]) -> None:
    """Unimplemented commands must fail loudly, never return a dummy value."""
    result = runner.invoke(app, argv)
    assert result.exit_code != 0
    assert isinstance(result.exception, NotImplementedError)
    assert "Phase" in str(result.exception), "the error must name the delivering phase"


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output

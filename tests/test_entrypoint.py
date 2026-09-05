"""The CLI entry points must actually run the CLI.

For thirty-nine commits ``smartann/cli.py`` had no ``if __name__ ==
"__main__"`` guard, so ``python -m smartann.cli data status`` imported the
module, defined every command, ran none of them, and exited **0** with no
output. Every runbook command used that form, and ``verify_all.py`` scored the
vacuous exit 0 as a passing data-status check -- a step whose verdict was
independent of its outcome, which is defect #1 of this project's own taxonomy
appearing for the third time (ADR-0042).

These tests run the entry points as real subprocesses. An in-process
``CliRunner`` cannot catch this bug: it invokes ``app`` directly and so tests
the one path that was never broken. The decisive assertion is on **stdout being
non-empty**, because a silent no-op and a successful run are distinguishable
only by what they emit.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

ENTRY_POINTS = [
    [sys.executable, "-m", "smartann.cli"],
    [sys.executable, "-m", "smartann"],
]


@pytest.mark.parametrize("argv", ENTRY_POINTS, ids=["module-path", "package"])
def test_entry_point_produces_output(argv: list[str]) -> None:
    """``--help`` must print. A no-op module exits 0 with an empty stdout."""
    proc = subprocess.run(
        [*argv, "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip(), (
        "entry point produced no output: the module was imported but the Typer "
        "app was never invoked"
    )


@pytest.mark.parametrize("argv", ENTRY_POINTS, ids=["module-path", "package"])
def test_entry_point_lists_the_subcommands(argv: list[str]) -> None:
    """Help text naming the sub-apps proves the Typer app itself ran."""
    proc = subprocess.run(
        [*argv, "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        encoding="utf-8",
        errors="replace",
    )
    for sub in ("data", "index", "config", "run", "figures"):
        assert sub in proc.stdout, f"{sub!r} missing from --help; sub-apps not wired"


@pytest.mark.parametrize("argv", ENTRY_POINTS, ids=["module-path", "package"])
def test_unknown_subcommand_fails(argv: list[str]) -> None:
    """An unknown command must exit non-zero.

    The sharpest form of the regression: the broken entry point exited 0 for
    *every* argument list, including nonsense, because it never parsed one.
    """
    proc = subprocess.run(
        [*argv, "definitely-not-a-command"],
        capture_output=True,
        text=True,
        timeout=120,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode != 0, "unknown subcommand exited 0; arguments are not being parsed"

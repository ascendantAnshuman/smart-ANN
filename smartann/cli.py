"""``smartann`` command line entry point.

Subcommands that are not implemented yet raise ``NotImplementedError`` with the
phase that will deliver them (Section 3, rule 2). They are never stubbed with a
plausible return value, and each one has a matching line in ``TODO.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from smartann import __version__
from smartann.config import config_hash, dump_yaml, resolve_paths

app = typer.Typer(
    name="smartann",
    help="SMART-ANN: runtime resource-aware adaptive control for on-device RAG retrieval.",
    no_args_is_help=True,
    add_completion=False,
)

data_app = typer.Typer(name="data", help="Dataset download, chunking and ground truth.")
index_app = typer.Typer(name="index", help="Build and inspect EcoVector indexes.")
config_app = typer.Typer(name="config", help="Inspect and hash configuration files.")
run_app = typer.Typer(name="run", help="Execute benchmark experiments.")
figures_app = typer.Typer(name="figures", help="Regenerate figures from stored parquet.")

app.add_typer(data_app)
app.add_typer(index_app)
app.add_typer(config_app)
app.add_typer(run_app)
app.add_typer(figures_app)


@app.command()
def version() -> None:
    """Print the SMART-ANN version."""
    typer.echo(__version__)


@app.command()
def paths() -> None:
    """Show the resolved data and results roots."""
    p = resolve_paths()
    typer.echo(dump_yaml(p).rstrip())


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
@config_app.command("hash")
def config_hash_cmd(
    path: Path = typer.Argument(..., exists=True, readable=True, help="YAML config file."),
) -> None:
    """Print the deterministic 12-character hash of a config file."""
    import yaml

    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    typer.echo(config_hash(data))


@config_app.command("show")
def config_show_cmd(
    path: Path = typer.Argument(..., exists=True, readable=True, help="YAML config file."),
    as_json: bool = typer.Option(False, "--json", help="Emit canonical JSON instead of YAML."),
) -> None:
    """Print a config in canonical form, exactly as it is hashed."""
    import yaml

    from smartann.config import canonical_dict

    data = canonical_dict(yaml.safe_load(path.read_text(encoding="utf-8")))
    if as_json:
        typer.echo(json.dumps(data, indent=2, sort_keys=True))
    else:
        typer.echo(dump_yaml(data).rstrip())


# ---------------------------------------------------------------------------
# not yet implemented -- see TODO.md
# ---------------------------------------------------------------------------
@data_app.command("status")
def data_status() -> None:
    """Table of which datasets are downloaded / processed / ground-truthed."""
    raise NotImplementedError("Phase 1: dataset preparation is not implemented yet.")


@data_app.command("prepare")
def data_prepare(name: str = typer.Argument(...)) -> None:
    """Download and process a dataset."""
    raise NotImplementedError("Phase 1: dataset preparation is not implemented yet.")


@data_app.command("groundtruth")
def data_groundtruth(name: str = typer.Argument(...)) -> None:
    """Compute exact top-100 neighbours by brute force."""
    raise NotImplementedError("Phase 1: ground-truth computation is not implemented yet.")


@index_app.command("build")
def index_build(config: Path = typer.Argument(...)) -> None:
    """Build an EcoVector index from an index config."""
    raise NotImplementedError("Phase 2: EcoVector index build is not implemented yet.")


@index_app.command("info")
def index_info(name: str = typer.Argument(...)) -> None:
    """Print index metadata and the predicted memory model."""
    raise NotImplementedError("Phase 2: EcoVector index build is not implemented yet.")


@run_app.command("experiment")
def run_experiment(
    config: Path = typer.Argument(...),
    seed: int = typer.Option(0, help="Random seed."),
) -> None:
    """Run one experiment definition end to end."""
    raise NotImplementedError("Phase 8: the experiment runner is not implemented yet.")


@figures_app.command("make")
def figures_make(experiment: str = typer.Argument(...)) -> None:
    """Regenerate all figures for an experiment from its stored parquet."""
    raise NotImplementedError("Phase 8: figure generation is not implemented yet.")


if __name__ == "__main__":  # pragma: no cover
    app()

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
# data
# ---------------------------------------------------------------------------
def _fmt_bytes(n: int) -> str:
    if n <= 0:
        return "-"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n = int(n) / 1024  # type: ignore[assignment]
    return f"{n} B"


@data_app.command("status")
def data_status() -> None:
    """Table of which datasets are downloaded / processed / ground-truthed."""
    from rich import box
    from rich.console import Console
    from rich.table import Table

    from smartann.data.manifest import DataManifest
    from smartann.data.prepare import status_table

    rows = status_table()
    paths_cfg = resolve_paths()

    # A fixed width keeps every column present regardless of the host terminal;
    # at the default 80 columns rich silently drops columns, which would make
    # this table quietly incomplete rather than obviously wrapped.
    console = Console(width=118)
    console.print(f"[bold]SMART-ANN datasets[/bold]   data root: {paths_cfg.data_root}")

    table = Table(box=box.SIMPLE, pad_edge=False)
    table.add_column("dataset")
    table.add_column("kind")
    for col in ("dl", "ver", "proc", "gt"):
        table.add_column(col, justify="center")
    table.add_column("n_base", justify="right")
    table.add_column("n_queries", justify="right")
    table.add_column("dim", justify="right")
    table.add_column("space")
    table.add_column("search")
    table.add_column("Nc", justify="right")
    table.add_column("disk", justify="right")

    def tick(ok: bool) -> str:
        return "[green]OK[/green]" if ok else "[red]x[/red]"

    for s in rows:
        table.add_row(
            s.name,
            s.kind,
            tick(s.downloaded),
            tick(s.verified),
            tick(s.processed),
            tick(s.ground_truthed),
            f"{s.n_base:,}" if s.n_base is not None else "-",
            f"{s.n_queries:,}" if s.n_queries is not None else "-",
            str(s.dim),
            s.space,
            s.search_space,
            str(s.resolved_nc) if s.resolved_nc is not None else "-",
            _fmt_bytes(s.bytes_on_disk),
        )
    console.print(table)

    manifest = DataManifest(paths_cfg.data_root)
    missing = manifest.missing_files()
    unrecorded = manifest.unrecorded_files()
    console.print(
        f"MANIFEST.json  {len(manifest)} artifacts, {_fmt_bytes(manifest.total_bytes())} recorded"
    )
    if missing:
        console.print(
            f"[red]  {len(missing)} recorded artifact(s) missing from disk:[/red] {missing[:5]}"
        )
    if unrecorded:
        console.print(
            f"[red]  {len(unrecorded)} file(s) on disk with no manifest entry:[/red] {unrecorded[:5]}"
        )
    if not missing and not unrecorded:
        console.print("[green]  every artifact on disk has a manifest entry[/green]")


@data_app.command("prepare")
def data_prepare(
    name: str = typer.Argument(..., help="Dataset name, or 'all'."),
    allow_mirror: bool = typer.Option(
        False, "--allow-mirror", help="Permit fallback to a registered mirror."
    ),
    no_record: bool = typer.Option(
        False,
        "--no-record-checksum",
        help="Do not write observed checksums to configs/datasets.yaml.",
    ),
    force: bool = typer.Option(False, "--force", help="Reprocess even if outputs exist."),
) -> None:
    """Download and process a dataset."""
    from smartann.data.prepare import prepare_dataset
    from smartann.data.registry import load_registry

    names = sorted(load_registry()) if name == "all" else [name]
    for n in names:
        typer.echo(f"=== preparing {n}")
        result = prepare_dataset(n, record=not no_record, allow_mirror=allow_mirror, force=force)
        typer.echo(
            f"    n_base={result.n_base:,}  n_queries={result.n_queries:,}  dim={result.dim}  "
            f"space={result.space} search_space={result.search_space}  Nc={result.resolved_nc}"
        )
        for k, v in sorted(result.details.items()):
            typer.echo(f"    {k}: {v}")


@data_app.command("verify")
def data_verify(name: str = typer.Argument(..., help="Dataset name, or 'all'.")) -> None:
    """Re-checksum a dataset's raw artifacts against the recorded values."""
    from smartann.data.download import sha256_file
    from smartann.data.registry import load_registry, spec_for

    names = sorted(load_registry()) if name == "all" else [name]
    failures = 0
    paths_cfg = resolve_paths()
    for n in names:
        spec = spec_for(n)
        for art in spec.artifacts:
            p = paths_cfg.raw / n / art.filename
            if not p.exists():
                typer.echo(f"[MISSING] {n}/{art.key}: {p}")
                failures += 1
                continue
            if art.sha256 is None:
                typer.echo(f"[NO RECORD] {n}/{art.key}: no recorded checksum to verify against")
                continue
            digest = sha256_file(p)
            if digest == art.sha256:
                typer.echo(f"[OK]      {n}/{art.key}: {digest[:16]}...")
            else:
                typer.echo(
                    f"[MISMATCH] {n}/{art.key}\n  recorded {art.sha256}\n  on disk  {digest}"
                )
                failures += 1
    if failures:
        raise typer.Exit(code=1)


@data_app.command("groundtruth")
def data_groundtruth(
    name: str = typer.Argument(...),
    k: int = typer.Option(100, help="Neighbours per query."),
    force: bool = typer.Option(False, "--force", help="Recompute even if it exists."),
) -> None:
    """Compute exact top-k neighbours by brute force (QA datasets only)."""
    import numpy as np

    from smartann.data.download import sha256_file
    from smartann.data.groundtruth import compute_topk
    from smartann.data.manifest import DataManifest
    from smartann.data.readers import load_processed
    from smartann.data.registry import spec_for

    spec = spec_for(name)
    paths_cfg = resolve_paths()
    out_dir = paths_cfg.processed / name
    gt_path = out_dir / "groundtruth_top100.npy"

    if spec.ships_groundtruth:
        typer.echo(f"{name} ships exact top-100 in its HDF5 file; nothing to compute.")
        return
    if gt_path.exists() and not force:
        typer.echo(f"{gt_path} exists; pass --force to recompute.")
        return

    base = load_processed(out_dir, "base")
    queries = load_processed(out_dir, "queries")
    typer.echo(
        f"brute force: {queries.shape[0]:,} queries x {base.shape[0]:,} base, space={spec.search_space}"
    )

    result = compute_topk(
        base,
        queries,
        k=k,
        space=spec.search_space,
        checkpoint=out_dir / "groundtruth.ckpt.npz",
    )
    np.save(gt_path, result.neighbors)

    manifest = DataManifest(paths_cfg.data_root)
    manifest.record(
        gt_path,
        dataset=name,
        stage="groundtruth",
        sha256=sha256_file(gt_path),
        extra={"k": k, "space": spec.search_space, "method": "exact brute force"},
    )
    manifest.write()
    ckpt = out_dir / "groundtruth.ckpt.npz"
    if ckpt.exists():
        ckpt.unlink()
    typer.echo(f"wrote {gt_path}  shape={result.neighbors.shape}")


# ---------------------------------------------------------------------------
# not yet implemented -- see TODO.md
# ---------------------------------------------------------------------------
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

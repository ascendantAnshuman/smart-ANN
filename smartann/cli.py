"""``smartann`` command line entry point.

Subcommands that are not implemented yet raise ``NotImplementedError`` with the
phase that will deliver them (Section 3, rule 2). They are never stubbed with a
plausible return value, and each one has a matching line in ``TODO.md``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

if TYPE_CHECKING:  # heavy imports stay out of CLI startup
    import numpy as np

    from smartann.config import PathsConfig
    from smartann.data.registry import DatasetSpec

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
    for col in ("dl", "ver", "proc", "gt", "gtd"):
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
            tick(s.gt_dists),
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

    # Exit status, not prose. A check a human reads is not a check.
    incomplete = [s for s in rows if not s.complete]
    if incomplete or missing or unrecorded:
        for s in incomplete:
            console.print(f"[red]  {s.name} is incomplete:[/red] missing {', '.join(s.missing())}")
        raise typer.Exit(code=1)


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
    typer.echo("computing ground-truth distances")
    _write_gt_dists(name, spec, out_dir, base, queries, result.neighbors, paths_cfg)
    ckpt = out_dir / "groundtruth.ckpt.npz"
    if ckpt.exists():
        ckpt.unlink()
    typer.echo(f"wrote {gt_path}  shape={result.neighbors.shape}")


def _write_gt_dists(
    name: str,
    spec: DatasetSpec,
    out_dir: Path,
    base: np.ndarray,
    queries: np.ndarray,
    neighbors: np.ndarray,
    paths_cfg: PathsConfig,
) -> None:
    """Write ``groundtruth_dists_top100.npy`` and verify it orders the ids."""
    import numpy as np

    from smartann.data.download import sha256_file
    from smartann.data.groundtruth import DIST_CONVENTION, check_row_monotonic, gather_distances
    from smartann.data.manifest import DataManifest
    from smartann.data.readers import duplicate_report

    space = spec.search_space
    dists = gather_distances(base, queries, neighbors, space)

    report = check_row_monotonic(dists)
    typer.echo(
        f"  monotonicity: {report.n_queries - report.n_violating_rows}/{report.n_queries} rows "
        f"non-decreasing, worst violation {report.worst_violation:.3e} "
        f"(query {report.worst_query}, depth {report.worst_depth}->{report.worst_depth + 1})"
    )
    if not report.ok:
        raise typer.BadParameter(
            f"{name}: ground-truth ids are not ordered by their own distances "
            f"(worst {report.worst_violation:.3e} < -{report.tol:g}). Either the remapping "
            "shifted an id onto the wrong row or the space is wrong -- do not ship this."
        )

    path = out_dir / "groundtruth_dists_top100.npy"
    np.save(path, dists)
    manifest = DataManifest(paths_cfg.data_root)
    manifest.record(
        path,
        dataset=name,
        stage="groundtruth",
        sha256=sha256_file(path),
        extra={
            "space": space,
            # Stored in hnswlib's units, so retrieved and true distances compare
            # directly with no transform at the comparison site.
            "dists_convention": DIST_CONVENTION[space],
            "aligned_with": "groundtruth_top100.npy",
            "depth": int(dists.shape[1]),
            "method": "exact, gathered against the stored neighbour ids",
            "monotonic_worst_violation": report.worst_violation,
            "monotonic_violating_rows": report.n_violating_rows,
            # float64 on this side only. Retrieved distances come back from
            # hnswlib as float32 by necessity, so the asymmetry is deliberate:
            # it keeps the recall band's atol covering one noise floor, not two.
            "dtype": str(dists.dtype),
            "retrieved_side_dtype": "float32 (hnswlib)",
        },
    )

    dup = duplicate_report(np.asarray(base))
    typer.echo(
        f"  duplicate base vectors: {dup.n_repeat_rows:,}/{dup.n_rows:,} "
        f"({dup.repeat_fraction * 100:.3f}%) are byte-identical repeats, "
        f"{dup.n_groups:,} groups, largest {dup.largest_group:,}"
    )
    base_path = out_dir / "base.npy"
    if manifest.get(base_path) is not None:
        manifest.annotate(
            base_path,
            {
                "duplicate_rows": dup.n_repeat_rows,
                "duplicate_groups": dup.n_groups,
                "largest_duplicate_group": dup.largest_group,
                "distinct_rows": dup.n_distinct,
            },
        )
    manifest.write()
    typer.echo(f"wrote {path}  shape={dists.shape}  units={DIST_CONVENTION[space]} {dists.dtype}")


@data_app.command("gt-dists")
def data_gt_dists(
    name: str = typer.Argument(...),
    force: bool = typer.Option(False, "--force", help="Recompute even if it exists."),
) -> None:
    """Compute exact distances for the stored ground-truth neighbours.

    Score-equality recall needs the true distance at depth k, which the shipped
    HDF5 files give in their own units against the pre-compaction base. This
    recomputes them from the vectors we actually index.
    """
    from smartann.data.readers import load_processed
    from smartann.data.registry import spec_for

    spec = spec_for(name)
    paths_cfg = resolve_paths()
    out_dir = paths_cfg.processed / name
    path = out_dir / "groundtruth_dists_top100.npy"
    if path.exists() and not force:
        typer.echo(f"{path} exists; pass --force to recompute.")
        return

    base = load_processed(out_dir, "base")
    queries = load_processed(out_dir, "queries")
    neighbors = load_processed(out_dir, "groundtruth")
    typer.echo(f"{name}: {neighbors.shape[0]:,} queries x depth {neighbors.shape[1]}")
    _write_gt_dists(name, spec, out_dir, base, queries, neighbors, paths_cfg)


@index_app.command("build")
def index_build(
    config: Path = typer.Argument(..., help="Path to an index build config YAML."),
    force: bool = typer.Option(False, "--force", help="Rebuild even if the artifact exists."),
) -> None:
    """Build an index. The metric comes from the dataset registry, never the config."""
    from smartann.config import config_hash, load_yaml
    from smartann.data.readers import load_processed
    from smartann.data.registry import spec_for
    from smartann.index.baselines import FLAT_EVERYTHING, build_flat_hnsw
    from smartann.index.builder import build_ecovector
    from smartann.index.storage import IndexBuildConfig, IndexPaths

    cfg = load_yaml(config, IndexBuildConfig)
    spec = spec_for(cfg.dataset)
    paths_cfg = resolve_paths()
    processed = paths_cfg.processed / cfg.dataset

    base = load_processed(processed, "base", mmap=False)
    nc = spec.resolve_nc(base.shape[0]) if cfg.nc == "auto" else int(cfg.nc)
    chash = config_hash(cfg)
    out = IndexPaths.for_build(paths_cfg.data_root / "index", cfg.dataset, chash)

    if out.meta.exists() and not force:
        typer.echo(f"{out.root} exists; pass --force to rebuild.")
        return
    if out.root.exists():
        # Clear rather than overwrite. A cluster that changes from graph to flat
        # leaves a stale .bin beside the new .npz -- never loaded, but counted in
        # bytes_on_disk, which is one of the numbers the paper reports.
        shutil.rmtree(out.root)

    # The metric is read here and asserted, not declared alongside the build
    # parameters. Two sources of truth for the space is the bug that silently
    # halves recall (ADR-0010).
    space = spec.search_space
    typer.echo(
        f"building {cfg.index_variant} for {cfg.dataset}: {base.shape[0]:,}x{base.shape[1]} "
        f"space={space} nc={nc} M={cfg.m} efC={cfg.ef_construction} hash={chash}"
    )

    if cfg.index_variant == "flat_hnsw":
        meta = build_flat_hnsw(
            base,
            dataset=cfg.dataset,
            space=space,
            m=cfg.m,
            ef_construction=cfg.ef_construction,
            seed=cfg.seed,
            config_hash=chash,
            out=out,
        )
    else:
        # ivf_disk is the same build with every cluster flat, so the ablation
        # holds partition, files and load path fixed by construction.
        min_graph = FLAT_EVERYTHING if cfg.index_variant == "ivf_disk" else cfg.min_graph_size
        meta = build_ecovector(
            base,
            dataset=cfg.dataset,
            space=space,
            nc=nc,
            m=cfg.m,
            ef_construction=cfg.ef_construction,
            min_graph_size=min_graph,
            seed=cfg.seed,
            config_hash=chash,
            out=out,
            index_variant=cfg.index_variant,
            kmeans_batch_size=cfg.kmeans_batch_size,
            kmeans_n_init=cfg.kmeans_n_init,
        ).meta

    typer.echo(
        f"built in {meta.build_seconds:.1f}s  peak build RSS {meta.build_peak_rss_mb:.0f} MB  "
        f"{meta.build_workers} workers  {_fmt_bytes(meta.bytes_on_disk)} on disk"
    )
    typer.echo(
        f"  clusters: {meta.n_graph_clusters} graph, {meta.n_flat_clusters} flat, "
        f"{meta.n_empty_clusters} empty; largest {meta.largest_cluster:,}, "
        f"smallest non-empty {meta.smallest_nonempty_cluster:,}; "
        f"{meta.n_duplicate_centroids} duplicate centroid(s)"
    )
    typer.echo(f"  {out.root}")


@index_app.command("info")
def index_info(root: Path = typer.Argument(..., help="Path to a built index directory.")) -> None:
    """Print index metadata and the predicted memory model (Section 3.4)."""
    from smartann.index.storage import IndexMeta, IndexPaths, memory_model_mb

    paths = IndexPaths(root)
    if not paths.meta.exists():
        raise typer.BadParameter(f"no index at {root} (meta.json missing)")
    meta = IndexMeta.from_json(paths.meta)

    typer.echo(f"{meta.index_variant}  {meta.dataset}  hash={meta.config_hash}")
    for key in (
        "space",
        "dim",
        "n_base",
        "nc",
        "m",
        "ef_construction",
        "min_graph_size",
        "seed",
        "kmeans_algorithm",
        "built_at",
        "build_seconds",
        "build_peak_rss_mb",
        "build_workers",
        "n_graph_clusters",
        "n_flat_clusters",
        "n_empty_clusters",
        "largest_cluster",
        "smallest_nonempty_cluster",
        "n_duplicate_centroids",
    ):
        typer.echo(f"  {key:26s} {getattr(meta, key)}")
    typer.echo(f"  {'bytes_on_disk':26s} {_fmt_bytes(meta.bytes_on_disk)}")

    if meta.nc:
        typer.echo("\npredicted resident footprint (Section 3.4), MB:")
        for term, mb in memory_model_mb(
            meta.nc, meta.dim, meta.n_base, meta.largest_cluster, meta.m
        ).items():
            typer.echo(f"  {term:26s} {mb:10.3f}")


# ---------------------------------------------------------------------------
# not yet implemented -- see TODO.md
# ---------------------------------------------------------------------------


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

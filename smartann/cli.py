"""``smartann`` command line entry point.

Every subcommand is implemented. While some were not, they raised with the phase
that would deliver them rather than returning a plausible value (Section 3, rule
2); that convention is kept for anything added later.

The experiment and figure commands are thin front ends over the scripts that
implement them, not second copies of their logic. Two implementations of an
experiment is two experiments, and only one of them produced the recorded
numbers.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from click import exceptions as click_exceptions

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
thermal_app = typer.Typer(name="thermal", help="Explain and score device heat from a trace.")
kb_app = typer.Typer(name="kb", help="Grow the knowledge base that ask searches beside SQuAD.")

app.add_typer(data_app)
app.add_typer(index_app)
app.add_typer(config_app)
app.add_typer(run_app)
app.add_typer(figures_app)
app.add_typer(thermal_app)
app.add_typer(kb_app)


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


@run_app.command("experiment")
def run_experiment(
    name: str = typer.Argument(..., help="contention | target-step | matrix | profile"),
    dataset: str = typer.Option("sift1m", help="Dataset to run against."),
    seconds: float = typer.Option(60.0, help="Seconds per phase or per arm."),
) -> None:
    """Run one experiment end to end.

    A thin front end over the scripts that already implement these, rather than a
    second copy of their logic. The scripts stay the source of truth -- two
    implementations of an experiment is two experiments, and only one of them
    would be the one that produced the recorded numbers.
    """
    import subprocess

    scripts = Path(__file__).resolve().parent.parent / "scripts"
    plans: dict[str, list[str]] = {
        "contention": [
            str(scripts / "experiment_contention.py"),
            "--dataset",
            dataset,
            "--quiet-s",
            str(seconds),
            "--contended-s",
            str(seconds * 2),
            "--recovery-s",
            str(seconds),
        ],
        "target-step": [
            str(scripts / "experiment_contention.py"),
            "--dataset",
            dataset,
            "--mode",
            "target-step",
            "--quiet-s",
            str(seconds),
            "--contended-s",
            str(seconds * 2),
            "--recovery-s",
            str(seconds),
        ],
        "matrix": [str(scripts / "experiment_matrix.py"), "--seconds-per-arm", str(seconds)],
        "profile": [
            str(scripts / "phase6_profile.py"),
            "--dataset",
            dataset,
            "--profile-seconds",
            str(seconds / 4),
            "--compare-seconds",
            str(seconds),
        ],
    }
    if name not in plans:
        raise typer.BadParameter(f"unknown experiment {name!r}; choose from {sorted(plans)}")

    typer.echo(f"NOTE: nothing else must run while this measures (ADR-0030). Starting {name}.")
    raise typer.Exit(subprocess.run([sys.executable, *plans[name]], check=False).returncode)


@figures_app.command("make")
def figures_make(
    runs: Path = typer.Argument(None, help="Run directory; defaults to the newest E_DEMO_*."),
    out: Path = typer.Option(Path("results/figures"), help="Where to write the figures."),
) -> None:
    """Regenerate figures for a run from its stored parquet.

    Every number comes from the parquet or from a summary JSON; a missing input
    fails loudly rather than drawing something plausible (Section 3, rule 1).
    """
    import subprocess

    script = Path(__file__).resolve().parent.parent / "scripts" / "make_figures.py"
    argv = [sys.executable, str(script), "--out", str(out)]
    if runs is not None:
        argv += ["--runs", str(runs)]
    raise typer.Exit(subprocess.run(argv, check=False).returncode)


# Printed once when `ask` runs with no question. Short on purpose: the panel
# that follows each answer is long, and a wall of help above it would push the
# first answer off the screen during a demo.
_ASK_BANNER = """
  Type a question and press Enter. 'quit' ends the session.
  The models stay loaded, so every question after the first is fast.
"""


def _open_replay(name: str, sensor: str) -> Any:
    """A recorded handset trace, or a message naming what is on disk instead."""
    from smartann.telemetry.replay import ReplayFeed, available_traces, load_trace, trace_path

    device_dir = Path(resolve_paths().results_root) / "device"
    if not device_dir.is_dir():
        device_dir = Path("results") / "device"
    path = trace_path(device_dir, name)
    if not path.exists():
        have = available_traces(device_dir) or ["none on disk"]
        typer.echo(
            f"no recorded trace named {name!r} in {device_dir}. Have: {', '.join(have)}", err=True
        )
        raise typer.Exit(2)
    return ReplayFeed(load_trace(path, sensor=sensor))


@app.command()
def ask(
    question: str = typer.Argument(None, help="A question in quotes; omit it to type questions."),
    budget_ms: float = typer.Option(
        None, "--budget-ms", help="Latency budget the controller honours."
    ),
    nprobe: int = typer.Option(8, help="Effort level, when no budget is given."),
    k: int = typer.Option(5, help="Chunks to retrieve."),
    dataset: str = typer.Option("squad", help="Which built index to answer from."),
    scr: bool = typer.Option(False, "--scr", help="Turn context reduction on."),
    top_windows: int = typer.Option(2, help="Windows kept per chunk when --scr is set."),
    table: Path = typer.Option(
        Path("results/device_profile_table.json"),
        help="Measured profile table; must name the same dataset.",
    ),
    show_context: bool = typer.Option(True, help="Print the retrieved passages."),
    json_out: Path = typer.Option(None, help="Also write the full result as JSON."),
    model: str = typer.Option(
        "google/flan-t5-base",
        help="Generator. google/flan-t5-large is larger and slower; not benchmarked here.",
    ),
    verify: bool = typer.Option(
        True, help="Withhold a span unless the model judges the passages contain the answer."
    ),
    fallback: bool = typer.Option(
        False,
        "--fallback",
        help="When the corpus cannot answer, ask the model from memory. Measured: flan-t5-base "
        "got 4 of 4 test questions wrong this way.",
    ),
    replay: str = typer.Option(
        None,
        "--replay",
        help="Drive the controller from a recorded handset trace, e.g. c5. "
        "Each question advances along it, so the device warms as you ask.",
    ),
    sensor: str = typer.Option(
        "hottest",
        "--sensor",
        help="Which thermometer the policy reads: hottest (inside) or skin (the case).",
    ),
    query_adaptive: bool = typer.Option(
        False,
        "--query-adaptive",
        help="Spend effort by measured query difficulty (C6). Needs a phase 9 fit at "
        "this exact budget; run scripts/phase9_query_adaptive.py first.",
    ),
    fit_path: Path = typer.Option(
        Path("results/phase9/query_adaptive.json"),
        help="The recorded phase 9 allocation --query-adaptive spends.",
    ),
    explain: bool = typer.Option(
        False,
        "--explain",
        help="Print the query trace as a step-by-step story instead of the report: "
        "question, device, controller decision, clusters opened, passages, answer, cost.",
    ),
    no_kb: bool = typer.Option(
        False,
        "--no-kb",
        help="Answer from the index alone, ignoring the knowledge base. Use it when a "
        "demonstration must show one corpus and not a merge of two.",
    ),
    trace_dir: Path = typer.Option(
        Path("traces"),
        help="Where each query's trace JSON is written. Every query writes one.",
    ),
    no_trace: bool = typer.Option(False, "--no-trace", help="Do not write trace files."),
    guard: bool = typer.Option(
        True,
        "--guard/--no-guard",
        help="Refuse to answer when the best passage scores below the calibrated "
        "threshold, instead of guessing. Needs a threshold measured for THIS dataset; "
        "without one the guard stays off and says so.",
    ),
    guard_path: Path = typer.Option(
        Path("results/grounding_threshold.json"),
        help="Calibrated refusal threshold, from scripts/phase12_calibrate.py.",
    ),
    cache: bool = typer.Option(
        False,
        "--cache/--no-cache",
        help="Reuse the answer to a question already asked in other words. Off by "
        "default: a reused answer hides whatever the controller would have done.",
    ),
    cache_path: Path = typer.Option(
        Path("results/cache_threshold.json"),
        help="Calibrated query-similarity threshold for the cache.",
    ),
    stress: str = typer.Option(
        None,
        "--stress",
        help="Inject a named device condition: normal, hot, hot-44, low-battery, "
        "battery-saver. Measured ones come from a recorded trace; the rest are "
        "labelled hypothetical. Overrides --replay.",
    ),
) -> None:
    """Answer a question end to end, showing what each stage decided and cost.

    Retrieval alone returns passages; this returns an answer, and prints the
    controller's choice, the evidence and the per-stage cost alongside it. It
    drives the same objects Phase 7 scores, so an answer here and an answer in
    the C2 results come from one implementation rather than two.

    Recall is not reported for a typed question. There is no ground truth for
    one, and the profile table's recall describes the effort level over the
    benchmark set rather than this query (Section 3, rule 1).
    """
    from smartann.pipeline.ask import SCR_OFF, RagPipeline
    from smartann.pipeline.ask_report import render
    from smartann.pipeline.explain import render_story
    from smartann.pipeline.querytrace import NetworkWitness, doc_slugs, from_ask_result
    from smartann.telemetry.provenance import device_evidence
    from smartann.types import ScrConfig

    # SQuAD carries IPA pronunciation guides, and a Windows console defaults to
    # cp1252, which cannot encode them. Without this the whole answer is lost to
    # a UnicodeEncodeError after the work has already been done.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    from smartann.telemetry.replay import SENSORS

    if sensor not in SENSORS:
        typer.echo(f"--sensor must be one of {', '.join(SENSORS)}", err=True)
        raise typer.Exit(2)
    feed = _open_replay(replay, sensor) if replay else None

    # 12.3: a named condition pins the device state for every question asked,
    # which is what makes a live demonstration of the controller possible. It
    # takes precedence over a replay, because a replay advances and a stress
    # mode is meant to hold still.
    stress_mode = None
    if stress is not None:
        from smartann.telemetry.stress import load_modes

        device_dir = Path(resolve_paths().results_root) / "device"
        if not device_dir.is_dir():
            device_dir = Path("results") / "device"
        modes = load_modes(device_dir)
        if stress not in modes:
            typer.echo(f"--stress must be one of {', '.join(modes)}; got {stress!r}", err=True)
            raise typer.Exit(2)
        stress_mode = modes[stress]
        if feed is not None:
            typer.echo("NOTE: --stress overrides --replay; the state will not advance.", err=True)
            feed = None
        typer.echo(f"stress mode {stress!r}: {stress_mode.detail}", err=True)

    the_guard = None
    if guard:
        from smartann.pipeline.guard import GroundingGuard

        if not guard_path.exists():
            typer.echo(
                f"NOTE: no calibrated threshold at {guard_path}, so the grounding guard "
                f"is OFF. Run scripts/phase12_calibrate.py to measure one.",
                err=True,
            )
        else:
            try:
                the_guard = GroundingGuard.load(guard_path, dataset=dataset)
            except ValueError as exc:
                typer.echo(f"NOTE: grounding guard OFF -- {exc}", err=True)

    the_cache: Any = None
    if cache:
        from smartann.pipeline.cache import SemanticCache

        threshold = 0.95
        if cache_path.exists():
            threshold = float(json.loads(cache_path.read_text(encoding="utf-8"))["threshold"])
        else:
            typer.echo(
                f"NOTE: no calibrated threshold at {cache_path}; falling back to "
                f"{threshold}, which is a guess and not a measurement.",
                err=True,
            )
        the_cache = SemanticCache(threshold=threshold)

    qfit = None
    if query_adaptive:
        from smartann.controller.policies.query import QueryEffortFit

        if budget_ms is None:
            typer.echo("--query-adaptive needs --budget-ms: the fit is per budget.", err=True)
            raise typer.Exit(2)
        try:
            qfit = QueryEffortFit.load(fit_path, budget_ms=budget_ms, dataset=dataset)
        except (OSError, ValueError, KeyError) as exc:
            typer.echo(f"cannot use the query-difficulty fit: {exc}", err=True)
            raise typer.Exit(2) from exc
        if not qfit.is_demonstrated:
            # Loadable, because a negative result must stay inspectable. Said
            # out loud, because nobody should demo a policy whose own held-out
            # comparison recorded no gain without knowing that.
            typer.echo(
                f"NOTE: this fit demonstrated nothing on its held-out half "
                f"(delta recall {qfit.delta_recall:+.4f}, cost saving "
                f"{100 * qfit.cost_saving:.1f}%). Running it anyway.",
                err=True,
            )
        if replay:
            typer.echo(
                "NOTE: --replay takes precedence; device state overrides query difficulty.",
                err=True,
            )

    cfg = ScrConfig(1, 0, 0, top_windows, 0.0, enabled=True) if scr else SCR_OFF
    pipeline = RagPipeline(
        dataset=dataset,
        k=k,
        profile_table_path=table,
        model_id=model,
        use_kb=not no_kb,
    )

    try:
        typer.echo("loading index, embedder and generator ...", err=True)
        pipeline.load()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        pipeline.close()
        typer.echo(f"cannot answer: {exc}", err=True)
        raise typer.Exit(2) from exc

    # Loading costs about thirty seconds, almost all of it the generator. Paying
    # that once and then answering many questions is the only reason an
    # interactive mode is worth having over repeated one-shot invocations.
    answers = []
    traces: list[Any] = []
    written: list[Path] = []
    # chunk -> document slug, read once. The trace cites documents, and the
    # retrieved chunk carries only a chunk id.
    slug_map = doc_slugs(Path(resolve_paths().data_root) / "processed" / dataset / "chunks.sqlite")
    # Computed once per session, not per query: scanning the recorded handset
    # runs costs tens of milliseconds and the answer does not change between
    # questions.
    _device_dir = Path(resolve_paths().results_root) / "device"
    if not _device_dir.is_dir():
        _device_dir = Path("results") / "device"
    evidence = device_evidence(_device_dir, Path("results/device_profile_table.json"))

    def _ask(text: str) -> Any:
        """One question, with the next measured device state when replaying.

        The network witness wraps the whole query rather than any one stage,
        because the claim being checked is about the query: that answering it
        sent nothing anywhere. A witness around retrieval alone would miss a
        generator that called home.
        """
        if stress_mode is not None:
            state, label = stress_mode.state, stress_mode.label
        elif feed is not None:
            state, label = feed.next()
        else:
            state, label = None, ""
        with NetworkWitness() as witness:
            result = pipeline.ask(
                text,
                budget_ms=budget_ms,
                nprobe=nprobe,
                scr=cfg,
                verify=verify,
                fallback=fallback,
                device=state,
                device_source=label,
                # A stress mode may be a typed hypothetical. Carrying it as
                # measured would put an invented temperature into a trace as a
                # reading, which is the one thing nothing here may do.
                device_is_measured=(
                    stress_mode.is_measured if stress_mode is not None else state is not None
                ),
                query_fit=qfit,
                guard=the_guard,
                cache=the_cache,
            )
        trace = from_ask_result(
            result,
            network_bytes_sent=witness.bytes_sent,
            network_connections=witness.connections,
            doc_ids=slug_map,
        )
        traces.append(trace)
        if not no_trace:
            written.append(trace.write(trace_dir))
        return result

    try:
        if question is not None:
            answers.append(_ask(question))
            typer.echo(
                render_story(traces[-1], evidence)
                if explain
                else render(answers[-1], show_context=show_context)
            )
        else:
            typer.echo(_ASK_BANNER)
            while True:
                try:
                    typed = typer.prompt("ask", prompt_suffix="> ").strip()
                except (EOFError, click_exceptions.Abort):
                    typer.echo("\nbye.")
                    break
                if typed.lower() in {"quit", "exit", ":q"}:
                    typer.echo("bye.")
                    break
                if not typed:
                    continue
                try:
                    answers.append(_ask(typed))
                except (RuntimeError, ValueError) as exc:
                    # One bad question must not end the session; the models are
                    # loaded and re-entering costs another thirty seconds.
                    typer.echo(f"cannot answer that: {exc}", err=True)
                    continue
                typer.echo(
                    render_story(traces[-1], evidence)
                    if explain
                    else render(answers[-1], show_context=show_context)
                )
    finally:
        pipeline.close()

    if written:
        label = "traces" if len(written) > 1 else "trace"
        typer.echo(f"\n{label}: " + ", ".join(str(w) for w in written[-3:]), err=True)

    if json_out is not None and answers:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = [a.as_dict() for a in answers]
        json_out.write_text(
            json.dumps(payload[0] if len(payload) == 1 else payload, indent=2), encoding="utf-8"
        )
        typer.echo(f"\nwrote {json_out}")


@app.command()
def dashboard(
    out: Path = typer.Option(Path("results/dashboard.html"), help="Where to write the page."),
    results: Path = typer.Option(Path("results"), help="Directory holding the result files."),
    prefix: str = typer.Option("c5_", help="Device trace to attribute heat from."),
    answers: Path = typer.Option(None, help="JSON written by `smartann ask --json-out`."),
    open_it: bool = typer.Option(False, "--open", help="Open the page in a browser."),
) -> None:
    """Build one self-contained HTML page from every result on disk.

    No value is typed by hand: each is read from the same JSON the figures and
    the paper read, so the page cannot drift from the results it describes. A
    section whose input is missing renders as *not run* rather than vanishing,
    because "we measured this and it was small" and "we never measured this"
    must not look the same (Section 3, rule 1).
    """
    from smartann.dashboard import collect
    from smartann.dashboard_html import render_html

    state = collect(results=results, thermal_prefix=prefix, answers_path=answers)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(state), encoding="utf-8")

    size_kb = out.stat().st_size / 1024
    typer.echo(f"wrote {out}  ({size_kb:,.0f} KB, self-contained)")
    if state.missing:
        typer.echo(f"sections with no input, shown as not run: {', '.join(state.missing)}")
    if open_it:
        import webbrowser

        webbrowser.open(out.resolve().as_uri())


@app.command()
def serve(
    port: int = typer.Option(8000, help="Port on 127.0.0.1."),
    budget_ms: float = typer.Option(
        120.0, "--budget-ms", help="Latency budget used when the page does not pick one."
    ),
    dataset: str = typer.Option("squad", help="Which built index to answer from."),
    model: str = typer.Option("google/flan-t5-base", help="Generator."),
    table: Path = typer.Option(
        Path("results/device_profile_table.json"), help="Measured profile table."
    ),
    open_it: bool = typer.Option(True, "--open/--no-open", help="Open a browser."),
) -> None:
    """Serve the ask page and the dashboard at http://127.0.0.1:8000.

    Loopback only: the page runs a local model over a local index, and putting
    that on a shared network would let anyone on it spend this machine's CPU.
    One question is answered at a time, so every latency reported is a query
    that ran alone.
    """
    import webbrowser

    from smartann.dashboard import collect
    from smartann.dashboard_html import render_html
    from smartann.pipeline.ask import RagPipeline
    from smartann.web import SingleThreadPipeline, TraceCatalog
    from smartann.web import serve as serve_http

    # Loaded and called on one worker thread: the embedding cache is a SQLite
    # connection, and SQLite refuses one used from a thread other than the one
    # that opened it. Requests arrive on their own threads.
    runner = SingleThreadPipeline(
        RagPipeline(dataset=dataset, profile_table_path=table, model_id=model),
        default_budget_ms=budget_ms,
    )
    try:
        typer.echo("loading index, knowledge base, embedder and generator ...")
        runner.load()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        typer.echo(f"cannot start: {exc}", err=True)
        raise typer.Exit(2) from exc

    catalog = TraceCatalog(Path("results") / "device")
    if catalog.names:
        typer.echo(f"recorded handset traces the page can replay: {', '.join(catalog.names)}")
    else:
        typer.echo("no recorded handset traces found; the page will offer this host only")

    httpd = serve_http(runner, lambda: render_html(collect()), port=port, traces=catalog)
    url = f"http://127.0.0.1:{port}/"
    typer.echo(f"ready at {url}   (Ctrl+C to stop)")
    if open_it:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        typer.echo("stopped.")
    finally:
        httpd.server_close()
        runner.close()


# ---------------------------------------------------------------------------
# knowledge base
# ---------------------------------------------------------------------------
# A curated starter set, chosen to cover the kinds of question a demo audience
# asks first and SQuAD's 2016 snapshot cannot answer: current office-holders,
# recent events, and everyday facts SQuAD never had an article for.
KB_STARTER = (
    "Google",
    "Alphabet Inc.",
    "Microsoft",
    "Apple Inc.",
    "OpenAI",
    "Cricket",
    "Association football",
    "2022 FIFA World Cup",
    "India",
    "Artificial intelligence",
    "Large language model",
    "Retrieval-augmented generation",
    "Nearest neighbor search",
    "Smartphone",
    "Lithium-ion battery",
)


def _kb_open() -> Any:
    from smartann.kb import KnowledgeBase

    return KnowledgeBase(Path(resolve_paths().data_root) / "kb")


def _kb_embedder() -> Any:
    from smartann.pipeline.embedder import Embedder

    return Embedder(cache_path=Path(resolve_paths().data_root) / "cache" / "kb.sqlite")


@kb_app.command("add-wiki")
def kb_add_wiki(
    titles: list[str] = typer.Argument(None, help="Wikipedia article titles."),
    starter: bool = typer.Option(False, "--starter", help="Add the curated starter set."),
) -> None:
    """Fetch Wikipedia articles and add them to the knowledge base.

    Each article is chunked exactly like SQuAD (three-sentence windows, one of
    overlap) and embedded with the same model, so its passages compete with the
    benchmark's on equal terms. The SQuAD index itself is never modified, so
    every recorded benchmark number stays reproducible.
    """
    from smartann.kb import fetch_wikipedia

    wanted = list(titles or []) + (list(KB_STARTER) if starter else [])
    if not wanted:
        typer.echo("give one or more titles, or --starter", err=True)
        raise typer.Exit(2)

    kb, emb = _kb_open(), _kb_embedder()
    failed = []
    try:
        for title in wanted:
            try:
                canonical, url, paras = fetch_wikipedia(title)
                info = kb.add(canonical, paras, url, "wikipedia", emb.encode)
            except (LookupError, ConnectionError, ValueError) as exc:
                failed.append(title)
                typer.echo(f"  skipped  {title}: {exc}", err=True)
                continue
            typer.echo(f"  added    {info.title:<34} {info.n_chunks:>5} chunks")
    finally:
        typer.echo(f"knowledge base now holds {len(kb):,} chunks from {len(kb.sources())} sources")
        kb.close()
    if failed:
        raise typer.Exit(1)


@kb_app.command("add-file")
def kb_add_file(
    paths_in: list[Path] = typer.Argument(..., help="Text or Markdown files, or folders of them."),
) -> None:
    """Add your own .txt or .md documents to the knowledge base."""
    from smartann.kb import read_text_file

    files: list[Path] = []
    for p in paths_in:
        if p.is_dir():
            files += sorted(q for q in p.rglob("*") if q.suffix.lower() in {".txt", ".md"})
        else:
            files.append(p)

    kb, emb = _kb_open(), _kb_embedder()
    failed = []
    try:
        for f in files:
            try:
                title, url, paras = read_text_file(f)
                info = kb.add(title, paras, url, "file", emb.encode)
            except (FileNotFoundError, ValueError) as exc:
                failed.append(f)
                typer.echo(f"  skipped  {f}: {exc}", err=True)
                continue
            typer.echo(f"  added    {info.title:<34} {info.n_chunks:>5} chunks")
    finally:
        typer.echo(f"knowledge base now holds {len(kb):,} chunks from {len(kb.sources())} sources")
        kb.close()
    if failed:
        raise typer.Exit(1)


@kb_app.command("list")
def kb_list() -> None:
    """Show every document in the knowledge base."""
    kb = _kb_open()
    try:
        srcs = kb.sources()
        if not srcs:
            typer.echo("knowledge base is empty; try `smartann kb add-wiki --starter`")
            return
        for s in srcs:
            typer.echo(f"  {s.title:<36} {s.kind:<10} {s.n_chunks:>5} chunks   added {s.added_at}")
        typer.echo(f"{len(kb):,} chunks from {len(srcs)} sources")
    finally:
        kb.close()


@kb_app.command("remove")
def kb_remove(title: str = typer.Argument(..., help="Exact title as shown by `kb list`.")) -> None:
    """Remove a document from the knowledge base."""
    kb = _kb_open()
    try:
        n = kb.remove(title)
        typer.echo(
            f"removed {n} source(s) titled '{title}'" if n else f"no source titled '{title}'"
        )
    finally:
        kb.close()
    if not n:
        raise typer.Exit(1)


@thermal_app.command("explain")
def thermal_explain(
    prefix: str = typer.Option("c5_", help="Trace file prefix, e.g. c4_ or c5_."),
    trace_dir: Path = typer.Option(Path("results/device"), help="Where the trace files live."),
    energy_model: Path = typer.Option(Path("results/energy_model.json")),
    thermal_model: Path = typer.Option(Path("results/thermal_model.json")),
    json_out: Path = typer.Option(None, help="Also write the attribution as JSON."),
) -> None:
    """Explain why the device got hot, and score how much of it retrieval controls.

    Answers the question the cable makes obvious: the phone is warm and nothing
    is charging it, so what is? Splits the measured battery draw into the idle
    floor, the retrieval workload, and whatever neither accounts for.

    Every input is a measurement. A missing trace or an energy fit that declared
    itself unreportable stops the command rather than producing a number
    (Section 3, rule 1).
    """
    from smartann.thermal.attribution import EnergyModel, ThermalModel, attribute, load_trace
    from smartann.thermal.report import render

    try:
        trace = load_trace(trace_dir, prefix)
        em = EnergyModel.load(energy_model)
        tm = ThermalModel.load(thermal_model)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"cannot attribute: {exc}", err=True)
        raise typer.Exit(2) from exc

    result = attribute(trace, em, tm)
    typer.echo(render(result))

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
        typer.echo(f"\nwrote {json_out}")


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess in tests
    # Without this, `python -m smartann.cli data status` imports the module,
    # defines every command, runs none of them, and exits 0. It is the worst
    # possible failure: silent, and indistinguishable from success. It stood
    # undetected because `verify_all.py` invoked the data-status step through
    # exactly this form and scored its vacuous exit 0 as a pass (ADR-0042).
    app()

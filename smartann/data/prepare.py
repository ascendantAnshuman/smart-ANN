"""Dataset preparation: download -> verify -> process -> record.

One orchestrator so ``smartann data prepare`` and ``scripts/prepare_datasets.py``
share a single code path, and so status reporting can ask the same questions
preparation answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from smartann.config import PathsConfig, resolve_paths
from smartann.data.download import fetch, sha256_file
from smartann.data.manifest import DataManifest
from smartann.data.readers import (
    l2_normalize,
    read_ann_benchmarks_hdf5,
)
from smartann.data.registry import DatasetSpec, load_registry, record_checksum, spec_for

__all__ = ["DatasetStatus", "PrepareResult", "prepare_dataset", "status_for", "status_table"]


@dataclass(frozen=True)
class PrepareResult:
    name: str
    n_base: int
    n_queries: int
    dim: int
    space: str
    search_space: str
    resolved_nc: int
    ships_groundtruth: bool
    bytes_on_disk: int
    details: dict[str, Any]


@dataclass(frozen=True)
class DatasetStatus:
    name: str
    kind: str
    downloaded: bool
    verified: bool
    processed: bool
    ground_truthed: bool
    n_base: int | None
    n_queries: int | None
    dim: int
    space: str
    search_space: str
    resolved_nc: int | None
    bytes_on_disk: int


def _raw_dir(paths: PathsConfig, name: str) -> Path:
    return paths.raw / name


def _processed_dir(paths: PathsConfig, name: str) -> Path:
    return paths.processed / name


def _dir_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_artifacts(
    spec: DatasetSpec,
    paths: PathsConfig,
    manifest: DataManifest,
    record: bool = True,
    allow_mirror: bool = False,
    quiet: bool = False,
) -> dict[str, Path]:
    """Fetch every artifact, verifying or recording its checksum."""
    out: dict[str, Path] = {}
    raw = _raw_dir(paths, spec.name)
    for art in spec.artifacts:
        dest = raw / art.filename
        result = fetch(
            art.urls,
            dest,
            expected_sha256=art.sha256,
            allow_mirror=allow_mirror,
            quiet=quiet,
        )
        if art.sha256 is None and record:
            record_checksum(
                spec.name,
                art.key,
                result.sha256,
                result.n_bytes,
                result.url,
            )
        manifest.record(
            dest,
            dataset=spec.name,
            stage="raw",
            sha256=result.sha256,
            source_url=result.url if not result.cached else art.resolved_url or art.primary_url,
            extra={"artifact_key": art.key, "cached": result.cached, "resumed": result.resumed},
        )
        out[art.key] = dest
    return out


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
def _process_ann(
    spec: DatasetSpec,
    raw_paths: dict[str, Path],
    out_dir: Path,
    manifest: DataManifest,
    quiet: bool = False,
) -> PrepareResult:
    ds = read_ann_benchmarks_hdf5(raw_paths["hdf5"], spec)

    base = ds.train
    queries = ds.test
    if spec.normalized:
        base = l2_normalize(base, f"{spec.name}/base")
        queries = l2_normalize(queries, f"{spec.name}/queries")

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[Path, str]] = []
    for arr, filename in (
        (base, "base.npy"),
        (queries, "queries.npy"),
        (ds.neighbors, "groundtruth_top100.npy"),
    ):
        p = out_dir / filename
        np.save(p, arr)
        written.append((p, filename))

    for p, filename in written:
        manifest.record(
            p,
            dataset=spec.name,
            stage="groundtruth" if "groundtruth" in filename else "processed",
            sha256=sha256_file(p),
            source_url=None,
            extra={
                "derived_from": raw_paths["hdf5"].name,
                "normalized": spec.normalized,
                "hdf5_distance": ds.distance_attr,
            },
        )

    return PrepareResult(
        name=spec.name,
        n_base=int(base.shape[0]),
        n_queries=int(queries.shape[0]),
        dim=int(base.shape[1]),
        space=spec.space,
        search_space=spec.search_space,
        resolved_nc=spec.resolve_nc(int(base.shape[0])),
        ships_groundtruth=True,
        bytes_on_disk=_dir_bytes(out_dir),
        details={
            "hdf5_distance": ds.distance_attr,
            "normalized_at_write": spec.normalized,
            "groundtruth_depth": int(ds.neighbors.shape[1]),
        },
    )


def _process_qa(
    spec: DatasetSpec,
    raw_paths: dict[str, Path],
    out_dir: Path,
    paths: PathsConfig,
    manifest: DataManifest,
    quiet: bool = False,
) -> PrepareResult:
    from smartann.data.chunking import build_qa_sqlite
    from smartann.pipeline.embedder import Embedder

    embedder = Embedder(
        model_name=spec.embedding_model or "thenlper/gte-small",
        cache_path=paths.data_root / "cache" / "embeddings.sqlite",
        dim=spec.dim,
    )
    try:
        stats = build_qa_sqlite(spec, raw_paths, out_dir, embedder, quiet=quiet)
    finally:
        embedder.close()

    for filename, stage in (
        ("base.npy", "processed"),
        ("queries.npy", "processed"),
        ("chunks.sqlite", "processed"),
    ):
        p = out_dir / filename
        manifest.record(
            p,
            dataset=spec.name,
            stage=stage,
            sha256=sha256_file(p),
            source_url=None,
            extra={
                "derived_from": sorted(x.name for x in raw_paths.values()),
                "embedding_model": spec.embedding_model,
                "n_documents": stats.n_documents,
                "n_chunks": stats.n_chunks,
            },
        )

    return PrepareResult(
        name=spec.name,
        n_base=stats.n_chunks,
        n_queries=stats.n_queries,
        dim=spec.dim,
        space=spec.space,
        search_space=spec.search_space,
        resolved_nc=spec.resolve_nc(stats.n_chunks),
        ships_groundtruth=False,
        bytes_on_disk=_dir_bytes(out_dir),
        details={
            "n_documents": stats.n_documents,
            "n_chunks": stats.n_chunks,
            "n_train_contexts": stats.n_train_contexts,
            "n_dev_contexts": stats.n_dev_contexts,
            "n_shared_contexts": stats.n_shared_contexts,
            "mean_sentences_per_doc": round(stats.mean_sentences_per_doc, 2),
            "mean_chunk_chars": round(stats.mean_chunk_chars, 1),
        },
    )


def prepare_dataset(
    name: str,
    paths: PathsConfig | None = None,
    record: bool = True,
    allow_mirror: bool = False,
    force: bool = False,
    quiet: bool = False,
) -> PrepareResult:
    """Download and process one dataset end to end."""
    paths = paths or resolve_paths()
    spec = spec_for(name)
    manifest = DataManifest(paths.data_root)

    out_dir = _processed_dir(paths, name)
    if force and out_dir.exists():
        for p in out_dir.glob("*"):
            if p.is_file():
                p.unlink()

    raw_paths = download_artifacts(
        spec, paths, manifest, record=record, allow_mirror=allow_mirror, quiet=quiet
    )

    if spec.kind == "ann":
        result = _process_ann(spec, raw_paths, out_dir, manifest, quiet=quiet)
    else:
        result = _process_qa(spec, raw_paths, out_dir, paths, manifest, quiet=quiet)

    manifest.write()
    return result


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
def status_for(name: str, paths: PathsConfig | None = None) -> DatasetStatus:
    paths = paths or resolve_paths()
    spec = spec_for(name)
    raw = _raw_dir(paths, name)
    processed = _processed_dir(paths, name)

    downloaded = all((raw / a.filename).exists() for a in spec.artifacts)
    verified = downloaded and all(
        a.sha256 is not None and sha256_file(raw / a.filename) == a.sha256 for a in spec.artifacts
    )

    base_p = processed / "base.npy"
    queries_p = processed / "queries.npy"
    gt_p = processed / "groundtruth_top100.npy"
    is_processed = base_p.exists() and queries_p.exists()

    n_base = n_queries = None
    resolved_nc = None
    if is_processed:
        base = np.load(base_p, mmap_mode="r")
        queries = np.load(queries_p, mmap_mode="r")
        n_base, n_queries = int(base.shape[0]), int(queries.shape[0])
        resolved_nc = spec.resolve_nc(n_base)

    return DatasetStatus(
        name=name,
        kind=spec.kind,
        downloaded=downloaded,
        verified=verified,
        processed=is_processed,
        ground_truthed=gt_p.exists(),
        n_base=n_base,
        n_queries=n_queries,
        dim=spec.dim,
        space=spec.space,
        search_space=spec.search_space,
        resolved_nc=resolved_nc,
        bytes_on_disk=_dir_bytes(raw) + _dir_bytes(processed),
    )


def status_table(paths: PathsConfig | None = None) -> list[DatasetStatus]:
    paths = paths or resolve_paths()
    return [status_for(name, paths) for name in sorted(load_registry())]

"""The dataset registry: the single source of truth for every dataset fact.

The distance metric in particular lives here and nowhere else. Two independent
declarations of the metric -- one here, one in the index config -- is the bug
that silently halves recall and costs a week to find, so Phase 2's index config
*reads* the space from this registry and asserts a match at build time rather
than declaring its own (ADR-0010).

Immutable facts (dimension, metric, URLs, expected counts) are code. Observed
facts (SHA-256 checksums, byte counts, download timestamps) are recorded into
``configs/datasets.yaml`` on first successful download and verified on every run
thereafter -- we do not have the checksums up front and neither does anyone
else, so they are recorded rather than invented.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import yaml

__all__ = [
    "CHECKSUM_CONFIG",
    "DatasetArtifact",
    "DatasetSpec",
    "Kind",
    "Space",
    "load_registry",
    "record_checksum",
    "spec_for",
]

Kind = Literal["ann", "qa"]
Space = Literal["l2", "ip", "cosine"]

CHECKSUM_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "datasets.yaml"

# ANN-Benchmarks HDF5 `distance` attribute -> our metric vocabulary.
HDF5_DISTANCE_TO_SPACE: dict[str, Space] = {
    "euclidean": "l2",
    "angular": "cosine",
    "dot": "ip",
}


@dataclass(frozen=True)
class DatasetArtifact:
    """One downloadable file belonging to a dataset.

    A dataset is a list of these rather than a single URL set (ADR-0013):
    SQuAD is two files, HotpotQA would be two more, and the single-file shape
    would force a refactor at exactly the point where adding a dataset is
    supposed to be a config entry.
    """

    key: str  # "hdf5", "train", "dev"
    filename: str  # name under data/raw/<dataset>/
    urls: tuple[str, ...]  # primary first; the rest are mirrors, never used silently
    sha256: str | None = None  # recorded on first download, then verified
    n_bytes: int | None = None
    recorded_at: str | None = None
    resolved_url: str | None = None

    @property
    def primary_url(self) -> str:
        return self.urls[0]


@dataclass(frozen=True)
class DatasetSpec:
    """Everything the pipeline needs to know about a dataset."""

    name: str  # "sift1m", "nytimes", "squad"
    kind: Kind
    dim: int
    space: Space  # the dataset's TRUE metric
    normalized: bool  # vectors are unit-norm on disk
    expected_n_base: int | None  # None for QA -- depends on chunking
    expected_n_queries: int
    artifacts: tuple[DatasetArtifact, ...]
    ships_groundtruth: bool
    default_nc: int | Literal["auto"]
    notes: str
    embedding_model: str | None = None  # QA only
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def search_space(self) -> Space:
        """The metric the index is actually built with.

        Derived, never stored. On unit-norm vectors cosine similarity and inner
        product induce the same ordering, so a normalised cosine dataset is
        searched as ``ip`` -- and the shipped ``neighbors`` arrays stay valid
        ground truth because normalisation preserves cosine ordering exactly.
        """
        if self.normalized:
            return "ip"
        return self.space

    @property
    def source_urls(self) -> list[str]:
        """Every URL this dataset can come from, primary first."""
        return [u for a in self.artifacts for u in a.urls]

    @property
    def sha256(self) -> str | None:
        """Recorded checksum, for single-artifact datasets.

        Multi-artifact datasets have one checksum per artifact; asking for a
        single one is a category error, so it returns None rather than an
        arbitrary member.
        """
        if len(self.artifacts) != 1:
            return None
        return self.artifacts[0].sha256

    def artifact(self, key: str) -> DatasetArtifact:
        for a in self.artifacts:
            if a.key == key:
                return a
        known = ", ".join(a.key for a in self.artifacts)
        raise KeyError(f"{self.name} has no artifact {key!r}; known: {known}")

    def resolve_nc(self, n_base: int) -> int:
        """Resolve ``default_nc`` against the actual base-vector count.

        ``"auto"`` means ``round(sqrt(N))``. The resolved value is recorded in
        ``index_meta.json`` and the run manifest and is never recomputed at
        search time -- a cluster count that drifts between build and search
        would silently corrupt every result.
        """
        if self.default_nc == "auto":
            if n_base < 1:
                raise ValueError(f"cannot resolve nc for {self.name}: n_base={n_base}")
            return int(round(math.sqrt(n_base)))
        return int(self.default_nc)

    def assert_space_matches(self, hdf5_distance: str) -> None:
        """Check an HDF5 ``distance`` attribute against the registered metric."""
        mapped = HDF5_DISTANCE_TO_SPACE.get(hdf5_distance)
        if mapped is None:
            raise ValueError(
                f"{self.name}: unrecognised HDF5 distance attribute {hdf5_distance!r}; "
                f"known: {', '.join(sorted(HDF5_DISTANCE_TO_SPACE))}"
            )
        if mapped != self.space:
            raise ValueError(
                f"{self.name}: metric mismatch. The HDF5 file declares "
                f"distance={hdf5_distance!r} (-> {mapped!r}) but the registry says "
                f"space={self.space!r}. One of them is wrong; refusing to guess."
            )


# ---------------------------------------------------------------------------
# The registry itself
# ---------------------------------------------------------------------------
_ANN_BENCH = "https://ann-benchmarks.com"

_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        name="sift1m",
        kind="ann",
        dim=128,
        space="l2",
        normalized=False,
        expected_n_base=1_000_000,
        expected_n_queries=10_000,
        artifacts=(
            DatasetArtifact(
                key="hdf5",
                filename="sift-128-euclidean.hdf5",
                urls=(
                    f"{_ANN_BENCH}/sift-128-euclidean.hdf5",
                    "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
                ),
            ),
        ),
        ships_groundtruth=True,
        default_nc=2000,
        notes=(
            "Nc pinned to 2000 rather than auto (=1000) so the EcoVector build is "
            "directly comparable with MobileRAG, whose Fig. 11 sweeps 1000-4000. "
            "EX sweeps it empirically. The TEXMEX mirror is the canonical SIFT "
            "source and is read via read_fvecs/read_ivecs; it is a fallback only."
        ),
    ),
    DatasetSpec(
        name="nytimes",
        kind="ann",
        dim=256,
        space="cosine",
        normalized=True,
        expected_n_base=290_000,
        expected_n_queries=10_000,
        artifacts=(
            DatasetArtifact(
                key="hdf5",
                filename="nytimes-256-angular.hdf5",
                urls=(f"{_ANN_BENCH}/nytimes-256-angular.hdf5",),
            ),
        ),
        ships_groundtruth=True,
        default_nc="auto",
        notes=(
            "Angular/cosine metric. Vectors are L2-normalised at write time so "
            "search_space derives to 'ip', k-means becomes spherical k-means, and "
            "the shipped neighbors stay valid (normalisation preserves cosine "
            "ordering). Zero-norm rows are detected and reported, not normalised."
        ),
    ),
    DatasetSpec(
        name="squad",
        kind="qa",
        dim=384,
        space="cosine",
        normalized=True,
        expected_n_base=None,  # depends on chunking; recorded after preparation
        expected_n_queries=1_000,
        artifacts=(
            DatasetArtifact(
                key="train",
                filename="train-v1.1.json",
                urls=("https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",),
            ),
            DatasetArtifact(
                key="dev",
                filename="dev-v1.1.json",
                urls=("https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",),
            ),
        ),
        ships_groundtruth=False,
        default_nc="auto",
        notes=(
            "Corpus is the union of unique train AND dev contexts: SQuAD v1.1's dev "
            "contexts are disjoint from train, so indexing train alone while querying "
            "dev questions would give identically zero gold-document recall. "
            "gte-small is a cosine model, so embeddings are L2-normalised at write "
            "time and search_space derives to 'ip'. Chunk count diverges from "
            "MobileRAG's reported 87,599 by design -- see ADR-0009."
        ),
        embedding_model="thenlper/gte-small",
        extra={
            "chunk_window_sentences": 3,
            "chunk_overlap_sentences": 1,
            "query_sample_seed": 0,
            "query_split": "dev",
        },
    ),
)

_BY_NAME: dict[str, DatasetSpec] = {s.name: s for s in _SPECS}


# ---------------------------------------------------------------------------
# Recorded checksums
# ---------------------------------------------------------------------------
def _read_checksum_file(path: Path | None = None) -> dict[str, Any]:
    p = path or CHECKSUM_CONFIG
    if not p.exists():
        return {}
    loaded = yaml.safe_load(p.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def load_registry(checksum_path: Path | None = None) -> dict[str, DatasetSpec]:
    """The registry with recorded checksums overlaid from ``configs/datasets.yaml``."""
    recorded = _read_checksum_file(checksum_path)
    out: dict[str, DatasetSpec] = {}
    for name, spec in _BY_NAME.items():
        entries = recorded.get(name, {})
        artifacts = tuple(
            replace(a, **_recorded_fields(entries.get(a.key, {}))) for a in spec.artifacts
        )
        out[name] = replace(spec, artifacts=artifacts)
    return out


def _recorded_fields(entry: dict[str, Any]) -> dict[str, Any]:
    keys = ("sha256", "n_bytes", "recorded_at", "resolved_url")
    return {k: entry[k] for k in keys if k in entry}


def spec_for(name: str, checksum_path: Path | None = None) -> DatasetSpec:
    """Look up one dataset, with recorded checksums applied."""
    registry = load_registry(checksum_path)
    if name not in registry:
        raise KeyError(f"unknown dataset {name!r}; known: {', '.join(sorted(registry))}")
    return registry[name]


def record_checksum(
    dataset: str,
    artifact_key: str,
    sha256: str,
    n_bytes: int,
    resolved_url: str,
    checksum_path: Path | None = None,
) -> None:
    """Write an observed checksum into ``configs/datasets.yaml``.

    Refuses to overwrite a differing recorded value: once a checksum is
    committed it is a fact about the dataset, and quietly replacing it would
    destroy the only evidence that a source changed underneath us.
    """
    path = checksum_path or CHECKSUM_CONFIG
    data = _read_checksum_file(path)
    entry = data.setdefault(dataset, {}).setdefault(artifact_key, {})

    existing = entry.get("sha256")
    if existing and existing != sha256:
        raise ValueError(
            f"{dataset}/{artifact_key}: refusing to overwrite recorded checksum.\n"
            f"  recorded: {existing}\n"
            f"  observed: {sha256}\n"
            "The upstream file changed, or the download is corrupt. Investigate before "
            "editing configs/datasets.yaml by hand."
        )

    entry.update(
        {
            "sha256": sha256,
            "n_bytes": n_bytes,
            "resolved_url": resolved_url,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Recorded on first successful download, then verified on every run.\n"
        "# Do not edit by hand: a mismatch means the source changed or the file is\n"
        "# corrupt, and that is information worth keeping.\n"
        + yaml.safe_dump(data, sort_keys=True, default_flow_style=False),
        encoding="utf-8",
    )

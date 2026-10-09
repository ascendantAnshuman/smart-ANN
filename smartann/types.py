"""Shared data contracts for SMART-ANN.

These are the Section 6 contracts, reproduced verbatim in field name, order and
type. Everything else in the package depends on this module, so it deliberately
has **no third-party imports**: it sits on the hot query path (a ``QueryRecord``
is constructed for every query) and must not pay a pydantic validation cost
there. Validation and YAML handling live in :mod:`smartann.config`, which
validates *these* dataclasses via ``pydantic.TypeAdapter`` rather than
redeclaring their fields.

Config objects are frozen: a decision, once made by a policy for a query, is
immutable and safe to store in a ``QueryRecord``. ``QueryRecord`` itself is
mutable because the pipeline fills it in stage by stage.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "AnnConfig",
    "Constraints",
    "ControllerState",
    "DeviceState",
    "PerfHistory",
    "PipelineConfig",
    "QueryFeatures",
    "QueryRecord",
    "ScrConfig",
]


@dataclass(frozen=True)
class AnnConfig:
    """Per-query ANN search effort. Every knob must take effect without a rebuild."""

    nprobe: int  # clusters to probe
    ef_centroid: int  # ef for the centroid graph search
    ef_list: int  # ef for per-cluster graph search
    k: int  # documents returned
    per_cluster_k: int  # neighbours requested FROM EACH probed cluster before merging


@dataclass(frozen=True)
class ScrConfig:
    """Per-query Selective Content Reduction aggressiveness."""

    sliding_window_size: int  # sentences per window
    overlap_size: int
    context_extension_size: int
    top_windows: int  # windows kept per document
    similarity_threshold: float  # tau; drop windows below this. 0.0 = keep top_windows only
    enabled: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    """The complete decision a policy makes for one query."""

    ann: AnnConfig
    scr: ScrConfig


@dataclass(frozen=True)
class DeviceState:
    """A telemetry snapshot. Fields a platform cannot measure are reported as NaN
    by the monitor and must never be silently back-filled from the cost model."""

    timestamp: float
    temperature_c: float
    temperature_slope_c_per_s: float
    freq_ratio: float  # current clock / max clock, in (0, 1]
    throttled: bool
    cpu_util: float  # 0..1
    mem_available_mb: float
    mem_pressure: float  # 0..1
    battery_pct: float
    charging: bool
    discharge_rate_ma: float


@dataclass(frozen=True)
class QueryFeatures:
    """Cheap signals available BEFORE the expensive per-cluster search."""

    centroid_top1_dist: float
    # d2 - d1, a difference and not the ratio this comment used to claim: a ratio
    # divides by a distance that can be zero or negative for duplicate vectors
    # (see EcoVectorIndex.probe). A small gap means the query sits between
    # clusters and is harder; Phase 9 measures recall 0.278 at nprobe=1 in the
    # smallest-gap fifth against 0.978 in the largest.
    centroid_dist_gap: float
    centroid_dist_entropy: float
    query_norm: float
    query_token_len: int


@dataclass(frozen=True)
class PerfHistory:
    """Rolling-window performance summary fed to the controller."""

    ewma_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    ewma_energy_mj: float
    slo_violation_rate: float  # over the sliding window
    queries_in_window: int
    interarrival_ewma_s: float


@dataclass(frozen=True)
class ControllerState:
    """The complete input to ``Policy.decide``."""

    device: DeviceState
    query: QueryFeatures
    history: PerfHistory


@dataclass(frozen=True)
class Constraints:
    """Externally imposed service objectives for a run."""

    latency_target_ms: float  # SLO; soft
    latency_hard_ms: float  # p99 bound; hard
    energy_budget_mj: float  # per query
    min_recall: float | None  # e.g. 0.90, or None if unconstrained


@dataclass
class QueryRecord:
    """One row of queries.parquet. This is the unit of all analysis.

    Append-only: every experiment writes exactly one row per query, and every
    plot and summary statistic is derived from this and nowhere else. If a
    metric cannot be computed from these fields, add a field here -- do not
    compute it in the plotting code.
    """

    query_id: str
    run_id: str
    t_wall: float
    # provenance: which dataset, which index, which build. Without these, three
    # datasets or three index variants write indistinguishable rows into one
    # parquet and every cross-run aggregate silently averages them together.
    dataset: str
    index_variant: str  # "ecovector" | "flat_hnsw" | "ivf_disk"
    index_config_hash: str
    cache_state: str  # "cold" | "warm" | "mixed"
    # decisions
    config: PipelineConfig
    policy_name: str
    controller_overhead_us: float
    # retrieval outcome
    retrieved_ids: list[int]
    retrieved_dists: list[float]  # in the index's own convention; needed for score-equality recall
    recall_at_k: float | None  # score-equality (tie-aware); None when ground truth unavailable
    recall_at_k_index: float | None  # index-overlap; the gap between the two is the tie rate
    recall_proxy: float
    # timing breakdown (ms)
    t_embed: float
    t_centroid: float
    t_cluster_load: float
    t_cluster_search: float
    t_merge: float  # merging per-cluster results into the final top-k. There is no
    # rerank pass: hnswlib returns true distances to the candidates it finds, so
    # the merge already yields the exact best-k among everything examined.
    t_scr: float
    t_generate: float
    t_total: float
    # cost. The split is not cosmetic: flat clusters count their comparisons
    # exactly, graph clusters can only be modelled as ef*M, and a cost model
    # regressed on the sum would be fitted partly to its own assumption.
    n_distance_computations: int  # == n_dist_exact + n_dist_modeled
    n_dist_exact: int  # flat-cluster and centroid-scan comparisons, actually counted
    n_dist_modeled: int  # graph-traversal comparisons, estimated as ef*M
    n_flat_clusters_scanned: int
    n_cluster_loads: int
    bytes_read: int
    energy_mj_model: float
    energy_mj_measured: float | None
    peak_rss_mb: float
    anon_rss_mb: float  # anonymous (non-file-backed) resident set: the index's true cost
    # device snapshot at query start
    device: DeviceState
    # generation (when the full RAG path is enabled)
    context_tokens: int | None
    answer: str | None
    em: float | None
    f1: float | None
    # set by the exhaustive-equivalence harness only; never by a normal run
    validation_mode: bool = False

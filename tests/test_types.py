"""Phase 0 acceptance tests: the Section 6 contracts hold their shape.

These are deliberately picky about field names and immutability. Every module
in the package depends on this surface, so drift here is expensive.
"""

from __future__ import annotations

import dataclasses

import pytest

from smartann.types import (
    AnnConfig,
    Constraints,
    ControllerState,
    DeviceState,
    PerfHistory,
    PipelineConfig,
    QueryFeatures,
    QueryRecord,
    ScrConfig,
)

FROZEN = [
    AnnConfig,
    ScrConfig,
    PipelineConfig,
    DeviceState,
    QueryFeatures,
    PerfHistory,
    ControllerState,
    Constraints,
]

EXPECTED_FIELDS = {
    AnnConfig: ["nprobe", "ef_centroid", "ef_list", "k", "per_cluster_k"],
    ScrConfig: [
        "sliding_window_size",
        "overlap_size",
        "context_extension_size",
        "top_windows",
        "similarity_threshold",
        "enabled",
    ],
    PipelineConfig: ["ann", "scr"],
    DeviceState: [
        "timestamp",
        "temperature_c",
        "temperature_slope_c_per_s",
        "freq_ratio",
        "throttled",
        "cpu_util",
        "mem_available_mb",
        "mem_pressure",
        "battery_pct",
        "charging",
        "discharge_rate_ma",
    ],
    QueryFeatures: [
        "centroid_top1_dist",
        "centroid_dist_gap",
        "centroid_dist_entropy",
        "query_norm",
        "query_token_len",
    ],
    PerfHistory: [
        "ewma_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "ewma_energy_mj",
        "slo_violation_rate",
        "queries_in_window",
        "interarrival_ewma_s",
    ],
    ControllerState: ["device", "query", "history"],
    Constraints: ["latency_target_ms", "latency_hard_ms", "energy_budget_mj", "min_recall"],
}


@pytest.mark.parametrize("cls", list(EXPECTED_FIELDS))
def test_field_names_and_order(cls: type) -> None:
    """Field names *and* order must match Section 6 exactly."""
    actual = [f.name for f in dataclasses.fields(cls)]
    assert actual == EXPECTED_FIELDS[cls]


@pytest.mark.parametrize("cls", FROZEN)
def test_config_types_are_frozen(cls: type) -> None:
    """A decision, once made for a query, must be immutable."""
    assert cls.__dataclass_params__.frozen is True  # type: ignore[attr-defined]


def test_query_record_is_mutable() -> None:
    """QueryRecord is filled in stage by stage, so it must not be frozen."""
    assert QueryRecord.__dataclass_params__.frozen is False  # type: ignore[attr-defined]


def test_frozen_config_rejects_mutation() -> None:
    cfg = AnnConfig(nprobe=4, ef_centroid=64, ef_list=64, k=5, per_cluster_k=20)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.nprobe = 8  # type: ignore[misc]


def test_frozen_configs_are_hashable_and_usable_as_dict_keys() -> None:
    """Profiling tables key on configs, so they must be usable as dict keys."""
    a = AnnConfig(nprobe=4, ef_centroid=64, ef_list=64, k=5, per_cluster_k=20)
    b = AnnConfig(nprobe=4, ef_centroid=64, ef_list=64, k=5, per_cluster_k=20)
    assert a == b
    assert {a: "x"}[b] == "x"


def test_scr_enabled_defaults_true() -> None:
    cfg = ScrConfig(
        sliding_window_size=3,
        overlap_size=2,
        context_extension_size=1,
        top_windows=1,
        similarity_threshold=0.0,
    )
    assert cfg.enabled is True


def test_optional_fields_accept_none() -> None:
    """Fields that mean 'not measurable here' must accept None, never a stand-in."""
    c = Constraints(
        latency_target_ms=100.0, latency_hard_ms=250.0, energy_budget_mj=1.0, min_recall=None
    )
    assert c.min_recall is None


def test_types_module_has_no_third_party_imports() -> None:
    """types.py sits on the hot query path and must stay dependency-free."""
    import pathlib

    import smartann.types as t

    source = pathlib.Path(t.__file__).read_text(encoding="utf-8")
    forbidden = ("import numpy", "import pydantic", "from pydantic", "import pandas", "import yaml")
    for token in forbidden:
        assert token not in source, f"types.py must not import third-party code: {token}"


# ---------------------------------------------------------------------------
# Phase 2 additions
# ---------------------------------------------------------------------------
def test_query_record_carries_its_own_provenance() -> None:
    """Three datasets or three index variants must not write indistinguishable rows."""
    names = {f.name for f in dataclasses.fields(QueryRecord)}
    assert {"dataset", "index_variant", "index_config_hash", "cache_state"} <= names


def test_query_record_splits_exact_from_modelled_distance_counts() -> None:
    """A cost model regressed on the sum would be fitted partly to its own assumption."""
    names = [f.name for f in dataclasses.fields(QueryRecord)]
    assert "n_dist_exact" in names and "n_dist_modeled" in names
    assert "n_distance_computations_is_modeled" not in names, (
        "the boolean was replaced by the split"
    )


def test_query_record_has_no_rerank_stage() -> None:
    """hnswlib returns true distances to what it finds, so the merge IS the rerank."""
    names = [f.name for f in dataclasses.fields(QueryRecord)]
    assert "t_merge" in names
    assert "t_rerank" not in names


def test_validation_mode_defaults_off() -> None:
    """Exhaustive equivalence must be unreachable from a normal run."""
    field = next(f for f in dataclasses.fields(QueryRecord) if f.name == "validation_mode")
    assert field.default is False

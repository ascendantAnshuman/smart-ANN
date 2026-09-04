"""Telemetry: never block the query path, never claim a measurement we cannot make."""

from __future__ import annotations

import json
import math
import time

import pytest

from smartann.telemetry import (
    Capability,
    CapabilityTable,
    DeviceMonitor,
    assert_manifest_consistent,
    unavailable_state,
)
from smartann.types import DeviceState


class FakeMonitor(DeviceMonitor):
    def __init__(self, hz: float = 50.0, fail_after: int | None = None) -> None:
        super().__init__(hz=hz)
        self.calls = 0
        self.fail_after = fail_after

    def sample(self) -> DeviceState:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("sensor went away")
        base = unavailable_state()
        return DeviceState(**{**base.__dict__, "cpu_util": float(self.calls)})

    def capabilities(self) -> CapabilityTable:
        return CapabilityTable([Capability("cpu_util", True, "fake")])


def test_unavailable_state_is_nan_not_zero() -> None:
    """A modelled stand-in would be indistinguishable from a measurement."""
    s = unavailable_state()
    for field in ("temperature_c", "cpu_util", "battery_pct", "discharge_rate_ma"):
        assert math.isnan(getattr(s, field)), f"{field} must be NaN, not a stand-in"


def test_latest_is_valid_before_any_background_sample() -> None:
    m = FakeMonitor()
    with m:
        assert m.latest().cpu_util == 1.0  # the synchronous sample taken by start()


def test_background_sampling_advances_the_value() -> None:
    m = FakeMonitor(hz=50.0)
    with m:
        first = m.latest().cpu_util
        time.sleep(0.25)
        assert m.latest().cpu_util > first
        assert m.n_samples > 0


def test_reading_never_blocks_on_a_slow_sensor() -> None:
    """The query path must not pay for telemetry."""

    class SlowMonitor(FakeMonitor):
        def sample(self) -> DeviceState:
            time.sleep(0.05)
            return super().sample()

    m = SlowMonitor(hz=20.0)
    with m:
        t0 = time.perf_counter()
        for _ in range(1000):
            m.latest()
        elapsed = time.perf_counter() - t0
    assert elapsed < 0.05, f"1000 reads took {elapsed:.3f}s; reads must not wait on a sample"


def test_a_failing_sensor_does_not_kill_the_monitor() -> None:
    """A telemetry failure must not take the run down with it."""
    m = FakeMonitor(hz=50.0, fail_after=2)
    with m:
        time.sleep(0.2)
        last = m.latest()
    assert math.isfinite(last.cpu_util), "the previous good value must stand"


def test_capability_table_lists_what_is_missing() -> None:
    t = CapabilityTable(
        [
            Capability("cpu_util", True, "psutil"),
            Capability("temperature_c", False, "unavailable", "no thermal zone"),
        ]
    )
    assert t.is_available("cpu_util")
    assert not t.is_available("temperature_c")
    assert t.unavailable_fields() == ["temperature_c"]
    assert "no thermal zone" in t.render()


# ---------------------------------------------------------------------------
# the ADR-0020 gate
# ---------------------------------------------------------------------------
def _table() -> CapabilityTable:
    return CapabilityTable(
        [
            Capability("cpu_util", True, "psutil"),
            Capability("energy_mj_measured", False, "unavailable", "no RAPL on Windows"),
        ]
    )


def test_manifest_may_not_claim_an_unavailable_measurement() -> None:
    """The check that makes the capability table a gate rather than a log."""
    manifest = {"capabilities": {}, "measured": {"energy_mj_measured": 12.5}}
    with pytest.raises(ValueError, match="capability table records it as unavailable"):
        assert_manifest_consistent(manifest, _table())


def test_manifest_without_a_capability_table_is_rejected() -> None:
    with pytest.raises(ValueError, match="no capability table"):
        assert_manifest_consistent({"measured": {}}, _table())


@pytest.mark.parametrize("value", [None, float("nan")])
def test_absent_or_nan_measurements_are_accepted(value: float | None) -> None:
    manifest = {"capabilities": {}, "measured": {"energy_mj_measured": value}}
    assert_manifest_consistent(manifest, _table())


def test_available_fields_may_carry_real_values() -> None:
    manifest = {"capabilities": {}, "measured": {"cpu_util": 0.42}}
    assert_manifest_consistent(manifest, _table())


def test_windows_monitor_declares_temperature_and_energy_unavailable() -> None:
    """ADR-0007/0008: this machine is not a source of thermal or energy truth."""
    from smartann.telemetry.windows import WindowsMonitor

    caps = WindowsMonitor().capabilities()
    assert not caps.is_available("temperature_c")
    assert not caps.is_available("energy_mj_measured")
    assert caps.is_available("cpu_util")


def test_windows_monitor_reports_nan_for_what_it_cannot_read() -> None:
    from smartann.telemetry.windows import WindowsMonitor

    s = WindowsMonitor().sample()
    assert math.isnan(s.temperature_c)
    assert math.isnan(s.discharge_rate_ma)
    assert 0.0 <= s.cpu_util <= 1.0
    assert s.mem_available_mb > 0


# ---------------------------------------------------------------------------
# run writer
# ---------------------------------------------------------------------------
def _record(qid: str) -> object:
    from smartann.types import AnnConfig, PipelineConfig, QueryRecord, ScrConfig

    return QueryRecord(
        query_id=qid,
        run_id="r1",
        t_wall=1.0,
        dataset="sift1m",
        index_variant="ecovector",
        index_config_hash="abc",
        cache_state="warm",
        config=PipelineConfig(
            ann=AnnConfig(nprobe=8, ef_centroid=64, ef_list=64, k=10, per_cluster_k=32),
            scr=ScrConfig(3, 1, 1, 2, 0.0),
        ),
        policy_name="static",
        controller_overhead_us=1.5,
        retrieved_ids=[1, 2, 3],
        retrieved_dists=[0.1, 0.2, 0.3],
        recall_at_k=0.9,
        recall_at_k_index=0.9,
        recall_proxy=0.0,
        t_embed=0.0,
        t_centroid=0.1,
        t_cluster_load=1.0,
        t_cluster_search=0.5,
        t_merge=0.01,
        t_scr=0.0,
        t_generate=0.0,
        t_total=2.0,
        n_distance_computations=100,
        n_dist_exact=100,
        n_dist_modeled=0,
        n_flat_clusters_scanned=8,
        n_cluster_loads=8,
        bytes_read=1024,
        energy_mj_model=0.0,
        energy_mj_measured=None,
        peak_rss_mb=100.0,
        anon_rss_mb=10.0,
        device=unavailable_state(),
        context_tokens=None,
        answer=None,
        em=None,
        f1=None,
    )


def test_writer_round_trips_through_parquet(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Every figure comes from the parquet, so the parquet must be faithful."""
    import pandas as pd

    from smartann.bench.writer import RunWriter

    w = RunWriter(tmp_path, "r1", "E0", _table())
    for i in range(5):
        w.append(_record(f"q{i}"))
    w.event("burners_start", workers=4)
    parquet, manifest_path = w.write()

    df = pd.read_parquet(parquet)
    assert len(df) == 5
    # nested configs are flattened, not left as structs to unpack at plot time
    assert df["cfg_ann_nprobe"].iloc[0] == 8
    assert df["cfg_scr_top_windows"].iloc[0] == 2
    assert "dev_temperature_c" in df.columns
    assert df["retrieved_ids"].iloc[0].tolist() == [1, 2, 3]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["n_queries"] == 5
    assert manifest["events"][0]["name"] == "burners_start"
    assert manifest["capabilities"]["energy_mj_measured"]["available"] is False


def test_writer_refuses_an_empty_run(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from smartann.bench.writer import RunWriter

    with pytest.raises(ValueError, match="no records"):
        RunWriter(tmp_path, "r1", "E0", _table()).write()


def test_writer_manifest_passes_the_capability_gate(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A run cannot record a measured energy value on a machine without one."""
    from smartann.bench.writer import RunWriter

    w = RunWriter(tmp_path, "r1", "E0", _table())
    w.append(_record("q0"))
    with pytest.raises(ValueError, match="unavailable"):
        w.write(extra={"measured": {"energy_mj_measured": 42.0}})

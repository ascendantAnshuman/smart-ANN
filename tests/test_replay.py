"""Replaying a measured trace must stay a measurement.

Three properties carry this module.

**The arithmetic matches the device's own.** The slope is recomputed here rather
than read, so the recomputation is checked against the `slope` field the monitor
recorded on the handset. If those diverge, the replay is reporting a rate of
change the device never saw.

**Absence survives the round trip.** Fields the trace does not carry must arrive
as NaN, because every policy in this project is written to decline on a NaN and
would read a zero as a cold, empty, healthy device.

**The sensor choice is load-bearing.** Skin and hottest-zone are different
measurements of the same run, and the policy's behaviour under each is the C5
result. A loader that quietly picked one would erase the finding.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from smartann.telemetry.replay import (
    SENSOR_HOTTEST,
    SENSOR_SKIN,
    ReplayFeed,
    available_traces,
    load_trace,
    what_if_state,
)

REAL_TRACE = Path("results/device/c5_telemetry.jsonl")


def write_trace(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def synthetic(tmp_path: Path) -> Path:
    rows = [
        {
            "t": 0.0,
            "skin_c": 30.0,
            "hottest_c": 50.0,
            "pct": 80,
            "status": "DISCHARGING",
            "current_ua": -300000,
            "slope": None,
        },
        {
            "t": 10.0,
            "skin_c": 31.0,
            "hottest_c": 60.0,
            "pct": 79,
            "status": "DISCHARGING",
            "current_ua": -310000,
            "slope": 0.1,
        },
        {
            "t": 20.0,
            "skin_c": 32.0,
            "hottest_c": 70.0,
            "pct": 78,
            "status": "CHARGING",
            "current_ua": 200000,
            "slope": 0.1,
        },
    ]
    return write_trace(tmp_path / "syn_telemetry.jsonl", rows)


def test_each_sensor_reports_its_own_temperature(tmp_path: Path) -> None:
    p = synthetic(tmp_path)
    assert load_trace(p, SENSOR_SKIN).state_at(2).temperature_c == 32.0
    assert load_trace(p, SENSOR_HOTTEST).state_at(2).temperature_c == 70.0


def test_the_slope_is_the_chosen_sensors_own(tmp_path: Path) -> None:
    """Handing back the skin's slope for the hottest zone would mislabel a rate."""
    p = synthetic(tmp_path)
    skin = load_trace(p, SENSOR_SKIN)
    hot = load_trace(p, SENSOR_HOTTEST)
    assert skin.state_at(1).temperature_slope_c_per_s == pytest.approx(0.1)
    assert hot.state_at(1).temperature_slope_c_per_s == pytest.approx(1.0)


def test_the_first_sample_has_no_slope(tmp_path: Path) -> None:
    """Nothing to difference against. Absent, not zero."""
    tr = load_trace(synthetic(tmp_path), SENSOR_SKIN)
    assert math.isnan(tr.state_at(0).temperature_slope_c_per_s)


def test_unrecorded_fields_arrive_as_nan(tmp_path: Path) -> None:
    st = load_trace(synthetic(tmp_path), SENSOR_SKIN).state_at(1)
    for field in ("freq_ratio", "cpu_util", "mem_available_mb", "mem_pressure"):
        assert math.isnan(getattr(st, field)), f"{field} was not recorded and must be NaN"


def test_charging_status_is_carried_through(tmp_path: Path) -> None:
    tr = load_trace(synthetic(tmp_path), SENSOR_SKIN)
    assert tr.state_at(1).charging is False
    assert tr.state_at(2).charging is True


def test_current_is_converted_to_milliamps(tmp_path: Path) -> None:
    st = load_trace(synthetic(tmp_path), SENSOR_SKIN).state_at(1)
    assert st.discharge_rate_ma == pytest.approx(-310.0)


def test_an_index_past_the_end_clamps(tmp_path: Path) -> None:
    """A demo that runs off the end of a trace should stop, not crash."""
    tr = load_trace(synthetic(tmp_path), SENSOR_SKIN)
    assert tr.state_at(999).temperature_c == tr.state_at(2).temperature_c


def test_a_partial_last_line_is_skipped(tmp_path: Path) -> None:
    """A run killed mid-write leaves half a JSON object behind."""
    p = tmp_path / "half_telemetry.jsonl"
    p.write_text('{"t":0,"skin_c":30,"hottest_c":40,"pct":9,"status":"X"}\n{"t":1,"skin', "utf-8")
    assert len(load_trace(p, SENSOR_SKIN)) == 1


def test_an_empty_trace_is_refused(tmp_path: Path) -> None:
    p = tmp_path / "empty_telemetry.jsonl"
    p.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no telemetry samples"):
        load_trace(p, SENSOR_SKIN)


def test_an_unknown_sensor_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sensor must be one of"):
        load_trace(synthetic(tmp_path), "cpu")


def test_the_feed_advances_and_stops_at_the_end(tmp_path: Path) -> None:
    feed = ReplayFeed(load_trace(synthetic(tmp_path), SENSOR_HOTTEST), over_queries=3)
    seen = [feed.next()[0].temperature_c for _ in range(6)]
    assert seen[0] == 50.0
    assert seen[-1] == 70.0, "a long session must settle at the end, not wrap around"


def test_the_feed_labels_every_state_with_its_sensor(tmp_path: Path) -> None:
    """A number without its thermometer cannot be checked by a reader."""
    _, label = ReplayFeed(load_trace(synthetic(tmp_path), SENSOR_HOTTEST)).next()
    assert "hottest" in label
    assert "syn" in label


def test_a_what_if_state_leaves_unstated_fields_absent() -> None:
    st = what_if_state(temperature_c=50.0)
    assert st.temperature_c == 50.0
    assert math.isnan(st.battery_pct), "a battery nobody stated is unknown, not empty"
    assert math.isnan(st.temperature_slope_c_per_s)


def test_available_traces_lists_nothing_when_there_is_no_directory(tmp_path: Path) -> None:
    assert available_traces(tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# Against the recorded handset trace, when it is present
# ---------------------------------------------------------------------------

needs_trace = pytest.mark.skipif(
    not REAL_TRACE.exists(), reason="the recorded handset trace is not on this machine"
)


@needs_trace
def test_recomputed_skin_slope_matches_what_the_monitor_recorded() -> None:
    """The replay's arithmetic must be the device's arithmetic.

    The monitor recorded `slope` for the skin sensor as it ran. Recomputing it
    from the recorded temperatures and timestamps has to land on the same
    numbers, or the replay is feeding the policy a rate of change the handset
    never reported.
    """
    tr = load_trace(REAL_TRACE, SENSOR_SKIN)
    compared = 0
    for i, sample in enumerate(tr.samples):
        recorded = sample.recorded_slope
        if not math.isfinite(recorded):
            continue
        assert tr.slopes[i] == pytest.approx(recorded, abs=1e-6)
        compared += 1
    assert compared > 10, "too few finite slopes in the trace to call this checked"


@needs_trace
def test_the_two_thermometers_disagree_on_the_recorded_run() -> None:
    """The C5 finding, pinned: if these ever agree, the result has changed."""
    skin = load_trace(REAL_TRACE, SENSOR_SKIN)
    hot = load_trace(REAL_TRACE, SENSOR_HOTTEST)
    assert hot.peak_c > skin.peak_c + 20.0, (
        "the recorded run had a 35 C skin and a 70 C hottest zone; a replay that "
        "loses that gap has stopped reproducing the measurement"
    )

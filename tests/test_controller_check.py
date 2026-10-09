"""The 11.1 stability check's arithmetic.

Everything the gate decides on is computed here: which latency a decision is
paired with, whether that number was measured or interpolated, and how a
direction reversal is counted. A bug in any of the three would let an unstable
loop pass the gate, or condemn a stable one, and the figure would look
plausible either way.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from phase11_controller_check import (
    ArmResult,
    LevelLatency,
    TraceLatency,
    load_query_trace,
    load_thermals,
    replay,
)


def _levels(spec: dict[int, list[tuple[float, float]]]) -> TraceLatency:
    return TraceLatency(
        {
            npb: LevelLatency(
                npb,
                np.array([p[0] for p in pairs], dtype=np.float64),
                np.array([p[1] for p in pairs], dtype=np.float64),
            )
            for npb, pairs in spec.items()
        }
    )


# ---------------------------------------------------------------------------
# Latency lookup
# ---------------------------------------------------------------------------
def test_level_latency_returns_the_nearest_sample_in_time() -> None:
    lv = LevelLatency(8, np.array([0.0, 10.0, 20.0]), np.array([100.0, 200.0, 300.0]))
    assert lv.at(0.0) == 100.0
    assert lv.at(4.0) == 100.0  # nearer t=0
    assert lv.at(6.0) == 200.0  # nearer t=10
    assert lv.at(10.0) == 200.0


def test_level_latency_clamps_outside_the_recorded_span() -> None:
    """Before the first sample and after the last, the nearest one is the answer.

    Not an extrapolation: it is the first or last latency the device actually
    recorded at this level, returned unchanged.
    """
    lv = LevelLatency(8, np.array([5.0, 15.0]), np.array([100.0, 200.0]))
    assert lv.at(-100.0) == 100.0
    assert lv.at(1e9) == 200.0


def test_a_measured_level_is_reported_as_measured() -> None:
    tl = _levels({1: [(0.0, 10.0)], 8: [(0.0, 80.0)]})
    ms, how = tl.at(0.0, 8)
    assert (ms, how) == (80.0, "measured")


def test_between_two_levels_the_value_is_interpolated_and_says_so() -> None:
    """The provenance string is the point: a number the device never produced
    must not be able to travel into a summary as a measurement."""
    tl = _levels({4: [(0.0, 40.0)], 8: [(0.0, 80.0)]})
    ms, how = tl.at(0.0, 6)
    assert how == "interpolated"
    assert ms == pytest.approx(60.0)  # halfway in nprobe, halfway in ms
    assert 40.0 < ms < 80.0


def test_requests_outside_the_swept_range_clamp_to_a_measured_level() -> None:
    """Nothing above the top swept level was measured, so nothing is invented."""
    tl = _levels({4: [(0.0, 40.0)], 8: [(0.0, 80.0)]})
    assert tl.at(0.0, 99) == (80.0, "measured")
    assert tl.at(0.0, 1) == (40.0, "measured")


# ---------------------------------------------------------------------------
# Reversals and amplitude
# ---------------------------------------------------------------------------
def _arm(nprobe: list[int], dt: float = 1.0) -> ArmResult:
    t = [i * dt for i in range(len(nprobe))]
    return ArmResult(0.0, "test", t, nprobe, [10.0] * len(nprobe), [0.0] * len(nprobe), 0)


def test_a_reversal_is_a_change_of_direction() -> None:
    # up, up, down, down, up  ->  directions + + - - +  ->  two reversals
    arm = _arm([1, 2, 3, 2, 1, 2]).summarise(target_ms=1000.0)
    assert arm.effort_changes == 5
    assert arm.reversals == 2


def test_holding_steady_is_not_a_direction() -> None:
    """A flat stretch must not count as agreeing with whatever preceded it, and
    must not break a run of increases into two runs."""
    arm = _arm([1, 2, 2, 2, 3]).summarise(target_ms=1000.0)
    assert arm.effort_changes == 2  # 1->2 and 2->3
    assert arm.reversals == 0


def test_a_monotone_climb_never_reverses() -> None:
    arm = _arm(list(range(1, 20))).summarise(target_ms=1000.0)
    assert arm.reversals == 0
    assert arm.reversals_per_min == 0.0


def test_reversals_are_reported_per_minute_of_trace_time() -> None:
    # 4 reversals over 60 s of trace time is 4/min, whatever the query count.
    arm = _arm([1, 2, 1, 2, 1, 2, 1], dt=10.0).summarise(target_ms=1000.0)
    assert arm.duration_s == pytest.approx(60.0)
    assert arm.reversals == 5
    assert arm.reversals_per_min == pytest.approx(5.0)


def test_a_collapsed_controller_scores_a_perfect_reversal_count() -> None:
    """The case that condemned the pre-declared criterion (ADR-0032 amendment).

    A controller pinned to its floor never changes direction, so it scores
    0.00 reversals/min -- a perfect result on a reversal-count criterion while
    being the exact pathology ADR-0032 exists to prevent. Both the floor check
    and the amplitude gate must catch it.
    """
    arm = _arm([1] * 50).summarise(target_ms=1000.0)
    assert arm.reversals_per_min == 0.0
    assert arm.stable_declared is False  # only because of the floor clause
    assert arm.nprobe_p50 == 1.0
    assert arm.stable is False


def test_a_healthy_sawtooth_passes_the_amended_gate_and_fails_the_declared_one() -> None:
    """A tight oscillation around a mid-range operating point, which is what
    AIMD does when it is working. It reverses often -- that is the mechanism --
    but travels a small part of the action range and holds its target."""
    arm = _arm([10, 11, 12, 11] * 60, dt=1.0).summarise(target_ms=1000.0)
    assert arm.reversals_per_min > 1.0  # fails the count criterion
    assert arm.stable_declared is False
    assert arm.swing_fraction_of_range < 0.5
    assert arm.stable is True  # passes the amplitude gate


def test_a_full_range_swing_fails_the_amended_gate() -> None:
    arm = _arm([1, 32] * 60, dt=1.0).summarise(target_ms=1000.0)
    assert arm.swing_fraction_of_range > 0.5
    assert arm.stable is False


def test_the_amended_gate_requires_the_delivered_p95_to_meet_the_target() -> None:
    """A loop can sit in a tight band and still miss the latency it controls to.
    That is not stability worth building a demonstration on."""
    arm = _arm([10, 11, 12, 11] * 60, dt=1.0)
    arm.latency_ms = [500.0] * len(arm.nprobe)
    arm.summarise(target_ms=100.0)
    assert arm.swing_fraction_of_range < 0.5
    assert arm.nprobe_p50 > 1
    assert arm.stable is False


def test_amplitude_separates_a_tight_sawtooth_from_a_full_range_swing() -> None:
    tight = _arm([10, 11, 12, 11, 10, 11, 12, 11] * 5).summarise(target_ms=1000.0)
    wild = _arm([1, 32, 1, 32, 1, 32, 1, 32] * 5).summarise(target_ms=1000.0)
    assert tight.swing_fraction_of_range < 0.1
    assert wild.swing_fraction_of_range > 0.9
    assert tight.max_abs_step < wild.max_abs_step


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def test_loading_groups_queries_by_effort_level(tmp_path: Path) -> None:
    rows = [
        {"t_wall": 0.0, "nprobe": 1, "t_total_ms": 5.0},
        {"t_wall": 0.1, "nprobe": 8, "t_total_ms": 40.0},
        {"t_wall": 0.2, "nprobe": 1, "t_total_ms": 6.0},
    ]
    p = tmp_path / "q.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    tl = load_query_trace(p)
    assert tl.keys == [1, 8]
    assert tl.levels[1].ms.tolist() == [5.0, 6.0]
    assert tl.p50(8) == 40.0


def test_telemetry_is_rebased_onto_the_query_clock(tmp_path: Path) -> None:
    rows = [
        {"t": 1_000_000.0, "skin_c": 30.0, "hottest_c": 50.0},
        {"t": 1_000_010.0, "skin_c": 31.0, "hottest_c": 60.0},
    ]
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    th = load_thermals(p)
    assert th.t.tolist() == [0.0, 10.0]
    assert th.hottest_at(5.0) == pytest.approx(55.0)
    # 10 C over 10 s between the bracketing samples.
    assert th.slope_at(5.0) == pytest.approx(1.0)
    # Nothing to difference against before the first sample.
    assert math.isnan(th.slope_at(0.0))


def test_a_file_with_no_samples_raises_rather_than_returning_an_empty_trace(
    tmp_path: Path,
) -> None:
    p = tmp_path / "empty.jsonl"
    p.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_thermals(p)


# ---------------------------------------------------------------------------
# The loop itself
# ---------------------------------------------------------------------------
def test_pacing_the_loop_keeps_it_off_the_floor_that_the_unpaced_loop_reaches() -> None:
    """The ADR-0032 mechanism, reproduced on a synthetic trace small enough to
    assert on. Latency rises steeply with effort, so a controller that acts
    faster than its own feedback ratchets down; one paced to the window's fill
    time does not."""
    tl = _levels(
        {
            1: [(0.0, 5.0), (1000.0, 5.0)],
            32: [(0.0, 400.0), (1000.0, 400.0)],
        }
    )
    from phase11_controller_check import Thermals

    thermals = Thermals(
        t=np.array([0.0, 1000.0]),
        skin_c=np.array([30.0, 30.0]),
        hottest_c=np.array([50.0, 50.0]),
    )
    target = 150.0
    unpaced = replay(tl, thermals, 0.0, target, 300.0, "unpaced")
    paced = replay(tl, thermals, 50 * 0.4, target, 300.0, "paced")

    assert unpaced.floor_fraction > paced.floor_fraction
    assert paced.nprobe_p50 > unpaced.nprobe_p50


def test_the_replay_advances_time_by_the_latency_it_measured() -> None:
    """A cheaper decision must buy more queries per second, as it would on the
    device. If time advanced by a fixed step, the controller's whole purpose
    would be invisible in the record."""
    cheap = _levels({1: [(0.0, 10.0), (100.0, 10.0)], 32: [(0.0, 10.0), (100.0, 10.0)]})
    dear = _levels({1: [(0.0, 100.0), (100.0, 100.0)], 32: [(0.0, 100.0), (100.0, 100.0)]})
    from phase11_controller_check import Thermals

    thermals = Thermals(
        t=np.array([0.0, 100.0]),
        skin_c=np.array([30.0, 30.0]),
        hottest_c=np.array([50.0, 50.0]),
    )
    a = replay(cheap, thermals, 1.0, 1e9, 10.0, "cheap")
    b = replay(dear, thermals, 1.0, 1e9, 10.0, "dear")
    assert len(a.nprobe) == pytest.approx(10 * len(b.nprobe), rel=0.05)

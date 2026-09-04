"""The control layer. Untested until now, which was a real gap.

ADR-0032 makes `control_period_s` load-bearing: it is the difference between a
controller that settles and one that collapses to its floor in every phase. A
knob that decides whether the system works at all should not be exercised only
by a 4-minute experiment.
"""

from __future__ import annotations

import math

import pytest

from smartann.controller import AimdConfig, AimdPolicy, Policy, StaticPolicy
from smartann.types import (
    AnnConfig,
    Constraints,
    ControllerState,
    DeviceState,
    PerfHistory,
    QueryFeatures,
    ScrConfig,
)

ANN = AnnConfig(nprobe=16, ef_centroid=128, ef_list=64, k=10, per_cluster_k=32)
SCR = ScrConfig(3, 1, 1, 2, 0.0, enabled=False)
CONSTRAINTS = Constraints(
    latency_target_ms=100.0, latency_hard_ms=300.0, energy_budget_mj=1.0, min_recall=None
)


def _state(p95: float, t: float = 0.0, in_window: int = 50) -> ControllerState:
    device = DeviceState(
        timestamp=t,
        temperature_c=math.nan,
        temperature_slope_c_per_s=math.nan,
        freq_ratio=math.nan,
        throttled=False,
        cpu_util=0.5,
        mem_available_mb=8000.0,
        mem_pressure=0.4,
        battery_pct=80.0,
        charging=False,
        discharge_rate_ma=math.nan,
    )
    history = PerfHistory(
        ewma_latency_ms=p95,
        p50_latency_ms=p95 * 0.8,
        p95_latency_ms=p95,
        p99_latency_ms=p95 * 1.1,
        ewma_energy_mj=0.0,
        slo_violation_rate=0.0,
        queries_in_window=in_window,
        interarrival_ewma_s=0.01,
    )
    return ControllerState(
        device=device, query=QueryFeatures(0.0, 0.0, 0.0, 1.0, 0), history=history
    )


def _drive(policy: AimdPolicy, p95: float, n: int, t0: float = 0.0, dt: float = 0.01) -> list[int]:
    """Run n decisions at a fixed p95 and return the nprobe chosen each time."""
    return [policy.decide(_state(p95, t=t0 + i * dt), CONSTRAINTS).ann.nprobe for i in range(n)]


# ---------------------------------------------------------------------------
# static baseline
# ---------------------------------------------------------------------------
def test_static_returns_the_same_config_regardless_of_state() -> None:
    """The thing adaptation is measured against must not itself adapt."""
    p = StaticPolicy(ANN, SCR, tuning_note="chosen by sweep")
    calm = p.decide(_state(10.0), CONSTRAINTS)
    panic = p.decide(_state(5000.0), CONSTRAINTS)
    assert calm == panic
    assert calm.ann.nprobe == 16


def test_static_carries_its_tuning_note() -> None:
    """Section 3 rule 3: a reader must be able to check the baseline was tuned."""
    p = StaticPolicy(ANN, SCR, tuning_note="largest nprobe meeting quiet p95")
    assert "largest nprobe" in p.tuning_note


# ---------------------------------------------------------------------------
# AIMD: the control law
# ---------------------------------------------------------------------------
def test_effort_decreases_when_over_target() -> None:
    p = AimdPolicy(AimdConfig(warmup_queries=0), ANN, SCR)
    before = p.effort
    _drive(p, p95=500.0, n=1)
    assert p.effort < before


def test_effort_increases_only_below_the_low_water_mark() -> None:
    """Between low_water*target and target the controller must hold, not creep."""
    cfg = AimdConfig(warmup_queries=0, low_water=0.8)
    p = AimdPolicy(cfg, ANN, SCR)
    p.effort = 0.5

    _drive(p, p95=0.9 * CONSTRAINTS.latency_target_ms, n=5)  # inside the dead band
    assert p.effort == pytest.approx(0.5), "no change is expected inside the dead band"

    # Continue the clock forward; a wall-clock pacing check must not see time
    # move backwards, which is a property of the test rather than of the policy.
    _drive(p, p95=0.5 * CONSTRAINTS.latency_target_ms, n=1, t0=1.0)  # below low water
    assert p.effort > 0.5


def test_warmup_suppresses_action() -> None:
    """Reacting to a two-sample p95 would be reacting to noise."""
    p = AimdPolicy(AimdConfig(warmup_queries=10), ANN, SCR)
    _drive(p, p95=5000.0, n=10)
    assert p.effort == 1.0, "no action until the window is meaningful"
    _drive(p, p95=5000.0, n=2)
    assert p.effort < 1.0


def test_effort_stays_within_bounds_under_sustained_pressure() -> None:
    p = AimdPolicy(AimdConfig(warmup_queries=0), ANN, SCR)
    for p95 in (5000.0, 0.0):
        _drive(p, p95=p95, n=200)
        assert 0.0 <= p.effort <= 1.0


def test_nprobe_mapping_is_monotone_and_spans_the_range() -> None:
    cfg = AimdConfig(nprobe_min=1, nprobe_max=64)
    p = AimdPolicy(cfg, ANN, SCR)
    seen = []
    for e in (0.0, 0.25, 0.5, 0.75, 1.0):
        p.effort = e
        seen.append(p._nprobe())
    assert seen == sorted(seen)
    assert seen[0] == 1 and seen[-1] == 64


def test_rejects_a_control_law_that_cannot_converge() -> None:
    with pytest.raises(ValueError, match="multiplicative_decrease"):
        AimdPolicy(AimdConfig(multiplicative_decrease=1.0), ANN, SCR)
    with pytest.raises(ValueError, match="additive_increase"):
        AimdPolicy(AimdConfig(additive_increase=0.0), ANN, SCR)
    with pytest.raises(ValueError, match="nprobe range"):
        AimdPolicy(AimdConfig(nprobe_min=64, nprobe_max=8), ANN, SCR)


# ---------------------------------------------------------------------------
# ADR-0032: pace actions no faster than the feedback delay
# ---------------------------------------------------------------------------
def test_zero_period_acts_every_query_and_ratchets() -> None:
    """The configuration that collapsed, reproduced.

    Seven consecutive multiplicative cuts take effort from 1.0 to below 0.01,
    and seven queries pass long before a 50-sample window refreshes.
    """
    p = AimdPolicy(AimdConfig(warmup_queries=0, control_period_s=0.0), ANN, SCR)
    _drive(p, p95=500.0, n=7, dt=0.001)
    assert p.effort < 0.01
    assert p._nprobe() == 1, "it reaches the floor in seven queries"


def test_a_control_period_paces_the_ratchet() -> None:
    """Same pressure, same query count, but actions paced to the feedback delay."""
    p = AimdPolicy(AimdConfig(warmup_queries=0, control_period_s=4.05), ANN, SCR)
    _drive(p, p95=500.0, n=7, dt=0.001)  # 7 queries spanning 6 ms
    assert p.effort == 0.5, "exactly one action can have fired inside 6 ms"


def test_the_period_is_wall_clock_not_query_count() -> None:
    """Query rate changes with effort, so pacing must be in time."""
    p = AimdPolicy(AimdConfig(warmup_queries=0, control_period_s=1.0), ANN, SCR)
    # 100 fast queries inside one second: one action.
    _drive(p, p95=500.0, n=100, t0=0.0, dt=0.001)
    assert p.effort == 0.5
    # A single query a full period later: a second action.
    p.decide(_state(500.0, t=1.5), CONSTRAINTS)
    assert p.effort == pytest.approx(0.25)


def test_control_period_defaults_to_the_collapsing_behaviour() -> None:
    """The default must not silently change results recorded before ADR-0032."""
    assert AimdConfig().control_period_s == 0.0


def test_reset_clears_the_pacing_clock_too() -> None:
    """A stale last-action time would suppress the first action of the next run."""
    p = AimdPolicy(AimdConfig(warmup_queries=0, control_period_s=1.0), ANN, SCR)
    _drive(p, p95=500.0, n=1, t0=1000.0)
    p.reset()
    assert p.effort == 1.0
    assert p._last_action_t is None


# ---------------------------------------------------------------------------
# overhead accounting
# ---------------------------------------------------------------------------
def test_decide_timed_reports_a_positive_overhead() -> None:
    """Hard rule 5: the controller's own cost is a measured metric."""
    p = AimdPolicy(AimdConfig(), ANN, SCR)
    d = p.decide_timed(_state(50.0), CONSTRAINTS)
    assert d.overhead_us > 0.0
    assert d.config.ann.nprobe >= 1


def test_every_policy_satisfies_the_contract() -> None:
    for policy in (StaticPolicy(ANN, SCR), AimdPolicy(AimdConfig(), ANN, SCR)):
        assert isinstance(policy, Policy)
        policy.update(object())  # must not raise
        policy.reset()
        assert policy.decide(_state(50.0), CONSTRAINTS).ann.k == 10

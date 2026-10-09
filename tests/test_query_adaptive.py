"""C6's policy must spend a measured allocation, and must refuse to invent one.

Four properties carry it.

**The budget is a ceiling, never a floor to be raised.** Difficulty may move
effort down inside the budget. A policy that could spend more than the budget
allowed on a query it judged hard would miss the latency target in the name of
recall, which is a trade nobody declared.

**A missing signal is not a middle bucket.** `probe()` can report a NaN gap.
Guessing a bucket from it would attribute an effort choice to a measurement that
was never taken -- Section 3, rule 1, in the one place on the query path where
it would be invisible.

**The allocation comes from a file.** Nothing here may hard-code that a gap of
1.2 deserves sixteen clusters. The mapping is loaded from a recorded experiment
or handed in explicitly, and a fit from another corpus is refused.

**A fit that demonstrated nothing still loads, and says so.** A negative result
must stay inspectable; what it must never do is look identical to a positive one
to a caller choosing a default.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from smartann.controller.policies.query import QueryAdaptivePolicy, QueryEffortFit
from smartann.controller.profile import ProfileEntry, ProfileTable
from smartann.types import (
    AnnConfig,
    Constraints,
    ControllerState,
    DeviceState,
    PerfHistory,
    QueryFeatures,
    ScrConfig,
)

BASE = AnnConfig(nprobe=8, ef_centroid=128, ef_list=64, k=10, per_cluster_k=32)
SCR = ScrConfig(1, 0, 0, 3, 0.0, enabled=False)


def table() -> ProfileTable:
    return ProfileTable(
        entries=(
            ProfileEntry(1, 1000, 5.0, 10.0, 0.60, 100.0),
            ProfileEntry(4, 1000, 25.0, 50.0, 0.84, 100.0),
            ProfileEntry(8, 1000, 50.0, 100.0, 0.90, 100.0),
            ProfileEntry(16, 1000, 95.0, 190.0, 0.95, 100.0),
        ),
        dataset="squad",
        measured_under="a stub",
        min_samples=100,
    )


def fit(assignment: tuple[int, ...] = (16, 8, 4, 1), delta: float = 0.01) -> QueryEffortFit:
    """Harder queries (low gap) get more effort, which is the hypothesis."""
    return QueryEffortFit(
        signal="centroid_dist_gap",
        edges=(1.1, 1.3, 1.6),
        assignment=assignment,
        budget_ms=100.0,
        dataset="squad",
        delta_recall=delta,
        cost_saving=0.1,
        n_held_out=500,
    )


def state(gap: float) -> ControllerState:
    return ControllerState(
        device=DeviceState(
            timestamp=0.0,
            temperature_c=math.nan,
            temperature_slope_c_per_s=math.nan,
            freq_ratio=math.nan,
            throttled=False,
            cpu_util=math.nan,
            mem_available_mb=math.nan,
            mem_pressure=math.nan,
            battery_pct=math.nan,
            charging=False,
            discharge_rate_ma=math.nan,
        ),
        query=QueryFeatures(
            centroid_top1_dist=0.1,
            centroid_dist_gap=gap,
            centroid_dist_entropy=1.0,
            query_norm=1.0,
            query_token_len=8,
        ),
        history=PerfHistory(
            ewma_latency_ms=math.nan,
            p50_latency_ms=math.nan,
            p95_latency_ms=math.nan,
            p99_latency_ms=math.nan,
            ewma_energy_mj=math.nan,
            slo_violation_rate=0.0,
            queries_in_window=0,
            interarrival_ewma_s=math.nan,
        ),
    )


def cons(budget_ms: float = 220.0) -> Constraints:
    return Constraints(
        latency_target_ms=budget_ms,
        latency_hard_ms=budget_ms,
        energy_budget_mj=float("inf"),
        min_recall=None,
    )


# ---------------------------------------------------------------------------
# The mapping
# ---------------------------------------------------------------------------


def test_each_bucket_gets_its_fitted_level() -> None:
    f = fit()
    assert f.level_for(1.0) == 16, "below the first edge is the hardest bucket"
    assert f.level_for(1.2) == 8
    assert f.level_for(1.4) == 4
    assert f.level_for(9.9) == 1, "above the last edge is the easiest bucket"


def test_a_value_exactly_on_an_edge_falls_in_the_upper_bucket() -> None:
    """Pinned so a boundary query cannot change bucket between runs."""
    assert fit().level_for(1.1) == 8
    assert fit().level_for(1.3) == 4


def test_an_absent_signal_selects_no_bucket() -> None:
    """NaN means the probe did not report a gap, not that the query was average."""
    assert fit().level_for(float("nan")) is None
    assert fit().level_for(float("inf")) is None


def test_an_assignment_that_does_not_match_the_edges_is_refused() -> None:
    with pytest.raises(ValueError, match="buckets"):
        QueryEffortFit(
            signal="centroid_dist_gap",
            edges=(1.1, 1.3),
            assignment=(8, 4),  # 2 edges define 3 buckets, not 2
            budget_ms=100.0,
            dataset="squad",
            delta_recall=0.0,
            cost_saving=0.0,
            n_held_out=1,
        )


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------


def test_an_easy_query_buys_less_effort_than_a_hard_one() -> None:
    """The whole hypothesis, on two queries."""
    p = QueryAdaptivePolicy(table(), BASE, SCR, fit())
    hard = p.decide(state(gap=1.0), cons()).ann.nprobe
    easy = p.decide(state(gap=9.0), cons()).ann.nprobe
    assert hard > easy


def test_the_budget_is_a_ceiling_difficulty_cannot_raise_effort() -> None:
    """A hard query must not buy a level the latency budget refused."""
    p = QueryAdaptivePolicy(table(), BASE, SCR, fit())
    # 60 ms with 0.9 headroom admits p95=50 -> nprobe 4, though the fit wants 16.
    chosen = p.decide(state(gap=1.0), cons(budget_ms=60.0)).ann.nprobe
    assert chosen == 4


def test_a_query_with_no_signal_gets_the_budgets_level_and_no_reason() -> None:
    p = QueryAdaptivePolicy(table(), BASE, SCR, fit())
    chosen = p.decide(state(gap=float("nan")), cons()).ann.nprobe
    assert chosen == 16, "the budget's own level, unmodified"
    assert p.reasons() == [], "no signal was read, so none may be claimed"


def test_the_reason_names_the_signal_and_its_value() -> None:
    """A decision a reader cannot check is not an explanation."""
    p = QueryAdaptivePolicy(table(), BASE, SCR, fit())
    p.decide(state(gap=1.45), cons())
    reason = p.reasons()[0]
    assert "centroid_dist_gap" in reason
    assert "1.45" in reason


def test_reset_clears_the_reasons() -> None:
    p = QueryAdaptivePolicy(table(), BASE, SCR, fit())
    p.decide(state(gap=1.0), cons())
    p.reset()
    assert p.reasons() == []


def test_it_does_not_claim_to_honour_an_energy_budget() -> None:
    """Spending less time is not honouring a declared joule figure."""
    assert QueryAdaptivePolicy(table(), BASE, SCR, fit()).honours_energy_budget() is False


def test_the_decision_is_timed_like_every_other_policy() -> None:
    d = QueryAdaptivePolicy(table(), BASE, SCR, fit()).decide_timed(state(gap=1.2), cons())
    assert d.overhead_us >= 0.0
    assert d.config.ann.nprobe > 0


# ---------------------------------------------------------------------------
# Loading a recorded fit
# ---------------------------------------------------------------------------


def recorded(tmp_path: Path, dataset: str = "squad", delta: float = 0.01) -> Path:
    p = tmp_path / "query_adaptive.json"
    p.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "signal": "centroid_dist_gap",
                "bucket_edges": [1.1, 1.3, 1.6],
                "comparisons": [
                    {
                        "uniform_nprobe": 8,
                        "budget_ms": 100.0,
                        "assignment": [16, 8, 4, 1],
                        "held_out": {
                            "n": 500,
                            "delta_recall": delta,
                            "adaptive_mean_cost_ms": 80.0,
                            "uniform_mean_cost_ms": 100.0,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return p


def test_a_recorded_fit_loads_with_its_measured_outcome(tmp_path: Path) -> None:
    f = QueryEffortFit.load(recorded(tmp_path), budget_ms=100.0, dataset="squad")
    assert f.assignment == (16, 8, 4, 1)
    assert f.delta_recall == 0.01
    assert f.cost_saving == pytest.approx(0.2)
    assert f.is_demonstrated is True


def test_a_fit_from_another_corpus_is_refused(tmp_path: Path) -> None:
    """Difficulty buckets are a property of one index's cluster geometry."""
    with pytest.raises(ValueError, match="do not transfer"):
        QueryEffortFit.load(recorded(tmp_path, dataset="sift1m"), 100.0, "squad")


def test_a_budget_between_fits_uses_the_one_it_would_have_bought(tmp_path: Path) -> None:
    """Never interpolated: the allocation in use must be one held-out queries scored."""
    p = tmp_path / "two.json"
    p.write_text(
        json.dumps(
            {
                "dataset": "squad",
                "signal": "centroid_dist_gap",
                "bucket_edges": [1.2],
                "comparisons": [
                    {
                        "budget_ms": 50.0,
                        "assignment": [4, 1],
                        "held_out": {
                            "n": 500,
                            "delta_recall": 0.005,
                            "adaptive_mean_cost_ms": 48.0,
                            "uniform_mean_cost_ms": 50.0,
                        },
                    },
                    {
                        "budget_ms": 100.0,
                        "assignment": [16, 4],
                        "held_out": {
                            "n": 500,
                            "delta_recall": 0.01,
                            "adaptive_mean_cost_ms": 98.0,
                            "uniform_mean_cost_ms": 100.0,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    # 120 ms * 0.9 headroom = 108, so the 100 ms allocation fits and the 50 does too.
    assert QueryEffortFit.load(p, 120.0, "squad").assignment == (16, 4)
    # 80 ms * 0.9 = 72: only the 50 ms allocation fits.
    assert QueryEffortFit.load(p, 80.0, "squad").assignment == (4, 1)
    # Below every fitted budget, the cheapest rather than a refusal or a guess.
    assert QueryEffortFit.load(p, 10.0, "squad").assignment == (4, 1)


def test_a_file_with_no_allocations_is_refused(tmp_path: Path) -> None:
    p = tmp_path / "empty.json"
    p.write_text(
        json.dumps({"dataset": "squad", "signal": "g", "bucket_edges": [], "comparisons": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no fitted allocations"):
        QueryEffortFit.load(p, 100.0, "squad")


def test_a_fit_that_demonstrated_nothing_still_loads_and_reports_it(tmp_path: Path) -> None:
    """A negative result must stay inspectable, and must not look positive."""
    p = tmp_path / "null.json"
    p.write_text(
        json.dumps(
            {
                "dataset": "squad",
                "signal": "centroid_dist_gap",
                "bucket_edges": [1.2],
                "comparisons": [
                    {
                        "budget_ms": 100.0,
                        "assignment": [8, 8],
                        "held_out": {
                            "n": 500,
                            "delta_recall": 0.0,
                            "adaptive_mean_cost_ms": 100.0,
                            "uniform_mean_cost_ms": 100.0,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    f = QueryEffortFit.load(p, 100.0, "squad")
    assert f.is_demonstrated is False, "no recall gain and no cost saving is not a demonstration"


# ---------------------------------------------------------------------------
# The experiment's own arithmetic
# ---------------------------------------------------------------------------


def test_the_allocation_never_exceeds_the_budget() -> None:
    """The constraint is the whole comparison; break it and the win is bought."""
    import numpy as np

    from scripts.phase9_query_adaptive import best_allocation

    recall = np.array([[0.5, 0.8, 0.95], [0.6, 0.85, 0.97]])
    weights = np.array([0.5, 0.5])
    costs = np.array([10.0, 50.0, 200.0])

    combo, _, cost = best_allocation(recall, weights, costs, budget_ms=60.0)
    assert cost <= 60.0
    assert all(costs[c] <= 200.0 for c in combo)


def test_the_allocation_spends_more_where_recall_rises_faster() -> None:
    """Bucket 0 gains nothing from effort; bucket 1 gains a lot. Effort must move."""
    import numpy as np

    from scripts.phase9_query_adaptive import best_allocation

    recall = np.array([[0.90, 0.90], [0.40, 0.95]])  # flat bucket, steep bucket
    weights = np.array([0.5, 0.5])
    costs = np.array([10.0, 100.0])

    combo, _, _ = best_allocation(recall, weights, costs, budget_ms=60.0)
    assert combo == (0, 1), "effort belongs where it buys something"


def test_a_budget_too_small_for_anything_returns_the_cheapest_combination() -> None:
    import numpy as np

    from scripts.phase9_query_adaptive import best_allocation

    recall = np.array([[0.5, 0.9]])
    combo, _, cost = best_allocation(recall, np.array([1.0]), np.array([10.0, 100.0]), 10.0)
    assert combo == (0,)
    assert cost == 10.0

"""Figure-side computations. These produce the numbers a reader sees.

A bug here is worse than a bug in the pipeline: the pipeline has acceptance
checks watching it, and a plotting script has only whoever looks at the picture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from make_figures import _controller_p95, _rolling_p95, phase_stats


def _frame(latencies: list[float], t0: float = 1000.0, dt: float = 0.01) -> pd.DataFrame:
    n = len(latencies)
    return pd.DataFrame(
        {
            "t_wall": t0 + np.arange(n) * dt,
            "t_total": np.asarray(latencies, dtype=float),
            "recall_at_k": np.full(n, 0.9),
            "controller_overhead_us": np.full(n, 5.0),
            "cfg_ann_nprobe": np.full(n, 16),
        }
    )


def _manifest(df: pd.DataFrame, split_a: int, split_b: int, mid_target: float | None):
    t = df["t_wall"].to_numpy()
    start = {"name": "burners_start", "t_wall": float(t[split_a])}
    if mid_target is not None:
        start["target_ms"] = mid_target
    return {"events": [start, {"name": "burners_stop", "t_wall": float(t[split_b])}]}


# ---------------------------------------------------------------------------
# per-phase target (defect #11)
# ---------------------------------------------------------------------------
def test_middle_phase_is_scored_against_the_target_in_force() -> None:
    """A target-step run tightens the budget mid-run.

    Scoring the middle phase against the ORIGINAL target measures compliance
    with a constraint that was not in force. It reported static at 0.1% when the
    true rate was 28.2%.
    """
    # 10 quiet @ 20 ms, 10 middle @ 20 ms, 10 recovery @ 20 ms.
    df = _frame([20.0] * 30)
    mf = _manifest(df, 10, 20, mid_target=15.0)  # tightened to 15 ms

    st = phase_stats(df, mf, target=30.0)

    # 20 ms is inside the original 30 ms budget...
    assert st["quiet"]["violation_pct"] == 0.0
    assert st["recovery"]["violation_pct"] == 0.0
    # ...and outside the 15 ms budget that actually applied in the middle.
    assert st["contended"]["violation_pct"] == 100.0
    assert st["contended"]["target_ms"] == 15.0
    assert st["quiet"]["target_ms"] == 30.0


def test_contention_mode_keeps_one_target_for_every_phase() -> None:
    """With no target_ms on the event, every phase uses the run's target."""
    df = _frame([20.0] * 30)
    mf = _manifest(df, 10, 20, mid_target=None)

    st = phase_stats(df, mf, target=30.0)
    assert {p["target_ms"] for p in st.values()} == {30.0}
    assert all(p["violation_pct"] == 0.0 for p in st.values())


def test_violation_rate_counts_the_right_queries() -> None:
    df = _frame([10.0] * 5 + [40.0] * 5 + [10.0] * 10 + [10.0] * 10)
    mf = _manifest(df, 10, 20, mid_target=None)
    st = phase_stats(df, mf, target=30.0)
    assert st["quiet"]["violation_pct"] == pytest.approx(50.0)  # 5 of the first 10


# ---------------------------------------------------------------------------
# overhead: percentiles of a ratio, not a ratio of percentiles (defect #9)
# ---------------------------------------------------------------------------
def test_overhead_p95_is_not_below_p50() -> None:
    """The bug that made p95 (0.173%) come out under p50 (0.570%).

    Latency's spread is wider than overhead's, so dividing percentile by
    percentile shrinks the high end. For percentiles of one distribution,
    p95 >= p50 always.
    """
    rng = np.random.default_rng(0)
    lat = rng.lognormal(mean=3.0, sigma=0.9, size=4000)
    df = _frame(lat.tolist())
    df["controller_overhead_us"] = rng.normal(60.0, 6.0, size=lat.size)
    mf = _manifest(df, 1000, 3000, mid_target=None)

    st = phase_stats(df, mf, target=1e9)
    for phase in st.values():
        assert phase["overhead_p95_pct"] >= phase["overhead_p50_pct"]


# ---------------------------------------------------------------------------
# the estimator the controller consumed (defect #13)
# ---------------------------------------------------------------------------
def test_controller_p95_uses_only_prior_queries() -> None:
    """At query i the policy saw the window of [i-w, i-1]; never query i itself."""
    v = np.arange(1.0, 21.0)
    got = _controller_p95(v, window=5)
    assert np.isnan(got[0]), "no history exists before the first query"
    assert got[6] == pytest.approx(np.percentile(v[1:6], 95))
    assert got[6] < v[6], "the current query must not be in its own window"


def test_rolling_p95_is_linear_and_matches_the_naive_form() -> None:
    """The O(n) rewrite must not change the numbers it produces."""
    rng = np.random.default_rng(1)
    v = rng.lognormal(3.0, 0.5, size=600)
    t = np.cumsum(rng.uniform(0.005, 0.02, size=600))

    fast = _rolling_p95(t, v, window_s=1.0, stride=1)
    naive = np.array(
        [
            np.percentile(v[np.searchsorted(t, t[i] - 1.0, side="left") : i + 1], 95)
            for i in range(v.size)
        ]
    )
    np.testing.assert_allclose(fast, naive, rtol=1e-9)

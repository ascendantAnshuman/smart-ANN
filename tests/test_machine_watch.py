"""The machine-load guard that runs for the whole measurement, not just before it.

`assert_machine_quiet` samples the machine once, at t=0, and a 13-minute timing
run then proceeds on the strength of a one-second observation. That held until a
game launched partway through a verification: the start check passed, the
acceptance step took 4,686 s against a previous 796 s, and the only reason it
was not recorded as a measurement is that it blew through a timeout. Had it been
10% slower rather than 490%, it would have been written into the summary as a
result.

These tests pin the classification, because the 10% threshold is a judgement and
a judgement with no test is a number that drifts.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from phase2_acceptance import MachineWatch


def _watch(samples: list[float], max_util: float = 0.25) -> dict[str, float]:
    w = MachineWatch(max_util=max_util)
    w.samples = samples
    return w.stop()


class TestContaminationVerdict:
    def test_quiet_run_is_not_contaminated(self) -> None:
        assert _watch([0.05] * 100)["contaminated"] is False

    def test_sustained_load_is_contaminated(self) -> None:
        assert _watch([0.60] * 100)["contaminated"] is True

    def test_a_brief_spike_is_not_contamination(self) -> None:
        """Five percent of samples over threshold is the scheduler, not a workload.

        A guard that fired on any spike would be disabled within a week, which
        is worse than one that fires on sustained load only.
        """
        assert _watch([0.05] * 95 + [0.9] * 5)["contaminated"] is False

    def test_threshold_is_fraction_of_samples_not_the_mean(self) -> None:
        """A short, enormous spike must not contaminate by dragging the mean.

        One sample at 100% lifts the mean of 100 quiet samples to 6%, which a
        mean-based rule near 5% would trip on. The rule counts samples over the
        line instead, so magnitude cannot substitute for duration.
        """
        got = _watch([0.02] * 99 + [1.0])
        assert got["contaminated"] is False
        assert got["fraction_over_threshold"] == 0.01

    def test_just_over_the_line_is_contaminated(self) -> None:
        assert _watch([0.9] * 11 + [0.02] * 89)["contaminated"] is True

    def test_just_under_the_line_is_not(self) -> None:
        assert _watch([0.9] * 10 + [0.02] * 90)["contaminated"] is False


class TestReportedStatistics:
    def test_reports_the_distribution_not_just_a_verdict(self) -> None:
        """A verdict alone cannot be argued with; the numbers behind it can."""
        got = _watch([0.1] * 50 + [0.5] * 50)
        assert got["n"] == 100
        assert got["max"] == 0.5
        assert 0.1 <= got["median"] <= 0.5
        assert got["fraction_over_threshold"] == 0.5

    def test_no_samples_is_not_a_pass(self) -> None:
        """An empty watch must report n=0 rather than a clean bill of health.

        A run too short to sample has not been shown to be quiet; it has not
        been observed at all, and the two must not look the same.
        """
        got = _watch([])
        assert got["n"] == 0
        assert got["contaminated"] is False

    def test_threshold_is_configurable_and_respected(self) -> None:
        assert _watch([0.4] * 100, max_util=0.25)["contaminated"] is True
        assert _watch([0.4] * 100, max_util=0.50)["contaminated"] is False

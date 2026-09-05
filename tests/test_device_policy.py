"""C5 and C4's second half: policies that read device state and measured energy.

Neither could exist honestly before the device probe. The development laptop
reports no temperature and no energy, so a device-aware policy running on it
would have been reacting to NaN -- code that looks like a contribution and
demonstrates none. These tests pin the two properties that keep it honest: it
must not act on signals it cannot see, and it must not claim an energy budget it
is not measuring.
"""

from __future__ import annotations

import math

import pytest

from smartann.controller.policies.device import DeviceStatePolicy, DeviceTiers
from smartann.controller.policies.energy import (
    EnergyBudgetPolicy,
    EnergySample,
    integrate_mj,
)
from smartann.controller.profile import ProfileEntry, ProfileTable
from smartann.telemetry.monitor import unavailable_state
from smartann.types import (
    AnnConfig,
    Constraints,
    ControllerState,
    DeviceState,
    PerfHistory,
    QueryFeatures,
    ScrConfig,
)

ANN = AnnConfig(nprobe=8, ef_centroid=128, ef_list=64, k=10, per_cluster_k=32)
SCR = ScrConfig(1, 0, 0, 3, 0.0, enabled=False)


def table() -> ProfileTable:
    return ProfileTable(
        entries=tuple(
            ProfileEntry(
                nprobe=n, n_samples=500, p50_ms=p, p95_ms=p * 1.5, recall=r, approx_tokens=0.0
            )
            # p95 = p*1.5 -> 1.5, 9.0, 24.0 ms against a 27 ms budget, so a
            # derate actually moves the selection. A table whose every entry
            # fits the budget would make these tests pass without the policy
            # doing anything.
            for n, p, r in ((1, 1.0, 0.30), (4, 6.0, 0.60), (16, 16.0, 0.90))
        ),
        dataset="test",
        measured_under="synthetic",
        min_samples=100,
    )


def state(**kw: object) -> ControllerState:
    base: dict[str, object] = {
        "timestamp": 0.0,
        "temperature_c": float("nan"),
        "temperature_slope_c_per_s": float("nan"),
        "freq_ratio": float("nan"),
        "throttled": False,
        "cpu_util": float("nan"),
        "mem_available_mb": float("nan"),
        "mem_pressure": float("nan"),
        "battery_pct": float("nan"),
        "charging": False,
        "discharge_rate_ma": float("nan"),
    }
    base.update(kw)
    return ControllerState(
        DeviceState(**base),  # type: ignore[arg-type]
        QueryFeatures(0.0, 0.0, 0.0, 1.0, 0),
        PerfHistory(0, 0, 0, 0, 0, 0, 0, 0),
    )


CONSTRAINTS = Constraints(27.0, 81.0, float("inf"), None)


class TestDeviceStatePolicyRefusesToActBlind:
    def test_all_nan_state_derates_nothing(self) -> None:
        """A platform with no sensors must get the same config as a cool one.

        This is the property that lets the policy ship on the laptop without
        silently disabling retrieval.
        """
        p = DeviceStatePolicy(table(), ANN, SCR)
        cfg = p.decide(state(), CONSTRAINTS)
        assert cfg.ann.nprobe == 16
        assert p.reasons() == []

    def test_unavailable_state_is_handled(self) -> None:
        """The project's canonical all-NaN state must not raise."""
        s = ControllerState(
            unavailable_state(),
            QueryFeatures(0.0, 0.0, 0.0, 1.0, 0),
            PerfHistory(0, 0, 0, 0, 0, 0, 0, 0),
        )
        p = DeviceStatePolicy(table(), ANN, SCR)
        assert p.decide(s, CONSTRAINTS).ann.nprobe == 16

    def test_nan_temperature_is_not_treated_as_cold(self) -> None:
        blind = DeviceStatePolicy(table(), ANN, SCR)
        blind.decide(state(), CONSTRAINTS)
        cool = DeviceStatePolicy(table(), ANN, SCR)
        cool.decide(state(temperature_c=25.0), CONSTRAINTS)
        # Same config, but only one of them can say why.
        assert blind.reasons() == []
        assert cool.reasons() == []


class TestDeviceStatePolicyDerates:
    def test_hot_device_cuts_effort_hardest(self) -> None:
        p = DeviceStatePolicy(table(), ANN, SCR)
        hot = p.decide(state(temperature_c=47.0), CONSTRAINTS).ann.nprobe
        warm = p.decide(state(temperature_c=43.0), CONSTRAINTS).ann.nprobe
        cool = p.decide(state(temperature_c=30.0), CONSTRAINTS).ann.nprobe
        assert hot <= warm <= cool

    def test_derating_is_monotone_in_temperature(self) -> None:
        p = DeviceStatePolicy(table(), ANN, SCR)
        probes = [p.decide(state(temperature_c=t), CONSTRAINTS).ann.nprobe for t in (20, 43, 47)]
        assert probes == sorted(probes, reverse=True)

    def test_critical_battery_cuts_effort(self) -> None:
        p = DeviceStatePolicy(table(), ANN, SCR)
        low = p.decide(state(battery_pct=5.0), CONSTRAINTS).ann.nprobe
        assert any("critical" in r for r in p.reasons())
        full = p.decide(state(battery_pct=95.0), CONSTRAINTS).ann.nprobe
        assert low < full
        assert p.reasons() == []  # a healthy battery is not a reason to derate

    def test_charging_suppresses_the_battery_derate(self) -> None:
        """A phone on the charger at 5% is not in the same state as one on battery."""
        p = DeviceStatePolicy(table(), ANN, SCR)
        on_battery = p.decide(state(battery_pct=5.0, charging=False), CONSTRAINTS).ann.nprobe
        charging = p.decide(state(battery_pct=5.0, charging=True), CONSTRAINTS).ann.nprobe
        assert charging > on_battery

    def test_throttled_flag_derates(self) -> None:
        p = DeviceStatePolicy(table(), ANN, SCR)
        assert (
            p.decide(state(throttled=True), CONSTRAINTS).ann.nprobe
            < p.decide(state(throttled=False), CONSTRAINTS).ann.nprobe
        )

    def test_reasons_name_the_signal_used(self) -> None:
        p = DeviceStatePolicy(table(), ANN, SCR)
        p.decide(state(temperature_c=47.0, battery_pct=5.0), CONSTRAINTS)
        joined = " ".join(p.reasons())
        assert "temperature" in joined
        assert "battery" in joined

    def test_device_state_never_raises_effort_above_the_latency_budget(self) -> None:
        """Thermal management must not be able to break the latency SLO."""
        p = DeviceStatePolicy(table(), ANN, SCR)
        tight = Constraints(2.0, 10.0, float("inf"), None)
        cool = p.decide(state(temperature_c=20.0), tight).ann.nprobe
        assert cool == 1

    def test_tiers_reject_inverted_thresholds(self) -> None:
        with pytest.raises(ValueError, match="warm_c must be below hot_c"):
            DeviceTiers(warm_c=50.0, hot_c=40.0)
        with pytest.raises(ValueError, match="critical_battery_pct must be below"):
            DeviceTiers(low_battery_pct=5.0, critical_battery_pct=20.0)

    def test_does_not_claim_an_energy_budget(self) -> None:
        """Reading battery percentage is not honouring an energy budget."""
        assert DeviceStatePolicy(table(), ANN, SCR).honours_energy_budget() is False


class TestEnergyIntegration:
    def test_constant_power_integrates_to_power_times_time(self) -> None:
        s = [EnergySample(t, 4.0, 0.5) for t in (0.0, 1.0, 2.0)]
        assert integrate_mj(s) == pytest.approx(2.0 * 2.0 * 1000.0)

    def test_uses_voltage_rather_than_a_nominal_constant(self) -> None:
        """Voltage sags under load (4280 -> 4248 mV); V*I*dt, not I*dt."""
        sag = integrate_mj([EnergySample(0.0, 4.248, 1.0), EnergySample(1.0, 4.248, 1.0)])
        nom = integrate_mj([EnergySample(0.0, 4.280, 1.0), EnergySample(1.0, 4.280, 1.0)])
        assert sag < nom

    def test_trapezoid_not_rectangle_across_a_transition(self) -> None:
        """A rectangle rule would attribute the whole interval to one endpoint."""
        got = integrate_mj([EnergySample(0.0, 4.0, 0.0), EnergySample(1.0, 4.0, 1.0)])
        assert got == pytest.approx(2000.0)

    def test_fewer_than_two_samples_is_zero_not_an_error(self) -> None:
        assert integrate_mj([]) == 0.0
        assert integrate_mj([EnergySample(0.0, 4.0, 1.0)]) == 0.0

    def test_non_advancing_time_is_skipped(self) -> None:
        assert integrate_mj([EnergySample(1.0, 4.0, 1.0), EnergySample(1.0, 4.0, 1.0)]) == 0.0


class TestEnergyBudgetPolicyEarnsItsClaim:
    def test_without_a_source_it_does_not_claim_an_energy_budget(self) -> None:
        """This is what runs on the laptop, and it must report False."""
        p = EnergyBudgetPolicy(table(), ANN, SCR)
        assert p.honours_energy_budget() is False

    def test_a_source_without_the_capability_flag_is_refused(self) -> None:
        """A wired mock is not a declared capability."""
        p = EnergyBudgetPolicy(table(), ANN, SCR, energy_source=lambda: EnergySample(0.0, 4.0, 1.0))
        assert p.honours_energy_budget() is False

    def test_a_flag_without_a_source_is_refused(self) -> None:
        p = EnergyBudgetPolicy(table(), ANN, SCR, energy_measurable=True)
        assert p.honours_energy_budget() is False

    def test_both_together_earn_the_claim(self) -> None:
        p = EnergyBudgetPolicy(
            table(),
            ANN,
            SCR,
            energy_source=lambda: EnergySample(0.0, 4.0, 1.0),
            energy_measurable=True,
        )
        assert p.honours_energy_budget() is True

    def test_observed_energy_is_nan_before_enough_samples(self) -> None:
        """NaN, not 0.0 -- zero would read as free and disable the constraint."""
        p = EnergyBudgetPolicy(table(), ANN, SCR)
        assert math.isnan(p.observed_mj_per_query())

    def test_exceeding_the_budget_lowers_effort(self) -> None:
        t = [0.0]

        def src() -> EnergySample:
            t[0] += 1.0
            return EnergySample(t[0], 4.0, 1.0)  # 4 W, large against any budget

        p = EnergyBudgetPolicy(table(), ANN, SCR, energy_source=src, energy_measurable=True)
        tiny = Constraints(27.0, 81.0, 0.001, None)
        for _ in range(6):
            cfg = p.decide(state(), tiny)
        assert cfg.ann.nprobe == 1
        assert p.observed_mj_per_query() > 0

    def test_an_infinite_budget_leaves_effort_alone(self) -> None:
        p = EnergyBudgetPolicy(
            table(),
            ANN,
            SCR,
            energy_source=lambda: EnergySample(0.0, 4.0, 1.0),
            energy_measurable=True,
        )
        assert p.decide(state(), CONSTRAINTS).ann.nprobe == 16

    def test_a_failing_source_does_not_read_as_zero_energy(self) -> None:
        """A failed read is missing data, not free queries."""

        def broken() -> EnergySample:
            raise RuntimeError("adb died")

        p = EnergyBudgetPolicy(table(), ANN, SCR, energy_source=broken, energy_measurable=True)
        p.decide(state(), Constraints(27.0, 81.0, 0.001, None))
        assert math.isnan(p.observed_mj_per_query())

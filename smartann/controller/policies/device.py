"""C5: a policy that reads battery and thermal state and acts on it.

Until the device probe (ADR-0046) this could not exist honestly. The
development laptop reports no temperature and no energy, so a device-aware
policy running on it would have been reading NaN and reacting to nothing --
code that looks like a contribution and demonstrates none.

**It refuses to act on state it cannot see.** Every tier below is gated on the
signal being finite. A NaN temperature means "not measured", not "cold", and a
policy that treats them the same would silently disable itself on exactly the
platforms this project says it cannot measure. `reasons()` reports which inputs
were actually used, so a run where every input was NaN is visible as one.

**The tiers are thresholds, not a model.** No thermal model exists yet -- Phase
4 must fit one from a measured trace, and choosing constants here would be the
"fitted, never chosen" rule broken in a new place. What this policy encodes is
the ordering that survives without a model: hotter and emptier means less
effort, monotonically. When Phase 4 lands, the thresholds become predictions and
this class is what they replace.
"""

from __future__ import annotations

import math

from smartann.controller.base import Policy
from smartann.controller.profile import ProfileTable
from smartann.types import AnnConfig, Constraints, ControllerState, PipelineConfig, ScrConfig

__all__ = ["DeviceStatePolicy", "DeviceTiers"]


class DeviceTiers:
    """Where effort is cut, and why each line sits where it does.

    The temperature thresholds are anchored on the one thermal measurement this
    project has: the handset's skin zone sat at 36.6 C idle and reached 41.4 C
    after 60 s of full load (ADR-0047). So 42 C is "sustained load", not an
    arbitrary round number, and 46 C is above anything observed -- deliberately,
    because a shield should trip on states the calibration did not visit.

    Battery thresholds are conventional (Android's own battery-saver default is
    15%) and are NOT measurements. They are labelled as such.
    """

    def __init__(
        self,
        warm_c: float = 42.0,
        hot_c: float = 46.0,
        low_battery_pct: float = 20.0,
        critical_battery_pct: float = 10.0,
        rising_c_per_s: float = 0.05,
    ) -> None:
        if not warm_c < hot_c:
            raise ValueError(f"warm_c must be below hot_c, got {warm_c} >= {hot_c}")
        if not critical_battery_pct < low_battery_pct:
            raise ValueError(
                f"critical_battery_pct must be below low_battery_pct, "
                f"got {critical_battery_pct} >= {low_battery_pct}"
            )
        self.warm_c = warm_c
        self.hot_c = hot_c
        self.low_battery_pct = low_battery_pct
        self.critical_battery_pct = critical_battery_pct
        self.rising_c_per_s = rising_c_per_s


class DeviceStatePolicy(Policy):
    """Chooses effort from the profile table, derated by device state.

    Composes with `BudgetPolicy` rather than replacing it: the latency budget
    still selects the ceiling, and device state only ever lowers it. That
    ordering matters -- a policy that could RAISE effort on a hot device would
    be able to violate the latency budget in the name of thermal management,
    which is not a trade anyone asked for.
    """

    name = "device"

    def __init__(
        self,
        table: ProfileTable,
        base_ann: AnnConfig,
        scr: ScrConfig,
        tiers: DeviceTiers | None = None,
        *,
        headroom: float = 0.9,
    ) -> None:
        self.table = table
        self.base_ann = base_ann
        self.scr = scr
        self.tiers = tiers or DeviceTiers()
        self.headroom = headroom
        self._reasons: list[str] = []

    def reset(self) -> None:
        self._reasons = []

    def reasons(self) -> list[str]:
        """Which device signals the last decision actually used.

        Empty means the policy ran blind -- every input was NaN. That is a
        legitimate state on a platform without sensors, and it must be
        distinguishable from "the device was fine", which is why this is
        reported rather than inferred from the chosen config.
        """
        return list(self._reasons)

    def _derate(self, state: ControllerState) -> tuple[float, list[str]]:
        """Multiplier in (0, 1] applied to the latency budget, and why.

        Derating the BUDGET rather than the config keeps the profile table the
        single source of what each effort level costs. Picking a lower nprobe
        directly would duplicate that knowledge and let the two disagree.
        """
        d = state.device
        factor = 1.0
        why: list[str] = []

        t = d.temperature_c
        if isinstance(t, float) and math.isfinite(t):
            if t >= self.tiers.hot_c:
                factor = min(factor, 0.35)
                why.append(f"temperature {t:.1f}C >= hot {self.tiers.hot_c:.0f}C")
            elif t >= self.tiers.warm_c:
                factor = min(factor, 0.65)
                why.append(f"temperature {t:.1f}C >= warm {self.tiers.warm_c:.0f}C")

        slope = d.temperature_slope_c_per_s
        if isinstance(slope, float) and math.isfinite(slope) and slope >= self.tiers.rising_c_per_s:
            factor = min(factor, 0.7)
            why.append(f"temperature rising {slope:+.3f} C/s")

        pct = d.battery_pct
        if isinstance(pct, float) and math.isfinite(pct) and not d.charging:
            if pct <= self.tiers.critical_battery_pct:
                factor = min(factor, 0.3)
                why.append(f"battery {pct:.0f}% <= critical")
            elif pct <= self.tiers.low_battery_pct:
                factor = min(factor, 0.6)
                why.append(f"battery {pct:.0f}% <= low")
        elif d.charging:
            # Charging is a real signal and is used, but only to DECLINE to
            # derate on battery grounds -- never to raise effort above what the
            # latency budget allows.
            why.append("charging: no battery derate")

        if d.throttled:
            factor = min(factor, 0.5)
            why.append("device reports throttled")

        return factor, why

    def decide(self, state: ControllerState, constraints: Constraints) -> PipelineConfig:
        factor, why = self._derate(state)
        self._reasons = why
        budget = constraints.latency_target_ms * self.headroom * factor
        entry = self.table.best_within(budget)
        return PipelineConfig(
            ann=AnnConfig(
                nprobe=entry.nprobe,
                ef_centroid=self.base_ann.ef_centroid,
                ef_list=self.base_ann.ef_list,
                k=self.base_ann.k,
                per_cluster_k=self.base_ann.per_cluster_k,
            ),
            scr=self.scr,
        )

    def honours_energy_budget(self) -> bool:
        """Still False, and it must stay False until energy is CONSUMED.

        The measurement now exists (ADR-0047) and this policy reads battery
        percentage and charging state. Neither is an energy budget. Returning
        True here because a related number became readable is exactly the slip
        the capability gate exists to prevent -- see `EnergyBudgetPolicy`, which
        earns it by integrating V*I*dt against a declared budget.
        """
        return False

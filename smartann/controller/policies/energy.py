"""C4's second half: a policy that honours an energy budget it can measure.

`BudgetPolicy.honours_energy_budget()` returns False and has since Phase 6,
because no energy could be measured anywhere in this project (ADR-0007,
ADR-0008). ADR-0047 changed that: battery current is readable without root, the
sign convention is established, and a coulomb counter agrees with the current
sensor to within 12%.

**This class returns True only when it is actually measuring.** It takes an
`energy_source` callable and a capability flag, and if either is absent it says
so and behaves exactly like the latency-only policy. The alternative -- deciding
from the Phase 3 energy model -- is available and deliberately refused: those
coefficients are MobileRAG's reference values, unfitted on this hardware, and a
policy acting on them would report an energy budget honoured against numbers no
one measured.

**Energy is integrated as V*I*dt, not I*dt.** Voltage sags under load, 4280 ->
4248 mV in the calibration, and treating it as constant understates load power.

**The controller cannot attribute energy per query yet, and does not pretend
to.** Idle draw on the calibrated handset spans -1050 to -265 mA because the
screen and radios are uncontrolled, which is larger than any single query's
cost. So this policy tracks a *rolling window* of measured energy against a
*per-query budget scaled to that window*, and its own docstring is the place
that says the resolution limit rather than a footnote in a paper.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from itertools import pairwise

from smartann.controller.base import Policy
from smartann.controller.profile import ProfileTable
from smartann.types import AnnConfig, Constraints, ControllerState, PipelineConfig, ScrConfig

__all__ = ["EnergyBudgetPolicy", "EnergySample", "integrate_mj"]


class EnergySample:
    """One instantaneous power reading, with the time it was taken."""

    __slots__ = ("amps", "t", "volts")

    def __init__(self, t: float, volts: float, amps: float) -> None:
        self.t = t
        self.volts = volts
        self.amps = amps

    @property
    def watts(self) -> float:
        return self.volts * self.amps


def integrate_mj(samples: list[EnergySample]) -> float:
    """Trapezoidal V*I*dt over consecutive samples, in millijoules.

    Trapezoid rather than rectangle because power is being sampled, not
    counted: a rectangle rule attributes the whole interval to whichever
    endpoint it picks, and at 2-second spacing across a load transition that is
    a visible error rather than a rounding one.
    """
    if len(samples) < 2:
        return 0.0
    total_j = 0.0
    for a, b in pairwise(samples):
        dt = b.t - a.t
        if dt <= 0:
            continue
        total_j += 0.5 * (a.watts + b.watts) * dt
    return total_j * 1000.0


class EnergyBudgetPolicy(Policy):
    """Latency-budget selection, further constrained by measured energy.

    Composes with the latency budget: energy can only ever LOWER the chosen
    effort. A policy that raised effort to spend a surplus would be optimising
    for depletion, which no constraint in this project asks for.
    """

    name = "energy_budget"

    def __init__(
        self,
        table: ProfileTable,
        base_ann: AnnConfig,
        scr: ScrConfig,
        *,
        energy_source: Callable[[], EnergySample] | None = None,
        energy_measurable: bool = False,
        window: int = 30,
        headroom: float = 0.9,
    ) -> None:
        self.table = table
        self.base_ann = base_ann
        self.scr = scr
        # BOTH are required. A source without the capability flag is a caller
        # who wired up a mock; a flag without a source is a caller who declared
        # a capability the platform has but this policy cannot reach.
        self.energy_source = energy_source
        self.energy_measurable = bool(energy_measurable and energy_source is not None)
        self.window = window
        self.headroom = headroom
        self._samples: deque[EnergySample] = deque(maxlen=window)
        self._queries_in_window = 0
        self._last_mj_per_query = float("nan")

    def reset(self) -> None:
        self._samples.clear()
        self._queries_in_window = 0
        self._last_mj_per_query = float("nan")

    def observed_mj_per_query(self) -> float:
        """Measured energy per query over the window, or NaN.

        NaN is the honest value before enough samples exist. Returning 0.0
        would read as "free", which is the direction that disables the
        constraint.
        """
        return self._last_mj_per_query

    def _update_energy(self) -> None:
        if not self.energy_measurable or self.energy_source is None:
            return
        try:
            self._samples.append(self.energy_source())
        except Exception:
            # A failed read is missing data, not zero energy. The window keeps
            # what it has and the estimate stays stale rather than dropping.
            return
        self._queries_in_window += 1
        if len(self._samples) >= 3 and self._queries_in_window > 0:
            total = integrate_mj(list(self._samples))
            self._last_mj_per_query = total / self._queries_in_window

    def decide(self, state: ControllerState, constraints: Constraints) -> PipelineConfig:
        # `state` is unused: this policy decides from MEASURED energy, not from
        # telemetry. Reading device state here and calling it energy is the slip
        # the capability gate exists to prevent -- DeviceStatePolicy is where
        # thermal and battery state belong.
        del state
        self._update_energy()

        budget_ms = constraints.latency_target_ms * self.headroom
        if (
            self.energy_measurable
            and math.isfinite(constraints.energy_budget_mj)
            and math.isfinite(self._last_mj_per_query)
            and constraints.energy_budget_mj > 0
            and self._last_mj_per_query > constraints.energy_budget_mj
        ):
            # Scale the latency budget by how far over we are, floored so a
            # transient spike cannot drive effort to zero. Proportional rather
            # than multiplicative-decrease because the energy signal is a
            # windowed mean and already heavily smoothed -- halving on top of
            # that would oscillate.
            ratio = constraints.energy_budget_mj / self._last_mj_per_query
            budget_ms *= max(0.25, ratio)

        entry = self.table.best_within(budget_ms)
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
        """True only when a real energy source is wired in and declared.

        This is the first True in the project's history, and it is conditional
        on the instance rather than on the class. Constructed without a source
        -- which is what happens on the development laptop -- it reports False,
        exactly as `BudgetPolicy` does, and every manifest records that.
        """
        return self.energy_measurable

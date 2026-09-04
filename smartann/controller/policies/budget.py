"""Budget-aware policy: pick the most accurate configuration that fits.

The third policy, and the one that closes contribution C4 as far as this
platform allows. Given a declared latency budget it looks up the highest-recall
effort level whose *measured* p95 fits, and it re-checks that choice as observed
latency drifts.

**It looks up rather than predicts, and that is the point.** Phase 3's latency
model holds at 14.75% MAPE inside the regime it was fitted on and fails at
35.87% outside it, because `cpu_util` saturates (ADR-0033). A policy predicting
latency from that model would be most confident exactly where it is most wrong.
The profile table is measurements, so its failure mode is honest: if the machine
no longer behaves as when the table was built, the *observed* p95 diverges from
the table's and the policy scales its effective budget to compensate.

**Energy budgets are accepted and not honoured, deliberately.** `Constraints`
carries `energy_budget_mj`, but no energy can be measured on this platform and
Phase 3 left the energy coefficients unfitted (ADR-0007, ADR-0008). Silently
treating a modelled energy as a measured one is exactly the failure the
capability gate exists to prevent, so this policy refuses to act on an energy
budget and says so. Phase 3.5 supplies the measurement that would let it.
"""

from __future__ import annotations

import math

from smartann.controller.base import Policy
from smartann.controller.profile import ProfileTable
from smartann.types import AnnConfig, Constraints, ControllerState, PipelineConfig, ScrConfig

__all__ = ["BudgetPolicy"]


class BudgetPolicy(Policy):
    """Chooses the highest-recall effort level whose measured p95 fits the budget."""

    name = "budget"

    def __init__(
        self,
        table: ProfileTable,
        base_ann: AnnConfig,
        scr: ScrConfig,
        *,
        headroom: float = 0.9,
        adapt_to_drift: bool = True,
        min_window: int = 30,
    ) -> None:
        if not 0.0 < headroom <= 1.0:
            raise ValueError(f"headroom must be in (0,1], got {headroom}")
        self.table = table
        self.base_ann = base_ann
        self.scr = scr
        # Aim below the budget rather than at it. The table's p95 was measured on
        # a machine in one state; targeting the budget exactly would put half the
        # margin of error on the wrong side of it.
        self.headroom = headroom
        self.adapt_to_drift = adapt_to_drift
        self.min_window = min_window
        self._scale = 1.0

    def reset(self) -> None:
        self._scale = 1.0

    def _effective_budget(self, state: ControllerState, budget_ms: float) -> float:
        """Shrink the budget when the machine is slower than the table assumed.

        Not a prediction: the ratio of observed p95 to the table's p95 for the
        level currently in use is a direct measurement of how far conditions have
        drifted. Nothing here extrapolates.
        """
        if not self.adapt_to_drift or state.history.queries_in_window < self.min_window:
            return budget_ms * self.headroom
        observed = state.history.p95_latency_ms
        expected = self.table.best_within(budget_ms * self.headroom).p95_ms
        if observed > 0 and expected > 0:
            self._scale = min(1.0, expected / observed)
        return budget_ms * self.headroom * self._scale

    def decide(self, state: ControllerState, constraints: Constraints) -> PipelineConfig:
        if math.isfinite(constraints.energy_budget_mj):
            # Accepted, not honoured. Acting on it would require an energy
            # measurement this platform cannot make; pretending otherwise is the
            # error the capability gate exists to prevent.
            pass

        entry = self.table.best_within(self._effective_budget(state, constraints.latency_target_ms))
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
        """False here, and it must stay false until energy can be measured.

        Exposed as a method so an experiment can record it in the manifest rather
        than a reader having to infer it from the absence of a number.
        """
        return False

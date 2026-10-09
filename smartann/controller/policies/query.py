"""C6: a policy that spends more effort on queries the probe says are harder.

**The signal was already free.** `probe()` computes the centroid distance gap --
how far the second nearest centroid lies beyond the nearest -- before any
cluster is opened, and hands it to every controller decision. It is a
difference, not a ratio, because a ratio divides by a distance that can be zero
or negative for duplicate vectors. A large gap means one cluster clearly
dominates and the answer is almost certainly inside it; a gap near zero means
the query sits between clusters and its neighbours are scattered. Until this
policy existed, every one of the five policies here ignored it, and every query
at a given budget got identical effort.

**The allocation is fitted, never chosen.** This class decides no threshold of
its own. It loads an assignment that `scripts/phase9_query_adaptive.py` produced
by measuring recall at every level for every difficulty bucket and taking
whichever assignment maximised mean recall under a measured device cost.
`QueryEffortFit` can be constructed directly, which is what the tests do, but
`load` is the path any reported result must come from.

**The separation it rests on is large.** Phase 9 measures recall at nprobe=1
running from 0.278 in the smallest-gap fifth of queries to 0.978 in the largest.
If those were equal the signal would carry nothing and no allocation could help.

**It refuses to run on a fit it cannot trust.** A fit whose held-out comparison
recorded no improvement is still loadable -- a negative result must stay
inspectable -- but `is_demonstrated` reports it, and the CLI declines to make it
the default on that basis rather than silently shipping a policy that measured
nothing.

**It cannot exceed the budget's level.** Like `DeviceStatePolicy`, this composes
over `BudgetPolicy` rather than replacing it: the budget still sets the ceiling
and difficulty only moves effort below it. A policy that could raise effort
above the budget on a query it judged hard would be able to miss the latency
target in the name of recall, which is not a trade anyone declared.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from smartann.controller.base import Policy
from smartann.controller.profile import ProfileTable
from smartann.types import AnnConfig, Constraints, ControllerState, PipelineConfig, ScrConfig

__all__ = ["QueryAdaptivePolicy", "QueryEffortFit"]


@dataclass(frozen=True)
class QueryEffortFit:
    """A measured mapping from query difficulty to effort level.

    `edges` are quantile boundaries of the gap distribution, measured on the fit
    half of a query set. `assignment[i]` is the nprobe for the bucket between
    edge i-1 and edge i. Both come from a recorded experiment; neither is a
    constant anyone typed.
    """

    signal: str
    edges: tuple[float, ...]
    assignment: tuple[int, ...]
    budget_ms: float
    dataset: str
    delta_recall: float
    cost_saving: float
    n_held_out: int

    def __post_init__(self) -> None:
        if len(self.assignment) != len(self.edges) + 1:
            raise ValueError(
                f"{len(self.edges)} edges define {len(self.edges) + 1} buckets but "
                f"{len(self.assignment)} levels were given"
            )
        if not self.assignment:
            raise ValueError("an empty assignment cannot decide anything")

    @property
    def is_demonstrated(self) -> bool:
        """Did the held-out comparison show this fit doing anything?

        False is a legitimate, loadable state. What it must never do is look the
        same as True to a caller choosing a default.
        """
        return self.delta_recall > 0.0 or self.cost_saving > 0.0

    def level_for(self, gap: float) -> int | None:
        """The fitted level for this query, or None when the signal is absent.

        None rather than a middle bucket: a NaN gap means the probe did not
        report one, and guessing a bucket would attribute an effort choice to a
        measurement that was never taken.
        """
        if not isinstance(gap, (int, float)) or not math.isfinite(gap):
            return None
        bucket = 0
        for edge in self.edges:
            if gap < edge:
                break
            bucket += 1
        return self.assignment[min(bucket, len(self.assignment) - 1)]

    @classmethod
    def load(
        cls, path: Path, budget_ms: float, dataset: str, *, headroom: float = 0.9
    ) -> QueryEffortFit:
        """The recorded allocation for the level this budget would have bought.

        Selection mirrors `ProfileTable.best_within`: the most expensive fitted
        allocation whose budget still fits, falling back to the cheapest when
        none does. That is deliberate rather than convenient -- it means the
        allocation in use is the one that was fitted and scored against exactly
        the uniform level the budget policy would otherwise have chosen, so the
        recorded comparison describes the substitution actually being made.

        Nothing is interpolated. An allocation blended between two measured ones
        would be a third that no held-out query ever scored, and its recall
        would be a guess wearing a measurement's clothes.
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("dataset") != dataset:
            raise ValueError(
                f"{path} was fitted on {raw.get('dataset')!r}; this index is {dataset!r}. "
                f"Difficulty buckets do not transfer between corpora."
            )
        comps = raw.get("comparisons") or []
        if not comps:
            raise ValueError(f"{path} records no fitted allocations")

        effective = budget_ms * headroom
        fits = [c for c in comps if c["budget_ms"] <= effective]
        match = (
            max(fits, key=lambda c: c["budget_ms"])
            if fits
            else min(comps, key=lambda c: c["budget_ms"])
        )
        h = match["held_out"]
        uniform_cost = h["uniform_mean_cost_ms"]
        return cls(
            signal=raw["signal"],
            edges=tuple(float(e) for e in raw["bucket_edges"]),
            assignment=tuple(int(a) for a in match["assignment"]),
            budget_ms=float(match["budget_ms"]),
            dataset=str(raw["dataset"]),
            delta_recall=float(h["delta_recall"]),
            cost_saving=(1.0 - h["adaptive_mean_cost_ms"] / uniform_cost if uniform_cost else 0.0),
            n_held_out=int(h["n"]),
        )


class QueryAdaptivePolicy(Policy):
    """Chooses effort from the fitted difficulty mapping, capped by the budget."""

    name = "query-adaptive"

    def __init__(
        self,
        table: ProfileTable,
        base_ann: AnnConfig,
        scr: ScrConfig,
        fit: QueryEffortFit,
        *,
        headroom: float = 0.9,
    ) -> None:
        self.table = table
        self.base_ann = base_ann
        self.scr = scr
        self.fit = fit
        self.headroom = headroom
        self._reasons: list[str] = []

    def reset(self) -> None:
        self._reasons = []

    def reasons(self) -> list[str]:
        return list(self._reasons)

    def decide(self, state: ControllerState, constraints: Constraints) -> PipelineConfig:
        ceiling = self.table.best_within(constraints.latency_target_ms * self.headroom)
        gap = getattr(state.query, self.fit.signal, float("nan"))
        wanted = self.fit.level_for(gap)

        if wanted is None:
            self._reasons = []  # the probe reported no gap; spend the budget's level
            nprobe = ceiling.nprobe
        else:
            # Capped, never raised: difficulty moves effort down inside the
            # budget, it does not buy a level the budget did not allow.
            nprobe = min(wanted, ceiling.nprobe)
            if nprobe < ceiling.nprobe:
                self._reasons = [
                    f"{self.fit.signal} {gap:.3f} -> fitted level {wanted} "
                    f"(budget allowed {ceiling.nprobe})"
                ]
            else:
                self._reasons = [f"{self.fit.signal} {gap:.3f} -> budget's level {nprobe}"]

        return PipelineConfig(
            ann=AnnConfig(
                nprobe=nprobe,
                ef_centroid=self.base_ann.ef_centroid,
                ef_list=self.base_ann.ef_list,
                k=self.base_ann.k,
                per_cluster_k=self.base_ann.per_cluster_k,
            ),
            scr=self.scr,
        )

    def honours_energy_budget(self) -> bool:
        """False. This policy reads query difficulty, which is not energy.

        Spending less time does use less energy, and the phase 9 result records
        a cost saving in milliseconds. Neither is an energy budget honoured
        against a declared joule figure, and returning True because a related
        quantity improved is the slip the capability gate exists to prevent.
        """
        return False

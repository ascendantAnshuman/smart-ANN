"""AIMD latency governor: the adaptive policy for the review demonstration.

Additive-increase / multiplicative-decrease, the control law TCP congestion
avoidance uses, applied to retrieval effort instead of window size. The analogy
is close: a shared resource whose available capacity changes without warning, a
signal that says "you are over budget", and no model of why.

That last point is the demonstration's whole argument. The governor does not
know whether latency rose because the CPU is contended, because the device is
thermally throttled, or because the query was hard. It observes p95 against a
target and moves effort. Thermal adaptation and contention adaptation are the
same mechanism responding to the same signal, which is why the phone's absence
does not prevent showing that the mechanism works.

Effort is a scalar in [0, 1] mapped monotonically onto `nprobe`. With `ef_list`
inert once in-cluster search is exhaustive (ADR-0024) and `ef_centroid` free
(ADR-0025), `nprobe` is the knob that actually moves the latency-recall
trade-off, so the action space is one-dimensional by measurement rather than by
simplification.
"""

from __future__ import annotations

from dataclasses import dataclass

from smartann.controller.base import Policy
from smartann.types import AnnConfig, Constraints, ControllerState, PipelineConfig, ScrConfig

__all__ = ["AimdConfig", "AimdPolicy"]


@dataclass(frozen=True)
class AimdConfig:
    """Tuning for the governor. Lives in YAML, not in the code."""

    nprobe_min: int = 1
    nprobe_max: int = 64
    additive_increase: float = 0.05  # effort added per query while under budget
    multiplicative_decrease: float = 0.5  # effort retained on an overshoot
    low_water: float = 0.8  # increase only below low_water * target
    warmup_queries: int = 20  # decide nothing until the window is meaningful
    effort_init: float = 1.0  # start at full effort and back off, not the reverse


class AimdPolicy(Policy):
    """Cuts effort when p95 latency exceeds target, restores it when it recovers."""

    name = "aimd"

    def __init__(self, cfg: AimdConfig, base_ann: AnnConfig, scr: ScrConfig) -> None:
        if not 0.0 < cfg.multiplicative_decrease < 1.0:
            raise ValueError(
                f"multiplicative_decrease must be in (0,1), got {cfg.multiplicative_decrease}"
            )
        if cfg.additive_increase <= 0:
            raise ValueError(f"additive_increase must be positive, got {cfg.additive_increase}")
        if cfg.nprobe_min < 1 or cfg.nprobe_max < cfg.nprobe_min:
            raise ValueError(f"bad nprobe range [{cfg.nprobe_min}, {cfg.nprobe_max}]")
        self.cfg = cfg
        self.base_ann = base_ann
        self.scr = scr
        self.effort = cfg.effort_init
        self._seen = 0

    def reset(self) -> None:
        self.effort = self.cfg.effort_init
        self._seen = 0

    def _nprobe(self) -> int:
        """Map effort in [0,1] onto the nprobe range, monotonically.

        Linear rather than geometric. Latency is close to linear in nprobe
        (each probe is one cluster load plus one scan), so a linear map makes
        the control law's additive step a roughly constant latency step, which
        is what AIMD's stability argument assumes.
        """
        lo, hi = self.cfg.nprobe_min, self.cfg.nprobe_max
        e = min(1.0, max(0.0, self.effort))
        return round(lo + e * (hi - lo))

    def decide(self, state: ControllerState, constraints: Constraints) -> PipelineConfig:
        self._seen += 1
        target = constraints.latency_target_ms
        p95 = state.history.p95_latency_ms

        # Hold full effort until the rolling window means something. Reacting to
        # a two-sample p95 would be reacting to noise.
        if self._seen > self.cfg.warmup_queries and state.history.queries_in_window > 0:
            if p95 > target:
                self.effort *= self.cfg.multiplicative_decrease
            elif p95 < self.cfg.low_water * target:
                self.effort += self.cfg.additive_increase
            self.effort = min(1.0, max(0.0, self.effort))

        return PipelineConfig(
            ann=AnnConfig(
                nprobe=self._nprobe(),
                ef_centroid=self.base_ann.ef_centroid,
                ef_list=self.base_ann.ef_list,
                k=self.base_ann.k,
                per_cluster_k=self.base_ann.per_cluster_k,
            ),
            scr=self.scr,
        )

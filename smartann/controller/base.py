"""The policy contract. Everything the controller does goes through this.

A policy maps observed state to a retrieval configuration. It sees only what a
real deployment could see -- device telemetry, cheap query features from the
probe phase, and its own rolling performance history -- and never the recall it
is about to achieve. A policy that could read the outcome would be an oracle,
which is a useful upper bound but not a controller.

``decide()`` is timed on every query and the cost recorded in
``controller_overhead_us``. A control layer that spends more than it saves is a
negative result, and Section 3 rule 5 makes measuring that mandatory rather than
optional.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from smartann.types import Constraints, ControllerState, PipelineConfig

__all__ = ["Policy", "TimedDecision"]


class TimedDecision:
    """A config plus what it cost to choose it."""

    __slots__ = ("config", "overhead_us")

    def __init__(self, config: PipelineConfig, overhead_us: float) -> None:
        self.config = config
        self.overhead_us = overhead_us


class Policy(ABC):
    """Maps (state, constraints) to a retrieval configuration."""

    name: str = "policy"

    @abstractmethod
    def decide(self, state: ControllerState, constraints: Constraints) -> PipelineConfig:
        """Choose a configuration for the query about to run.

        Must not block, must not perform I/O, and must not consult the ground
        truth. Called on the query path with the clock running.
        """

    def update(self, record: object) -> None:  # noqa: B027
        """Observe the outcome of the query just executed.

        Default is to ignore it: a static policy has nothing to learn, and
        making this abstract would force every policy to write an empty method.
        """

    def reset(self) -> None:  # noqa: B027
        """Clear any internal state. Called between runs so they are independent."""

    def decide_timed(self, state: ControllerState, constraints: Constraints) -> TimedDecision:
        """`decide()` with its own cost measured.

        perf_counter around the call and nothing else inside the timed region,
        so the number is the policy's cost and not the harness's.
        """
        t0 = time.perf_counter()
        cfg = self.decide(state, constraints)
        return TimedDecision(cfg, (time.perf_counter() - t0) * 1e6)

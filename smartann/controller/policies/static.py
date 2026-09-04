"""The static baseline. Tuned, not crippled (Section 3, rule 3).

The comparison this project rests on is adaptive-versus-static, and a static
baseline chosen badly makes the adaptive result meaningless. So the config it
holds is picked by a recorded sweep under no contention, not by intuition, and
`select_best_static` is the function that picks it -- kept here beside the policy
so the tuning cannot drift away from the thing it tunes.
"""

from __future__ import annotations

from smartann.controller.base import Policy
from smartann.types import AnnConfig, Constraints, ControllerState, PipelineConfig, ScrConfig

__all__ = ["StaticPolicy"]


class StaticPolicy(Policy):
    """Returns one configuration, always. The thing adaptation is measured against."""

    name = "static"

    def __init__(self, ann: AnnConfig, scr: ScrConfig, tuning_note: str = "") -> None:
        self.config = PipelineConfig(ann=ann, scr=scr)
        # How this config was chosen, carried into the run manifest so a reader
        # can check the baseline was tuned rather than guessed.
        self.tuning_note = tuning_note

    def decide(
        self,
        state: ControllerState,  # noqa: ARG002 - the point of a static policy
        constraints: Constraints,  # noqa: ARG002
    ) -> PipelineConfig:
        return self.config

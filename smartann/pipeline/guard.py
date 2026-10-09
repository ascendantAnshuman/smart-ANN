"""12.1 -- refuse to answer when the corpus does not contain the answer.

A retriever always returns its k nearest passages. It has no notion of "nothing
here is relevant", so a question the corpus cannot answer still produces five
confident-looking passages, and a generator handed them will usually write
something. The guard is the step that notices.

**The threshold is measured, not chosen.** Phase 11.4 ran twenty labelled
queries through this index and recorded the best similarity each one reached:

* questions the corpus *can* answer never scored below a floor
* questions with no relevant document at all never scored above a ceiling

and those two numbers do not overlap. Any threshold strictly inside that gap
refuses every unanswerable query in the evaluation set and answers every
answerable one. The guard takes the midpoint, carries both endpoints with it,
and reports which run they came from, so a reader can check the calibration
rather than trust the constant.

**It runs before generation, not after.** A refusal decided after the model has
written an answer costs the full generation and then throws it away. Deciding on
the retrieval scores costs nothing and saves the most expensive stage, which is
the difference between a guard and an apology.

**A gap can close.** The calibration is a property of one corpus and one
embedder. If the corpus grows until some unanswerable question scores above some
answerable one, there is no threshold that separates them and the honest report
is that this guard cannot work here -- which `calibrate` says rather than
returning a number that looks fine.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["REFUSAL", "GroundingGuard", "GuardVerdict"]

REFUSAL = "I couldn't find this in the on-device knowledge base."


@dataclass(frozen=True)
class GuardVerdict:
    """Whether the retrieval was good enough to answer from, and by how much."""

    grounded: bool
    best_score: float
    threshold: float
    margin: float  # best_score - threshold; negative when refused
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "grounded": self.grounded,
            "best_score": self.best_score,
            "threshold": self.threshold,
            "margin": self.margin,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GroundingGuard:
    """A similarity threshold, and the measurement it was calibrated from."""

    threshold: float
    answerable_floor: float
    unanswerable_ceiling: float
    calibrated_from: str
    # The corpus the separation was measured on. A threshold from one corpus
    # applied to another is the profile-table mistake in a new place: the
    # numbers stay plausible and the decision stops being measured.
    dataset: str = ""
    enabled: bool = True

    @property
    def gap(self) -> float:
        return self.answerable_floor - self.unanswerable_ceiling

    def verdict(self, best_score: float) -> GuardVerdict:
        if not self.enabled:
            return GuardVerdict(
                grounded=True,
                best_score=best_score,
                threshold=float("nan"),
                margin=float("nan"),
                reason="guard off: every retrieval is answered, whatever it scored",
            )
        if not math.isfinite(best_score):
            # Nothing was retrieved at all. Refusing is the only honest reply:
            # there is no evidence to be grounded in.
            return GuardVerdict(
                grounded=False,
                best_score=best_score,
                threshold=self.threshold,
                margin=float("nan"),
                reason="nothing was retrieved, so there is nothing to ground an answer in",
            )
        margin = best_score - self.threshold
        if margin >= 0:
            return GuardVerdict(
                grounded=True,
                best_score=best_score,
                threshold=self.threshold,
                margin=margin,
                reason=(
                    f"best passage scored {best_score:.3f}, at or above the "
                    f"{self.threshold:.3f} threshold -> answer from the corpus"
                ),
            )
        return GuardVerdict(
            grounded=False,
            best_score=best_score,
            threshold=self.threshold,
            margin=margin,
            reason=(
                f"best passage scored only {best_score:.3f}, below the "
                f"{self.threshold:.3f} threshold -> refuse rather than guess"
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "answerable_floor": self.answerable_floor,
            "unanswerable_ceiling": self.unanswerable_ceiling,
            "gap": self.gap,
            "calibrated_from": self.calibrated_from,
            "dataset": self.dataset,
            "enabled": self.enabled,
        }

    @classmethod
    def off(cls) -> GroundingGuard:
        return cls(
            threshold=float("nan"),
            answerable_floor=float("nan"),
            unanswerable_ceiling=float("nan"),
            calibrated_from="not calibrated; the guard is off",
            dataset="",
            enabled=False,
        )

    @classmethod
    def calibrate(cls, eval_json: Path) -> GroundingGuard:
        """Read the Phase 11.4 separation and take the midpoint of the gap.

        Raises when the two groups overlap. A guard with no gap to sit in cannot
        separate answerable from unanswerable at any threshold, and returning
        one anyway would hide that behind a plausible-looking number.
        """
        data = json.loads(Path(eval_json).read_text(encoding="utf-8"))
        sep = data["separation"]
        floor = float(sep["lowest_answerable"])
        ceiling = float(sep["highest_unanswerable"])
        if floor <= ceiling:
            raise ValueError(
                f"no gap to calibrate in: the lowest answerable query scored {floor:.3f} "
                f"and the highest unanswerable one {ceiling:.3f}. No single threshold "
                f"separates them, so this guard cannot work on this corpus."
            )
        n_answerable = int(data["paraphrases"]["n"]) + int(data["off_topic_in_corpus"]["n"])
        n_unanswerable = int(data["out_of_corpus"]["n"])
        return cls(
            # The midpoint, so that corpus drift in either direction has to
            # travel the same distance before the guard starts being wrong.
            threshold=(floor + ceiling) / 2.0,
            dataset=str(data.get("dataset", "")),
            answerable_floor=floor,
            unanswerable_ceiling=ceiling,
            calibrated_from=(
                f"{Path(eval_json).as_posix()}: {n_answerable} answerable queries scored "
                f"no lower than {floor:.3f}, {n_unanswerable} unanswerable ones no higher "
                f"than {ceiling:.3f}; threshold is the midpoint of that {floor - ceiling:.3f} gap"
            ),
        )

    @classmethod
    def load(cls, path: Path, dataset: str | None = None) -> GroundingGuard:
        """Read a calibrated guard, refusing one measured on another corpus.

        The separation between answerable and unanswerable is a property of a
        particular corpus and embedder. Carried across to a different index it
        would still produce confident refusals and confident answers, and
        nothing in the output would show that the number came from elsewhere.
        """
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        measured_on = str(d.get("dataset", ""))
        if dataset is not None and measured_on and measured_on != dataset:
            raise ValueError(
                f"{path} was calibrated on '{measured_on}' but this pipeline answers "
                f"from '{dataset}'. A refusal threshold from another corpus is not a "
                f"measured threshold. Recalibrate for '{dataset}', or run without a guard."
            )
        return cls(
            threshold=float(d["threshold"]),
            answerable_floor=float(d["answerable_floor"]),
            unanswerable_ceiling=float(d["unanswerable_ceiling"]),
            calibrated_from=str(d["calibrated_from"]),
            dataset=measured_on,
            enabled=bool(d.get("enabled", True)),
        )

    def write(self, path: Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")
        return p

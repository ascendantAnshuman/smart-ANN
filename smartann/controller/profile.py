"""The profile table: what each effort level actually costs and returns.

A lookup of measurements, deliberately not a model.

Phase 3 fitted a latency model that holds at 14.75% MAPE within the regime it
was fitted on and fails at 35.87% outside it, because `cpu_util` is censored --
12,469 contended samples pinned at exactly 1.000 while latency spanned 41x
(ADR-0033). A policy that *predicted* latency from that model would be
confidently wrong precisely when the device is under stress, which is the
situation the whole project exists to handle.

So a budget-aware policy reads measured percentiles instead. The table says what
`nprobe=8` cost and returned *when it was actually run*, and carries the
conditions it was measured under so a caller cannot silently apply quiet-machine
numbers to a loaded one.

The table is built from stored parquet, so it inherits the no-fabrication rule:
an entry exists because a query ran, or it does not exist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["ProfileEntry", "ProfileTable", "build_profile_table"]


@dataclass(frozen=True)
class ProfileEntry:
    """One effort level, as measured."""

    nprobe: int
    n_samples: int
    p50_ms: float
    p95_ms: float
    recall: float
    approx_tokens: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "nprobe": self.nprobe,
            "n_samples": self.n_samples,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "recall": self.recall,
            "approx_tokens": self.approx_tokens,
        }


@dataclass(frozen=True)
class ProfileTable:
    """Measured cost and return per effort level, plus the conditions of measurement."""

    entries: tuple[ProfileEntry, ...]
    dataset: str
    measured_under: str
    min_samples: int

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("a profile table with no entries cannot inform any decision")

    def best_within(self, budget_ms: float) -> ProfileEntry:
        """Highest-recall entry whose measured p95 fits the budget.

        Falls back to the cheapest entry when nothing fits, rather than raising:
        a budget too tight for any configuration is a real operating condition,
        and the honest response is minimum effort plus a violation, not a crash.
        """
        fits = [e for e in self.entries if e.p95_ms <= budget_ms]
        if fits:
            return max(fits, key=lambda e: (e.recall, -e.p95_ms))
        return min(self.entries, key=lambda e: e.p95_ms)

    def entry_for(self, nprobe: int) -> ProfileEntry | None:
        """The measured row for one effort level, or None when it was not measured.

        None rather than the nearest row: a report that named a level and then
        printed a neighbour's p95 and recall beside it would be describing a
        configuration nobody ran.
        """
        for e in self.entries:
            if e.nprobe == nprobe:
                return e
        return None

    def pareto_front(self) -> tuple[ProfileEntry, ...]:
        """Entries not dominated on both latency and recall.

        An entry is dominated when another is at least as fast AND at least as
        accurate. Dominated entries are never worth choosing, so a policy that
        searches the front searches a strictly smaller set for the same answer.
        """
        front: list[ProfileEntry] = []
        for e in sorted(self.entries, key=lambda x: x.p95_ms):
            if not any(o.p95_ms <= e.p95_ms and o.recall >= e.recall and o is not e for o in front):
                front.append(e)
        return tuple(front)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "measured_under": self.measured_under,
            "min_samples": self.min_samples,
            "entries": [e.as_dict() for e in self.entries],
            "pareto_front_nprobe": [e.nprobe for e in self.pareto_front()],
        }

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")
        return p

    @classmethod
    def read(cls, path: str | Path) -> ProfileTable:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            entries=tuple(ProfileEntry(**e) for e in d["entries"]),
            dataset=d["dataset"],
            measured_under=d["measured_under"],
            min_samples=d["min_samples"],
        )


def build_profile_table(
    frame: Any,
    dataset: str,
    measured_under: str,
    min_samples: int = 100,
) -> ProfileTable:
    """Group recorded queries by `nprobe` and summarise each level.

    Levels with fewer than `min_samples` observations are dropped rather than
    reported with wide error: a p95 over twelve samples is the second-largest of
    twelve, and a policy choosing on that would be choosing on noise. Dropping is
    visible -- the entry is absent -- where a noisy estimate is not.
    """
    entries = []
    for nprobe, group in frame.groupby("cfg_ann_nprobe"):
        if len(group) < min_samples:
            continue
        lat = group["t_total"].to_numpy(dtype=np.float64)
        rec = group["recall_at_k"].to_numpy(dtype=np.float64)
        tok = group["context_tokens"].to_numpy(dtype=np.float64)
        entries.append(
            ProfileEntry(
                nprobe=int(nprobe),
                n_samples=len(group),
                p50_ms=float(np.percentile(lat, 50)),
                p95_ms=float(np.percentile(lat, 95)),
                recall=float(np.nanmean(rec)) if np.isfinite(rec).any() else float("nan"),
                approx_tokens=float(np.nanmean(tok)) if np.isfinite(tok).any() else 0.0,
            )
        )
    if not entries:
        raise ValueError(
            f"no effort level had {min_samples}+ samples; the table would be noise. "
            f"Run the profiling sweep for longer."
        )
    return ProfileTable(
        entries=tuple(sorted(entries, key=lambda e: e.nprobe)),
        dataset=dataset,
        measured_under=measured_under,
        min_samples=min_samples,
    )

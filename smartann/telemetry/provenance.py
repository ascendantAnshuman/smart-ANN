"""What ran on the handset, and what did not.

This project is careful to label every host number as a host number: the profile
table says "THIS LAPTOP, not the handset", the terminal report ends with "Laptop
numbers, not device numbers", and every replayed device state carries the trace
it came from. That discipline is right, and on its own it gives a reader the
wrong impression -- because nothing anywhere said what *was* measured on real
hardware. Someone reading the demonstration would reasonably conclude the whole
thing is a laptop simulation with some telemetry sprinkled in, which is a worse
description than the truth.

So this module answers the other half of the question, and answers it from the
recorded files rather than from a sentence someone typed. `device_evidence`
counts what is actually in `results/device/`: how many queries ran on the phone,
over how long, at which effort levels, and which runs had a controller deciding
on-device. Those numbers move when the runs move.

**The stage table is a constant, and deliberately so.** Which pipeline stages
ran on the handset is a fact about the device scripts, not something recoverable
from their output -- a query record cannot say that no embedder was imported.
Writing it down and then letting it rot is the obvious failure, so
`tests/test_provenance.py` asserts each claim against the scripts themselves:
that the device scripts load pre-computed query vectors and import no embedder
or generator, and that the C5 script does construct the device policy. If
someone later runs generation on the phone, the test fails until this table is
corrected.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "PIPELINE_STAGES",
    "DeviceEvidence",
    "DeviceRun",
    "PipelineStage",
    "device_evidence",
]


@dataclass(frozen=True)
class PipelineStage:
    """One stage of the query path, and where it has actually been measured."""

    name: str
    on_device: bool
    evidence: str

    @property
    def where(self) -> str:
        return "handset" if self.on_device else "host only"


# Ordered as the query path runs. Each `evidence` string names what can be
# checked, so a reader can verify the row rather than trust it.
PIPELINE_STAGES: tuple[PipelineStage, ...] = (
    PipelineStage(
        "embed the query",
        False,
        "scripts/device_trace.py loads pre-computed queries.npy and imports no embedder; "
        "gte-small has never run in Termux",
    ),
    PipelineStage(
        "probe the centroid graph",
        True,
        "EcoVectorIndex.probe, in Termux, on the handset's own rebuilt centroid graph (ADR-0050)",
    ),
    PipelineStage(
        "controller decides the effort level",
        True,
        "DeviceStatePolicy in scripts/device_c5.py, reading the phone's thermal zones and "
        "battery; its reasons are recorded per query",
    ),
    PipelineStage(
        "search the chosen clusters",
        True,
        "EcoVectorIndex.finish, the same code path the host runs",
    ),
    PipelineStage(
        "reduce the context (SCR)",
        False,
        "no device run enabled SCR; the recorded runs are retrieval only",
    ),
    PipelineStage(
        "generate the answer",
        False,
        "no device script imports a generator; flan-t5-base has never run on the handset",
    ),
)


@dataclass(frozen=True)
class DeviceRun:
    """One recorded run that executed on the phone."""

    name: str
    n_queries: int
    minutes: float
    # From a bounded sample of the run's records, not every line -- see _scan.
    nprobe_levels: tuple[int, ...]
    ran_controller: bool
    scored_recall: bool
    path: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_queries": self.n_queries,
            "minutes": self.minutes,
            "nprobe_levels": list(self.nprobe_levels),
            "ran_controller": self.ran_controller,
            "scored_recall": self.scored_recall,
        }


@dataclass(frozen=True)
class DeviceEvidence:
    """Every recorded handset run, summarised."""

    runs: tuple[DeviceRun, ...]
    device: str

    @property
    def total_queries(self) -> int:
        return sum(r.n_queries for r in self.runs)

    @property
    def total_minutes(self) -> float:
        return sum(r.minutes for r in self.runs if math.isfinite(r.minutes))

    @property
    def controller_queries(self) -> int:
        """Queries where the policy itself decided on the phone.

        Separated from the total because it is the stronger claim: retrieval
        running on a handset is unremarkable, a control loop reading that
        handset's own sensors and acting on them is the contribution.
        """
        return sum(r.n_queries for r in self.runs if r.ran_controller)

    @property
    def any(self) -> bool:
        return bool(self.runs)

    def headline(self) -> str:
        if not self.any:
            return "No handset runs are on disk, so nothing here was measured on a phone."
        device = f" on the {self.device}" if self.device else ""
        line = (
            f"{self.total_queries:,} queries executed{device} across {len(self.runs)} recorded runs"
        )
        if self.controller_queries:
            line += f", {self.controller_queries:,} of them with the controller deciding on-device"
        return line + "."

    def as_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "total_queries": self.total_queries,
            "total_minutes": self.total_minutes,
            "controller_queries": self.controller_queries,
            "runs": [r.as_dict() for r in self.runs],
            "stages": [
                {"stage": s.name, "on_device": s.on_device, "evidence": s.evidence}
                for s in PIPELINE_STAGES
            ],
            "how_counted": (
                "query counts and durations are read from results/device/*_queries.jsonl "
                "and their meta/summary files. The stage table is a constant, checked "
                "against the device scripts by tests/test_provenance.py."
            ),
        }


def _minutes_for(stem: str, device_dir: Path) -> float:
    """Duration from whichever sidecar the run wrote, or NaN if neither exists."""
    for suffix, key in (("_meta.json", "minutes_actual"), ("_summary.json", "minutes")):
        p = device_dir / f"{stem}{suffix}"
        if not p.exists():
            continue
        try:
            value = json.loads(p.read_text(encoding="utf-8")).get(key)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return math.nan


SAMPLE_LINES = 500


def _scan(path: Path, sample: int = SAMPLE_LINES) -> tuple[int, tuple[int, ...], bool, bool]:
    """Count the run's queries and read off what its records prove.

    The count is exact -- it is a claim about how much ran on the phone, so it
    is not sampled. The *fields* are read from the first `sample` records only,
    because parsing 125,000 JSON objects takes about 600 ms and this is called
    from the query path. The field set is uniform within a run (one script wrote
    every line), and 500 records covers every level of a round-robin sweep many
    times over, so sampling costs nothing real here and makes the call free.

    A record carrying `reasons` or `arm` means a policy chose that query's
    effort on the phone; a record carrying `recall_at_k` means recall was scored
    there. Both are read from the data rather than assumed from the run's name.
    """
    n = 0
    levels: set[int] = set()
    ran_controller = False
    scored_recall = False
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            n += 1
            if n > sample:
                continue  # keep counting, stop parsing
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a run killed mid-write leaves a partial last line
            if isinstance(row.get("nprobe"), int):
                levels.add(int(row["nprobe"]))
            if "reasons" in row or "arm" in row:
                ran_controller = True
            if row.get("recall_at_k") is not None:
                scored_recall = True
    return n, tuple(sorted(levels)), ran_controller, scored_recall


def device_evidence(device_dir: Path, profile_table: Path | None = None) -> DeviceEvidence:
    """Summarise every recorded handset run found on disk.

    `profile_table` is read only for the device name, which the handset's own
    profile table records in its `measured_under` field. Absent, the name is
    simply left empty rather than guessed at.
    """
    device_dir = Path(device_dir)
    runs: list[DeviceRun] = []
    if device_dir.is_dir():
        for path in sorted(device_dir.glob("*_queries.jsonl")):
            stem = path.name[: -len("_queries.jsonl")]
            n, levels, controller, recall = _scan(path)
            if n == 0:
                continue
            runs.append(
                DeviceRun(
                    name=stem,
                    n_queries=n,
                    minutes=_minutes_for(stem, device_dir),
                    nprobe_levels=levels,
                    ran_controller=controller,
                    scored_recall=recall,
                    path=path,
                )
            )

    device = ""
    if profile_table is not None and Path(profile_table).exists():
        try:
            under = str(
                json.loads(Path(profile_table).read_text(encoding="utf-8")).get(
                    "measured_under", ""
                )
            )
        except (OSError, json.JSONDecodeError):
            under = ""
        # The handset table opens with the device name, e.g.
        # "vivo I2304, on-device, 25-min trace, ...". Anything else is a host
        # table and must not contribute a device name.
        if under and "on-device" in under:
            device = under.split(",")[0].strip()

    return DeviceEvidence(runs=tuple(runs), device=device)

"""11.3 -- the record of how one answer was produced, and at what cost.

An `AskResult` already carries what the pipeline decided and what each stage
cost. A `QueryTrace` is that result plus the three things a demonstration has to
be able to show and an `AskResult` does not carry:

* **where the resource readings came from.** A battery percentage means nothing
  without knowing whether it was read from this laptop, replayed from a recorded
  handset trace, or typed in as a hypothetical. The snapshot carries its own
  provenance so no downstream view can present one as another.
* **how many bytes left the device.** Claimed to be zero everywhere in this
  project, and therefore measured rather than asserted. `NetworkWitness` counts
  this process's own outbound socket traffic across the query.
* **an energy estimate, labelled as an estimate.** No energy sensor exists on
  this laptop (ADR-0007, ADR-0008). What exists is a coefficient measured on the
  handset, and applying it to a cluster count measured here produces an estimate
  of what the phone *would* have spent on this retrieval. That sentence is
  carried in the trace beside the number, because the number alone looks like a
  measurement.

**Nothing here recomputes anything.** Every field is copied from the objects the
query path already produced, so a trace cannot disagree with the run it
describes. The one exception is the energy estimate, which is a multiplication
that states its own two inputs.
"""

from __future__ import annotations

import json
import math
import socket
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from smartann.pipeline.ask import AskResult

__all__ = [
    "ControllerDecision",
    "EnergyEstimate",
    "NetworkWitness",
    "QueryTrace",
    "ResourceSnapshot",
    "RetrievedRef",
    "StageCost",
    "doc_slugs",
    "from_ask_result",
]

# Where a reading was taken. Three values, never conflated.
SOURCE_LIVE = "live sensor"
SOURCE_REPLAY = "trace-replay"
SOURCE_HYPOTHETICAL = "hypothetical"
SOURCE_ABSENT = "not measured"


# ---------------------------------------------------------------------------
# Measuring what left the device
# ---------------------------------------------------------------------------
class NetworkWitness:
    """Counts bytes this process sends while the block runs.

    **Why not psutil's interface counters.** Those are machine-wide. On this
    development machine OneDrive alone moves megabytes while a query runs, so a
    machine-wide delta would report traffic the pipeline did not cause and the
    "0 bytes left the device" claim would be unprovable on the very machine the
    demonstration runs on. Patching the socket class counts what *this process*
    sent, which is the thing being claimed.

    **It counts, it does not block.** A witness that refused connections would
    change the behaviour it is supposed to observe. Phase 12.5's offline proof
    is a separate, stricter instrument that blocks, and the two are kept apart so
    that a trace never silently becomes a test.

    Connection attempts are counted separately from bytes. A connection that is
    opened and sends nothing has still told a remote host that this device is
    here, and a byte count of zero would hide it.
    """

    _METHODS = ("send", "sendall", "sendto")

    def __init__(self) -> None:
        self.bytes_sent = 0
        self.connections = 0
        self._saved: dict[str, Any] = {}

    def __enter__(self) -> NetworkWitness:
        witness = self

        def wrap_send(name: str) -> Any:
            original = getattr(socket.socket, name)

            def counted(self: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
                try:
                    witness.bytes_sent += len(data)
                except TypeError:  # an object with no length is still a send
                    witness.bytes_sent += 1
                return original(self, data, *args, **kwargs)

            return counted

        for name in self._METHODS:
            self._saved[name] = getattr(socket.socket, name)
            setattr(socket.socket, name, wrap_send(name))

        original_connect = socket.socket.connect

        def counted_connect(self: Any, address: Any, *args: Any, **kwargs: Any) -> Any:
            witness.connections += 1
            return original_connect(self, address, *args, **kwargs)

        self._saved["connect"] = original_connect
        socket.socket.connect = counted_connect  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc: object) -> None:
        for name, original in self._saved.items():
            setattr(socket.socket, name, original)
        self._saved.clear()

    @property
    def silent(self) -> bool:
        return self.bytes_sent == 0 and self.connections == 0


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResourceSnapshot:
    """The device's condition at the moment the query started, and who said so."""

    battery_pct: float
    charging: bool
    temperature_c: float
    temperature_slope_c_per_s: float
    cpu_util: float
    mem_available_mb: float
    source: str  # one of the SOURCE_* constants
    source_detail: str  # the exact trace and sample, or the monitor's name
    measured_on: str  # "handset" | "this laptop" | "nothing -- typed by hand"

    @property
    def is_measured(self) -> bool:
        return self.source in (SOURCE_LIVE, SOURCE_REPLAY)

    def absent_fields(self) -> list[str]:
        """Which readings this platform could not give. Absence is not zero."""
        out = []
        for name in ("battery_pct", "temperature_c", "cpu_util", "mem_available_mb"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                out.append(name)
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "battery_pct": self.battery_pct,
            "charging": self.charging,
            "temperature_c": self.temperature_c,
            "temperature_slope_c_per_s": self.temperature_slope_c_per_s,
            "cpu_util": self.cpu_util,
            "mem_available_mb": self.mem_available_mb,
            "source": self.source,
            "source_detail": self.source_detail,
            "measured_on": self.measured_on,
            "is_measured": self.is_measured,
            "not_measured_fields": self.absent_fields(),
        }


@dataclass(frozen=True)
class ControllerDecision:
    """What the controller chose, and the one line that explains why."""

    policy: str
    nprobe: int
    k: int
    budget_ms: float | None
    baseline_nprobe: int | None
    derated: bool
    reason: str
    signals_used: tuple[str, ...]
    overhead_us: float
    table_measured_under: str
    level_recall: float
    level_p95_ms: float
    recall_given_up: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "chosen_nprobe": self.nprobe,
            "k": self.k,
            "budget_ms": self.budget_ms,
            "baseline_nprobe": self.baseline_nprobe,
            "derated": self.derated,
            "reason": self.reason,
            "signals_used": list(self.signals_used),
            "decision_cost_us": self.overhead_us,
            "level_recall_measured_under": self.table_measured_under,
            "level_recall": self.level_recall,
            "level_p95_ms": self.level_p95_ms,
            "recall_given_up_vs_baseline": self.recall_given_up,
            "note": (
                "level_recall describes this EFFORT LEVEL over the evaluation set, not "
                "this query. A typed question has no ground truth to score against."
            ),
        }


@dataclass(frozen=True)
class RetrievedRef:
    """One retrieved chunk, as a citation target.

    `doc_id` is the document the chunk came from, which is what a reader cites;
    `chunk_id` is what the index returned. Both are kept because two chunks of
    one document are two retrievals of the same source, and a trace that showed
    only chunk ids would hide that.
    """

    rank: int
    chunk_id: int
    doc_id: str
    title: str
    score: float
    store: str
    chars: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "score": self.score,
            "store": self.store,
            "chars": self.chars,
        }


@dataclass(frozen=True)
class StageCost:
    name: str
    ms: float
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"stage": self.name, "ms": self.ms, "detail": self.detail}


@dataclass(frozen=True)
class EnergyEstimate:
    """Energy for this query. An estimate, and it says so in every field."""

    mj: float
    label: str  # always "estimate" here; there is no energy sensor to say otherwise
    method: str
    coefficient_source: str
    covers: str
    excludes: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "mj": self.mj,
            "label": self.label,
            "method": self.method,
            "coefficient_source": self.coefficient_source,
            "covers": self.covers,
            "excludes": self.excludes,
        }


@dataclass(frozen=True)
class QueryTrace:
    """Everything about one answered question, in one serialisable record."""

    query: str
    timestamp: str
    dataset: str
    index_root: str
    generator: str
    resources: ResourceSnapshot
    controller: ControllerDecision
    clusters_probed: int
    vectors_scanned: int
    chunks: tuple[RetrievedRef, ...]
    stages: tuple[StageCost, ...]
    total_ms: float
    energy: EnergyEstimate
    answer: str
    answer_declined: bool
    cited_chunk_ids: tuple[int, ...]
    best_score: float
    network_bytes_sent: int
    network_connections: int
    context_chars: int
    prompt_tokens: int
    generated_tokens: int
    truncated: bool
    # 12.1 / 12.2. None means the feature did not run, which is not the same as
    # it running and deciding to do nothing.
    guard: dict[str, Any] | None = None
    cache: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def sent_nothing(self) -> bool:
        return self.network_bytes_sent == 0 and self.network_connections == 0

    @property
    def refused(self) -> bool:
        return self.guard is not None and not bool(self.guard.get("grounded", True))

    @property
    def cache_hit(self) -> bool:
        return self.cache is not None and bool(self.cache.get("hit", False))

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "timestamp": self.timestamp,
            "dataset": self.dataset,
            "index_root": self.index_root,
            "generator": self.generator,
            "resources": self.resources.as_dict(),
            "controller": self.controller.as_dict(),
            "retrieval": {
                "clusters_probed": self.clusters_probed,
                "vectors_scanned": self.vectors_scanned,
                "chunks": [c.as_dict() for c in self.chunks],
                "best_score": self.best_score,
            },
            "latency_ms": {
                "stages": [s.as_dict() for s in self.stages],
                "total": self.total_ms,
            },
            "energy": self.energy.as_dict(),
            "answer": {
                "text": self.answer,
                "declined": self.answer_declined,
                "cited_chunk_ids": list(self.cited_chunk_ids),
                "context_chars": self.context_chars,
                "prompt_tokens": self.prompt_tokens,
                "generated_tokens": self.generated_tokens,
                "truncated": self.truncated,
            },
            "grounding": self.guard
            if self.guard is not None
            else {"ran": False, "note": "no grounding guard was configured for this query"},
            "cache": self.cache
            if self.cache is not None
            else {"hit": False, "ran": False, "note": "the semantic cache was not enabled"},
            "network": {
                "bytes_sent": self.network_bytes_sent,
                "connections_opened": self.network_connections,
                "sent_nothing": self.sent_nothing,
                "how_measured": (
                    "this process's own socket sends, counted across the query. Not a "
                    "machine-wide counter, which would include other applications."
                ),
            },
            **({"extra": self.extra} if self.extra else {}),
        }

    def write(self, directory: Path) -> Path:
        """Save as JSON, named so traces sort chronologically and never collide."""
        directory.mkdir(parents=True, exist_ok=True)
        stamp = self.timestamp.replace(":", "").replace("-", "").replace("+0000", "")
        slug = "".join(ch if ch.isalnum() else "-" for ch in self.query.lower())[:48].strip("-")
        path = directory / f"{stamp}_{slug or 'query'}.json"
        n = 1
        while path.exists():
            path = directory / f"{stamp}_{slug or 'query'}_{n}.json"
            n += 1
        path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------
NO_ENERGY = EnergyEstimate(
    mj=math.nan,
    label="not measured",
    method="no energy model on disk; nothing was estimated and nothing is reported",
    coefficient_source="",
    covers="",
    excludes="",
)


def estimate_energy(n_cluster_loads: int, model_path: Path) -> EnergyEstimate:
    """Retrieval energy for this query, from the handset's measured coefficient.

    Phase 3.5 regressed the phone's measured power against its cluster-access
    rate and recorded `mj_per_cluster` in `results/energy_model.json`. Multiplying
    that by the clusters this query actually opened gives what the retrieval
    would have cost on the phone.

    Three things keep this from being read as a measurement. It is labelled an
    estimate; the coefficient's own goodness of fit travels with it, and it is
    weak (r2 ~ 0.16, so cluster rate explains a minority of the variance in
    power); and the generator's energy is excluded entirely rather than guessed
    at, because nothing in this project has ever measured it.
    """
    if not model_path.exists():
        return NO_ENERGY
    try:
        model = json.loads(model_path.read_text(encoding="utf-8"))
        attribution = model["attribution"]
        mj_per_cluster = float(attribution["mj_per_cluster"])
        r2 = float(attribution.get("r2", math.nan))
    except (OSError, KeyError, ValueError, TypeError):
        return NO_ENERGY

    return EnergyEstimate(
        mj=mj_per_cluster * n_cluster_loads,
        label="estimate",
        method=(
            f"{n_cluster_loads} cluster loads x {mj_per_cluster:.3f} mJ per cluster. "
            f"The cluster count is measured on this run; the coefficient is the "
            f"handset's, so this is what the PHONE would spend on this retrieval."
        ),
        coefficient_source=(
            f"measured: device power regressed on cluster-access rate over the Phase 3.5 "
            f"energy trace (results/energy_model.json), r2 = {r2:.3f} -- a weak fit, so "
            f"treat this as an order of magnitude rather than a figure"
        ),
        covers="retrieval only: opening and scanning clusters",
        excludes=(
            "embedding and generation. No energy sensor exists on this machine "
            "(ADR-0007/0008) and the phone traces never isolated the generator, so "
            "there is no measurement to base them on and none is invented."
        ),
    )


# ---------------------------------------------------------------------------
# Building one from a finished query
# ---------------------------------------------------------------------------
def _reason_line(r: AskResult) -> str:
    """One sentence a reader can follow, built from what the policy reported."""
    signals = "; ".join(r.policy_reasons) if r.policy_reasons else ""
    if r.derated and r.baseline_nprobe is not None:
        lost = (
            f", giving up {r.recall_given_up:.4f} measured recall"
            if math.isfinite(r.recall_given_up)
            else ""
        )
        return (
            f"{signals or 'device state crossed a threshold'} -> cut effort: "
            f"nprobe {r.baseline_nprobe} -> {r.ann.nprobe}{lost}"
        )
    if r.baseline_nprobe is not None:
        return (
            f"{signals or 'no threshold crossed'} -> no derate: nprobe {r.ann.nprobe}, "
            f"which is what the budget alone allowed"
        )
    if r.budget_ms is not None:
        return (
            f"budget {r.budget_ms:.0f} ms -> highest measured level that fits: "
            f"nprobe {r.ann.nprobe}"
        )
    if r.device_signals_absent:
        return (
            f"every device signal was NaN on this platform, so the policy declined to "
            f"act; nprobe {r.ann.nprobe} as given"
        )
    return f"no budget and no measured table -> fixed nprobe {r.ann.nprobe} as given"


def _snapshot(r: AskResult) -> ResourceSnapshot:
    d = r.device or {}

    def get(name: str) -> float:
        value = d.get(name, math.nan)
        return float(value) if isinstance(value, (int, float)) else math.nan

    if r.device is None:
        source, measured_on = SOURCE_ABSENT, "nothing -- no device state was attached"
    elif not r.device_is_measured:
        source, measured_on = SOURCE_HYPOTHETICAL, "nothing -- typed by hand"
    elif "trace" in r.device_source.lower() or "replay" in r.device_source.lower():
        source, measured_on = SOURCE_REPLAY, "handset (recorded trace)"
    else:
        source, measured_on = SOURCE_LIVE, "this laptop"

    return ResourceSnapshot(
        battery_pct=get("battery_pct"),
        charging=bool(d.get("charging", False)),
        temperature_c=get("temperature_c"),
        temperature_slope_c_per_s=get("temperature_slope_c_per_s"),
        cpu_util=get("cpu_util"),
        mem_available_mb=get("mem_available_mb"),
        source=source,
        source_detail=r.device_source or "none",
        measured_on=measured_on,
    )


def doc_slugs(chunks_db: Path) -> dict[int, str]:
    """chunk_id -> the document slug, read straight from the corpus database.

    Read here rather than carried on `RetrievedChunk` so that `ask.py` does not
    have to change shape for a field only the trace needs. An unreadable or
    absent database yields an empty mapping, and the trace then reports no doc
    id rather than a guessed one.
    """
    import sqlite3

    if not Path(chunks_db).exists():
        return {}
    try:
        conn = sqlite3.connect(str(chunks_db))
        try:
            rows = conn.execute(
                "SELECT m.chunk_id, d.file_path FROM metadata m "
                "JOIN documents d ON d.doc_id = m.doc_id"
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return {}
    return {int(cid): str(path).split("/", 1)[-1] for cid, path in rows}


def from_ask_result(
    r: AskResult,
    *,
    network_bytes_sent: int = 0,
    network_connections: int = 0,
    energy_model_path: Path = Path("results/energy_model.json"),
    doc_ids: dict[int, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> QueryTrace:
    """Assemble the trace from a finished query. Copies; never recomputes."""
    entry = r.profile_entry
    slugs = doc_ids or {}
    return QueryTrace(
        query=r.question,
        timestamp=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+0000"),
        dataset=r.dataset,
        index_root=str(r.index_root),
        generator=r.generator_name,
        resources=_snapshot(r),
        controller=ControllerDecision(
            policy=r.policy_name,
            nprobe=r.ann.nprobe,
            k=len(r.chunks),
            budget_ms=r.budget_ms,
            baseline_nprobe=r.baseline_nprobe,
            derated=r.derated,
            reason=_reason_line(r),
            signals_used=tuple(r.policy_reasons),
            overhead_us=r.controller_overhead_us,
            table_measured_under=r.table_measured_under,
            level_recall=entry.recall if entry else math.nan,
            level_p95_ms=entry.p95_ms if entry else math.nan,
            recall_given_up=r.recall_given_up,
        ),
        clusters_probed=r.n_cluster_loads,
        vectors_scanned=r.n_dist_exact,
        chunks=tuple(
            RetrievedRef(
                rank=c.rank,
                chunk_id=c.chunk_id,
                # Only index chunks have a slug in the corpus database; a
                # knowledge-base hit is identified by its title and URL instead.
                doc_id=slugs.get(c.chunk_id, "") if c.origin != "kb" else "",
                title=c.title,
                score=c.similarity,
                store=c.kind_label,
                chars=c.chars_out,
            )
            for c in r.chunks
        ),
        stages=tuple(StageCost(s.name, s.ms, s.detail) for s in r.stages),
        total_ms=r.total_ms,
        energy=estimate_energy(r.n_cluster_loads, energy_model_path),
        answer=r.answer,
        answer_declined=r.declined,
        guard=r.guard,
        cache=r.cache,
        cited_chunk_ids=tuple(c.chunk_id for c in (r.cited,) if c is not None),
        best_score=max((c.similarity for c in r.chunks), default=math.nan),
        network_bytes_sent=network_bytes_sent,
        network_connections=network_connections,
        context_chars=r.chars_out,
        prompt_tokens=r.prompt_tokens,
        generated_tokens=r.generated_tokens,
        truncated=r.truncated,
        extra=extra or {},
    )

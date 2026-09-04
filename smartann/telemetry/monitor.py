"""Device telemetry: the contract, and what a platform can honestly report.

Two rules shape this module.

**The query path must never block on telemetry.** Sampling runs on a background
thread at a fixed rate and publishes a single immutable ``DeviceState``; readers
take the latest published value and never wait for a sample in progress. A
monitor that stalls a query to read a sensor would put its own cost inside the
latency it exists to measure.

**A field a platform cannot measure is NaN, never a stand-in.** This machine can
read CPU utilisation and memory but not temperature and not energy (ADR-0007,
ADR-0008). The temptation is to substitute a modelled value so downstream code
has a number; that would make a modelled quantity indistinguishable from a
measured one in the parquet, which is the single most damaging thing this
project could do to its own results. Unavailable fields are NaN, the capability
table says which is which, and :func:`assert_manifest_consistent` makes a
manifest that claims otherwise impossible to write.
"""

from __future__ import annotations

import math
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from smartann.types import DeviceState

__all__ = [
    "Capability",
    "CapabilityTable",
    "DeviceMonitor",
    "assert_manifest_consistent",
    "unavailable_state",
]


@dataclass(frozen=True)
class Capability:
    """Whether one telemetry field can be measured here, and why not if not."""

    field: str
    available: bool
    source: str  # "psutil.cpu_percent", "unavailable", ...
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "available": self.available,
            "source": self.source,
            "reason": self.reason,
        }


class CapabilityTable:
    """What this platform can and cannot measure. Goes into the run manifest.

    A gate, not a log (ADR-0020). :func:`assert_manifest_consistent` refuses a
    manifest carrying a measured value for a field this table calls
    unavailable, so a false claim cannot survive to the parquet.
    """

    def __init__(self, capabilities: list[Capability]) -> None:
        self._caps = {c.field: c for c in capabilities}

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._caps.values())

    def is_available(self, field: str) -> bool:
        cap = self._caps.get(field)
        return bool(cap and cap.available)

    def unavailable_fields(self) -> list[str]:
        return sorted(f for f, c in self._caps.items() if not c.available)

    def as_dict(self) -> dict[str, Any]:
        return {f: c.as_dict() for f, c in sorted(self._caps.items())}

    def render(self) -> str:
        rows = ["  field                    available  source"]
        for c in sorted(self._caps.values(), key=lambda x: x.field):
            mark = "yes" if c.available else "NO "
            rows.append(f"  {c.field:<24} {mark:<10} {c.source}")
            if not c.available and c.reason:
                rows.append(f"  {'':<24} {'':<10} ({c.reason})")
        return "\n".join(rows)


def unavailable_state(timestamp: float | None = None) -> DeviceState:
    """A DeviceState with every field NaN. The honest zero value."""
    nan = math.nan
    return DeviceState(
        timestamp=timestamp if timestamp is not None else time.time(),
        temperature_c=nan,
        temperature_slope_c_per_s=nan,
        freq_ratio=nan,
        throttled=False,
        cpu_util=nan,
        mem_available_mb=nan,
        mem_pressure=nan,
        battery_pct=nan,
        charging=False,
        discharge_rate_ma=nan,
    )


class DeviceMonitor(ABC):
    """Samples device state on a background thread; readers never block."""

    def __init__(self, hz: float = 10.0) -> None:
        if hz <= 0:
            raise ValueError(f"sample rate must be positive, got {hz}")
        self.hz = hz
        self._latest: DeviceState = unavailable_state()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._samples = 0

    @abstractmethod
    def sample(self) -> DeviceState:
        """Read the platform once. Called only from the background thread."""

    @abstractmethod
    def capabilities(self) -> CapabilityTable:
        """What this platform can measure. Must not claim more than it reads."""

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> DeviceMonitor:
        if self._thread is not None:
            return self
        self._latest = self.sample()  # one synchronous sample so reads are valid
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="telemetry", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> DeviceMonitor:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _loop(self) -> None:
        period = 1.0 / self.hz
        while not self._stop.wait(period):
            try:
                state = self.sample()
            except Exception:
                # A telemetry failure must not kill the run or the query path.
                # The previous value stands and the sample count stops rising,
                # which is visible in the manifest.
                continue
            self._latest = state  # single reference assignment: atomic in CPython
            self._samples += 1

    # -- read path ---------------------------------------------------------
    def latest(self) -> DeviceState:
        """The most recent sample. Never blocks, never waits on a lock.

        ``DeviceState`` is frozen and the publish above is a single reference
        assignment, so a reader either sees the old state or the new one and
        never a half-updated one. That is why this needs no lock, and why the
        state must stay immutable.
        """
        return self._latest

    @property
    def n_samples(self) -> int:
        return self._samples


def assert_manifest_consistent(manifest: dict[str, Any], table: CapabilityTable) -> None:
    """Refuse a manifest that claims a measurement the platform cannot make.

    The check for ADR-0020: a capability table that is merely printed proves
    nothing. This makes it impossible for a run to carry, say, a measured energy
    value while the table records energy as unavailable -- which is exactly the
    error that would let a modelled number be read as a measured one.
    """
    caps = manifest.get("capabilities")
    if caps is None:
        raise ValueError("run manifest has no capability table; telemetry provenance is unknown")

    measured = manifest.get("measured", {})
    for field in table.unavailable_fields():
        value = measured.get(field)
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        raise ValueError(
            f"manifest reports a measured value for {field!r} ({value!r}) but the capability "
            f"table records it as unavailable on this platform. Either the table is wrong or "
            f"the value is modelled -- and a modelled value must never be recorded as measured."
        )

"""Replay a measured handset telemetry trace as device state for a live query.

**Why this exists.** The whole claim of this project is that retrieval effort
should follow the device's condition. On the development laptop that claim
cannot be shown: the host reports no temperature and no battery current, so
`DeviceStatePolicy` correctly declines to act and a demonstration shows a
controller doing nothing. Inventing a temperature to make it move would be the
"never substitute a modelled value for a measured one" rule broken in the most
visible place in the project.

The handset traces are the way out. `results/device/*_telemetry.jsonl` holds
what the real phone reported, sample by sample, during the recorded runs: skin
temperature, the hottest thermal zone, battery percentage, charge status and
current. Replaying those samples drives the controller from measurements that
were actually taken, on the device the project is about. Nothing here is
modelled, and nothing is generated.

**Which thermometer.** The phone exposes both a skin sensor and the hottest of
its thermal zones, and on the recorded runs they disagree by tens of degrees:
C5's skin peaked at 35.3 C while its hottest zone reached 70.4 C. `DeviceTiers`
derates at 42 C, so a policy reading the skin never trips on the very trace that
recorded a 70 C component, and a policy reading the hottest zone derates almost
throughout. That is the C5 result made visible rather than argued, which is why
the sensor is a parameter here instead of a decision buried in a loader.

**The slope is recomputed, not read.** Each trace records one `slope` field,
measured for the skin sensor. Asking for the hottest zone and handing back the
skin's slope would report one sensor's rate of change as another's. Both
sensors' slopes are therefore recomputed the same way, as a finite difference
between consecutive recorded samples -- arithmetic on measurements, not a model.
`tests/test_replay.py` checks the recomputed skin slope against the field the
monitor recorded, so the arithmetic is pinned to what the device itself did.

**Fields the trace does not carry stay NaN.** Clock ratio, CPU utilisation and
memory pressure were not recorded, so they are absent here, and every policy in
this project is written to decline on a NaN rather than read it as a healthy
zero. `throttled` is the one exception forced by its type: a bool cannot hold
"not measured", so it is False, which is also the value that makes
`DeviceStatePolicy` take no throttle action. No reason mentioning throttling can
therefore ever be produced from a replay, which is the honest outcome.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

from smartann.types import DeviceState

__all__ = [
    "SENSORS",
    "SENSOR_HOTTEST",
    "SENSOR_MEANING",
    "SENSOR_SKIN",
    "ReplayFeed",
    "TraceReplay",
    "TraceSample",
    "available_traces",
    "load_trace",
    "trace_path",
    "what_if_state",
]

# The two thermometers the handset exposes. Named rather than boolean because a
# reader of a report must see which one a number came from.
SENSOR_SKIN = "skin"
SENSOR_HOTTEST = "hottest"
SENSORS = (SENSOR_SKIN, SENSOR_HOTTEST)

_SENSOR_FIELD = {SENSOR_SKIN: "skin_c", SENSOR_HOTTEST: "hottest_c"}

# How a reader should understand each one, carried alongside the number so the
# page and the terminal cannot describe them differently.
SENSOR_MEANING = {
    SENSOR_SKIN: "the case the hand touches",
    SENSOR_HOTTEST: "the hottest thermal zone inside",
}


def _finite(value: object) -> float:
    """A trace value as a float, with anything unusable becoming NaN.

    The traces contain literal `NaN` for the first slope of a run, which
    `json.loads` accepts and produces as a float nan. Everything else that is
    not a number is absence too, and absence is NaN here rather than zero.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return math.nan
    return float(value)


@dataclass(frozen=True)
class TraceSample:
    """One recorded moment, with both thermometers kept side by side."""

    t: float
    skin_c: float
    hottest_c: float
    battery_pct: float
    charging: bool
    current_ua: float
    recorded_slope: float

    @property
    def elapsed_hint(self) -> float:
        return self.t


@dataclass(frozen=True)
class TraceReplay:
    """A measured trace, exposed as device state at any point along it."""

    name: str
    sensor: str
    samples: tuple[TraceSample, ...]
    slopes: tuple[float, ...]
    path: Path

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def duration_s(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return self.samples[-1].t - self.samples[0].t

    @property
    def sensor_field(self) -> str:
        return _SENSOR_FIELD[self.sensor]

    def temperature_at(self, i: int) -> float:
        s = self.samples[self._clamp(i)]
        return s.hottest_c if self.sensor == SENSOR_HOTTEST else s.skin_c

    @property
    def peak_c(self) -> float:
        values = [t for t in (self.temperature_at(i) for i in range(len(self))) if math.isfinite(t)]
        return max(values) if values else math.nan

    def _clamp(self, i: int) -> int:
        if not self.samples:
            raise IndexError("an empty trace has no samples to replay")
        return max(0, min(i, len(self.samples) - 1))

    def state_at(self, i: int) -> DeviceState:
        """The device state the handset was actually in at sample `i`."""
        idx = self._clamp(i)
        s = self.samples[idx]
        return DeviceState(
            timestamp=s.t,
            temperature_c=self.temperature_at(idx),
            temperature_slope_c_per_s=self.slopes[idx],
            freq_ratio=math.nan,
            # Not recorded by the probe. False is what the type allows and is
            # also the value that makes every policy take no throttle action,
            # so no decision can be attributed to a signal that was never read.
            throttled=False,
            cpu_util=math.nan,
            mem_available_mb=math.nan,
            mem_pressure=math.nan,
            battery_pct=s.battery_pct,
            charging=s.charging,
            discharge_rate_ma=s.current_ua / 1000.0,
        )

    def describe(self, i: int) -> str:
        """One line naming the trace, the sensor and where along it we are."""
        idx = self._clamp(i)
        pos = (self.samples[idx].t - self.samples[0].t) if len(self.samples) > 1 else 0.0
        return (
            f"measured trace {self.name}, sample {idx + 1}/{len(self.samples)} "
            f"at t+{pos:.0f}s, {self.sensor} sensor ({SENSOR_MEANING[self.sensor]})"
        )


def _slopes(samples: tuple[TraceSample, ...], sensor: str) -> tuple[float, ...]:
    """Finite differences of the chosen sensor, in C/s, NaN for the first.

    The first sample has nothing to difference against, so its slope is absent
    -- exactly as the monitor itself recorded it.
    """
    out: list[float] = [math.nan]
    for prev, cur in itertools.pairwise(samples):
        dt = cur.t - prev.t
        a = prev.hottest_c if sensor == SENSOR_HOTTEST else prev.skin_c
        b = cur.hottest_c if sensor == SENSOR_HOTTEST else cur.skin_c
        if dt <= 0 or not (math.isfinite(a) and math.isfinite(b)):
            out.append(math.nan)
        else:
            out.append((b - a) / dt)
    return tuple(out)


def load_trace(path: Path, sensor: str = SENSOR_HOTTEST, name: str | None = None) -> TraceReplay:
    """Read a recorded telemetry JSONL into a replayable trace."""
    if sensor not in SENSORS:
        raise ValueError(f"sensor must be one of {SENSORS}, got {sensor!r}")
    if not path.exists():
        raise FileNotFoundError(f"no telemetry trace at {path}")

    samples: list[TraceSample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # a run killed mid-write leaves a partial last line
        if not isinstance(row, dict) or "t" not in row:
            continue
        samples.append(
            TraceSample(
                t=_finite(row.get("t")),
                skin_c=_finite(row.get("skin_c")),
                hottest_c=_finite(row.get("hottest_c")),
                battery_pct=_finite(row.get("pct")),
                charging=str(row.get("status", "")).upper() == "CHARGING",
                current_ua=_finite(row.get("current_ua")),
                recorded_slope=_finite(row.get("slope")),
            )
        )
    if not samples:
        raise ValueError(f"{path} holds no telemetry samples")

    tup = tuple(samples)
    return TraceReplay(
        name=name or path.stem.replace("_telemetry", ""),
        sensor=sensor,
        samples=tup,
        slopes=_slopes(tup, sensor),
        path=path,
    )


class ReplayFeed:
    """Walks a trace, handing out one measured state per query.

    A demonstration asks several questions in a row, and the interesting thing
    about the recorded runs is that the device changes during them. Advancing a
    step per question replays that: the phone warms, and the controller's choice
    follows it. The stride spreads a whole trace across roughly `over_queries`
    questions, so a short session still reaches the hot end rather than spending
    every question in the first minute.
    """

    def __init__(self, trace: TraceReplay, over_queries: int = 12) -> None:
        self.trace = trace
        self.stride = max(1, len(trace) // max(1, over_queries))
        self.index = 0

    def next(self) -> tuple[DeviceState, str]:
        state = self.trace.state_at(self.index)
        label = self.trace.describe(self.index)
        self.index = min(self.index + self.stride, len(self.trace) - 1)
        return state, label


def available_traces(device_dir: Path) -> list[str]:
    """Names of the recorded traces on disk, newest run last."""
    if not device_dir.is_dir():
        return []
    return sorted(p.stem.replace("_telemetry", "") for p in device_dir.glob("*_telemetry.jsonl"))


def trace_path(device_dir: Path, name: str) -> Path:
    return device_dir / f"{name}_telemetry.jsonl"


def what_if_state(
    temperature_c: float = math.nan,
    battery_pct: float = math.nan,
    charging: bool = False,
    slope_c_per_s: float = math.nan,
) -> DeviceState:
    """A hypothetical device state, for asking what the controller WOULD do.

    Every caller of this must label the result as hypothetical. It is not a
    measurement and nothing derived from it may be reported as one; it exists so
    a reader can drive the policy to a state the recorded traces never visited
    and see the decision it produces. Fields left out stay NaN, so the policy
    declines to act on them rather than reading a zero as a cold, empty device.
    """
    return DeviceState(
        timestamp=0.0,
        temperature_c=float(temperature_c),
        temperature_slope_c_per_s=float(slope_c_per_s),
        freq_ratio=math.nan,
        throttled=False,
        cpu_util=math.nan,
        mem_available_mb=math.nan,
        mem_pressure=math.nan,
        battery_pct=float(battery_pct),
        charging=bool(charging),
        discharge_rate_ma=math.nan,
    )

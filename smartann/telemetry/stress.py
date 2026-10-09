"""12.3 -- drive the controller to a named device condition, on demand.

A demonstration cannot wait for a phone to get hot. These are named states that
can be injected so an audience sees the controller react within one query.

**Every mode says where its numbers came from, and they are not all the same.**
Three of the four below are built from real samples in the recorded handset
traces: the mode names a sample, and the state handed to the policy is that
sample's own readings. One is not, and cannot be.

**Why "hot phone at 44 C" is not a measurement here.** The handset exposes two
thermometers that disagree by tens of degrees, and 44 C is on neither scale in
the way the request implies:

* the *skin* sensor never exceeded 36.5 C across every recorded trace, so a
  44 C skin reading has never been observed on this device
* the *hottest internal zone* never went BELOW 45.2 C, even at idle, so 44 C is
  cooler than this phone's resting internal temperature

So `hot` uses the hottest moment actually recorded -- 70.4 C internal, from the
C5 trace -- which is a real state of a real phone and trips the policy's 46 C
tier honestly. `hot-44` is offered as well, because it is what was asked for,
and it is labelled a hypothetical everywhere it appears.

**Why "low battery" is a hypothetical.** Battery across every recorded trace
runs 63-80%. Nothing below that was ever measured, so a 12% state is typed. It
is still worth having: the policy's battery tiers are real code with real
thresholds, and showing what they do is honest as long as nobody calls the input
a reading.

**Why "battery saver" is a hypothetical too.** It is an operating-system mode,
not a sensor. What it does to a phone -- clock limits and background
restrictions -- reaches this policy as `throttled` and a reduced clock ratio, so
that is how it is expressed here, and the mapping is an interpretation rather
than a measurement.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smartann.types import DeviceState

__all__ = ["MODE_NAMES", "StressMode", "available_modes", "load_modes"]

MODE_NAMES = ("normal", "hot", "hot-44", "low-battery", "battery-saver")

MEASURED = "measured"
HYPOTHETICAL = "hypothetical"


@dataclass(frozen=True)
class StressMode:
    """One named device condition, with its provenance attached."""

    name: str
    state: DeviceState
    provenance: str  # MEASURED or HYPOTHETICAL
    detail: str
    label: str  # what to print as the device source

    @property
    def is_measured(self) -> bool:
        return self.provenance == MEASURED

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provenance": self.provenance,
            "detail": self.detail,
            "label": self.label,
            "temperature_c": self.state.temperature_c,
            "battery_pct": self.state.battery_pct,
            "charging": self.state.charging,
            "throttled": self.state.throttled,
        }


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return math.nan
    return float(value)


def _samples(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and "t" in row:
            out.append(row)
    return out


def _state_from(row: dict[str, Any], slope: float = math.nan) -> DeviceState:
    """A recorded sample, as the DeviceState the policy will read.

    Fields the probe did not record stay NaN, exactly as `telemetry/replay.py`
    leaves them, so no policy can act on a value the device never produced.
    """
    current = _finite(row.get("current_ua"))
    return DeviceState(
        timestamp=_finite(row.get("t")),
        temperature_c=_finite(row.get("hottest_c")),
        temperature_slope_c_per_s=slope,
        freq_ratio=math.nan,
        throttled=False,
        cpu_util=math.nan,
        mem_available_mb=math.nan,
        mem_pressure=math.nan,
        battery_pct=_finite(row.get("pct")),
        charging=str(row.get("status", "")).upper() == "CHARGING",
        discharge_rate_ma=current / 1000.0 if math.isfinite(current) else math.nan,
    )


def load_modes(device_dir: Path, trace: str = "c5") -> dict[str, StressMode]:
    """Build the named modes, sourcing from a recorded trace where one exists."""
    path = device_dir / f"{trace}_telemetry.jsonl"
    rows = _samples(path)
    hottest = [r for r in rows if math.isfinite(_finite(r.get("hottest_c")))]

    modes: dict[str, StressMode] = {}

    if hottest:
        coolest_row = min(hottest, key=lambda r: _finite(r["hottest_c"]))
        hottest_row = max(hottest, key=lambda r: _finite(r["hottest_c"]))

        modes["normal"] = StressMode(
            name="normal",
            state=_state_from(coolest_row),
            provenance=MEASURED,
            detail=(
                f"the coolest sample in the {trace} trace: "
                f"{_finite(coolest_row['hottest_c']):.1f} C internal, "
                f"battery {_finite(coolest_row.get('pct')):.0f}%"
            ),
            label=f"stress mode 'normal' -- measured sample from trace {trace}",
        )
        modes["hot"] = StressMode(
            name="hot",
            state=_state_from(hottest_row),
            provenance=MEASURED,
            detail=(
                f"the hottest moment ever recorded on this handset: "
                f"{_finite(hottest_row['hottest_c']):.1f} C internal, from the {trace} trace. "
                f"Not a typed 44 C -- this phone idles above that internally."
            ),
            label=f"stress mode 'hot' -- measured sample from trace {trace}",
        )
    else:
        # No trace on disk. The measured modes simply do not exist rather than
        # being filled in with plausible numbers.
        pass

    modes["hot-44"] = StressMode(
        name="hot-44",
        state=DeviceState(
            timestamp=0.0,
            temperature_c=44.0,
            temperature_slope_c_per_s=math.nan,
            freq_ratio=math.nan,
            throttled=False,
            cpu_util=math.nan,
            mem_available_mb=math.nan,
            mem_pressure=math.nan,
            battery_pct=math.nan,
            charging=False,
            discharge_rate_ma=math.nan,
        ),
        provenance=HYPOTHETICAL,
        detail=(
            "a typed 44 C. Never observed on this handset on either scale: skin "
            "peaked at 36.5 C across all traces, and the hottest internal zone never "
            "fell below 45.2 C. It sits between the policy's warm (42 C) and hot "
            "(46 C) tiers, so it shows the middle tier that no recorded trace visits."
        ),
        label="stress mode 'hot-44' -- HYPOTHETICAL, typed, not measured",
    )

    modes["low-battery"] = StressMode(
        name="low-battery",
        state=DeviceState(
            timestamp=0.0,
            temperature_c=math.nan,
            temperature_slope_c_per_s=math.nan,
            freq_ratio=math.nan,
            throttled=False,
            cpu_util=math.nan,
            mem_available_mb=math.nan,
            mem_pressure=math.nan,
            battery_pct=12.0,
            charging=False,
            discharge_rate_ma=math.nan,
        ),
        provenance=HYPOTHETICAL,
        detail=(
            "a typed 12%. Battery across every recorded trace runs 63-80%, so nothing "
            "this low was ever measured. Temperature is left absent rather than set to "
            "a comfortable value, so the derate shown is the battery tier alone."
        ),
        label="stress mode 'low-battery' -- HYPOTHETICAL, typed, not measured",
    )

    modes["battery-saver"] = StressMode(
        name="battery-saver",
        state=DeviceState(
            timestamp=0.0,
            temperature_c=math.nan,
            temperature_slope_c_per_s=math.nan,
            freq_ratio=0.6,
            throttled=True,
            cpu_util=math.nan,
            mem_available_mb=math.nan,
            mem_pressure=math.nan,
            battery_pct=18.0,
            charging=False,
            discharge_rate_ma=math.nan,
        ),
        provenance=HYPOTHETICAL,
        detail=(
            "an interpretation, not a reading. Battery saver is an operating-system "
            "mode; what it does to this policy is restrict clocks, which arrives as "
            "throttled=True and a reduced clock ratio. The 18% battery is the level at "
            "which Android offers to switch it on, and is a convention rather than a "
            "measurement."
        ),
        label="stress mode 'battery-saver' -- HYPOTHETICAL, an interpretation of an OS mode",
    )

    return modes


def available_modes(device_dir: Path, trace: str = "c5") -> list[str]:
    return list(load_modes(device_dir, trace))

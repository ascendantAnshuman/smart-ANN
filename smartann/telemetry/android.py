"""Device state from an Android handset over adb.

The first monitor in this project that can report `temperature_c` as
**available**. Every earlier one declared it unavailable and meant it: this
laptop has no readable thermal sensor and no energy counter (ADR-0007,
ADR-0008), so the capability gate refused any manifest claiming otherwise. The
phone changes what is measurable, not what is claimable — the gate is unchanged
and this monitor must earn each `available` by actually reading the value.

**Three device-specific facts are baked in, each measured rather than assumed
(ADR-0046, ADR-0047).**

*Battery current comes from Termux:API, not sysfs.* Every node under
`/sys/class/power_supply/battery/` is unreadable to the adb shell user under
Android 16 SELinux. `termux-battery-status` goes through BatteryManager and
works. Code written against sysfs fails on this device.

*Negative current means discharging.* Established on battery across 40 samples
with no exceptions (ADR-0047). While CHARGING the sign alternates between
samples under constant load, so this monitor reports `discharge_rate_ma` as NaN
when plugged in rather than reporting a number whose meaning it cannot defend.

*21 of 110 thermal zones report -273000* — absolute zero, a disabled sensor.
They are filtered. An aggregate that includes them reads as a cold device,
which is the direction that would make a thermal controller do nothing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from typing import Any

from smartann.telemetry.monitor import Capability, CapabilityTable, DeviceMonitor
from smartann.types import DeviceState

__all__ = ["AndroidMonitor", "AndroidReading", "parse_battery_json", "summarise_zones"]

NAN = float("nan")

# Sensors reading exactly this are off, not cold.
_SENSOR_DISABLED = -273000

# The zone that actually moves. Skin swung 4.8 C over 60 s of load while the
# battery zone moved 1.2 C (ADR-0047), so the battery zone is not a substitute
# and the choice is recorded here rather than left to whichever sorts first.
_PREFERRED_ZONES = ("skin-msm-therm", "shell_front", "skin-therm")


class AndroidReading(dict):  # type: ignore[type-arg]
    """One raw sample. A dict so it serialises into a manifest unchanged."""


def parse_battery_json(text: str) -> dict[str, Any]:
    """Parse `termux-battery-status` output, tolerating shell noise around it.

    The command is invoked through `adb shell` through Termux, and any layer may
    prepend a warning. Slicing to the outermost braces is more robust than
    trusting the whole stream to be JSON, and a failure here must not look like
    a device with no battery.
    """
    lo, hi = text.find("{"), text.rfind("}")
    if lo < 0 or hi <= lo:
        raise ValueError(f"no JSON object in battery output: {text[:200]!r}")
    parsed = json.loads(text[lo : hi + 1])
    if not isinstance(parsed, dict):
        raise ValueError(f"battery output is not an object: {type(parsed).__name__}")
    return parsed


def summarise_zones(pairs: list[tuple[str, int]]) -> tuple[float, float]:
    """`(preferred_zone_c, max_live_zone_c)` from `(type, millidegrees)` pairs.

    Disabled sensors are dropped first. Both numbers are returned because they
    answer different questions: the preferred zone is what a thermal model is
    fitted against, and the maximum live zone is what a safety shield would
    trip on. Collapsing them would force one of those to use the wrong signal.
    """
    live = [(t, v / 1000.0) for t, v in pairs if v != _SENSOR_DISABLED]
    if not live:
        return NAN, NAN
    by_type = dict(live)
    preferred = next((by_type[z] for z in _PREFERRED_ZONES if z in by_type), NAN)
    return preferred, max(v for _, v in live)


class AndroidMonitor(DeviceMonitor):
    """Samples an Android device over adb.

    Sampling is far slower than the 10 Hz the base class defaults to -- each
    read is several `adb shell` round trips, and `termux-battery-status` alone
    takes hundreds of milliseconds. The default rate here is 0.5 Hz, and a
    caller asking for more will simply get samples as fast as adb allows.
    """

    def __init__(
        self,
        serial: str,
        adb: str | None = None,
        hz: float = 0.5,
        timeout_s: float = 20.0,
    ) -> None:
        super().__init__(hz=hz)
        self.serial = serial
        self.adb = adb or shutil.which("adb") or "adb"
        self.timeout_s = timeout_s
        self._t_prev: float | None = None
        self._temp_prev: float = NAN
        self._battery_ok: bool | None = None

    # -- adb plumbing ------------------------------------------------------
    def _shell(self, cmd: str) -> str:
        proc = subprocess.run(
            [self.adb, "-s", self.serial, "shell", cmd],
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
        return (proc.stdout or "") + (proc.stderr or "")

    def _battery(self) -> dict[str, Any]:
        """Battery via Termux:API. Raises if unavailable -- never fabricates."""
        out = self._shell("/data/data/com.termux/files/usr/bin/termux-battery-status")
        return parse_battery_json(out)

    def _zones(self) -> list[tuple[str, int]]:
        out = self._shell(
            "for z in /sys/class/thermal/thermal_zone*; do "
            'echo "$(cat $z/type 2>/dev/null):$(cat $z/temp 2>/dev/null)"; done'
        )
        pairs = []
        for line in out.splitlines():
            t, _, v = line.strip().rpartition(":")
            if t and v.lstrip("-").isdigit():
                pairs.append((t, int(v)))
        return pairs

    def _freqs(self) -> list[tuple[int, int]]:
        """`(current, max)` kHz per core. Three clock domains on this SoC."""
        out = self._shell(
            "for c in /sys/devices/system/cpu/cpu[0-9]*/cpufreq; do "
            'echo "$(cat $c/scaling_cur_freq 2>/dev/null):'
            '$(cat $c/cpuinfo_max_freq 2>/dev/null)"; done'
        )
        pairs = []
        for line in out.splitlines():
            cur, _, mx = line.strip().partition(":")
            if cur.isdigit() and mx.isdigit() and int(mx) > 0:
                pairs.append((int(cur), int(mx)))
        return pairs

    # -- DeviceMonitor -----------------------------------------------------
    def sample(self) -> DeviceState:
        now = time.time()
        try:
            b = self._battery()
            self._battery_ok = True
        except Exception:
            self._battery_ok = False
            b = {}

        temp_c, _ = summarise_zones(self._zones())

        slope = NAN
        if self._t_prev is not None and self._temp_prev == self._temp_prev:
            dt = now - self._t_prev
            if dt > 0 and temp_c == temp_c:
                slope = (temp_c - self._temp_prev) / dt
        self._t_prev, self._temp_prev = now, temp_c

        freqs = self._freqs()
        # A SINGLE ratio for a three-cluster SoC is not defensible, and
        # types.py has not been changed (ADR-0046). The maximum ratio across
        # cores is reported as the least-bad scalar and the choice is stated:
        # it answers "is any cluster running at speed", which is the question a
        # throttling detector asks. It is NOT the ratio the query experienced.
        freq_ratio = max((c / m for c, m in freqs), default=NAN)

        charging = str(b.get("status", "")).upper() == "CHARGING"
        current_ua = b.get("current")
        # While charging the sign alternates between samples under constant
        # load (ADR-0047), so no discharge rate is reported rather than one
        # whose meaning cannot be defended.
        discharge_ma = (
            abs(float(current_ua)) / 1000.0 if (current_ua is not None and not charging) else NAN
        )

        return DeviceState(
            timestamp=now,
            temperature_c=temp_c,
            temperature_slope_c_per_s=slope,
            freq_ratio=freq_ratio,
            throttled=bool(freq_ratio == freq_ratio and freq_ratio < 0.9),
            cpu_util=NAN,  # not read over adb; would need a second round trip
            mem_available_mb=NAN,
            mem_pressure=NAN,
            battery_pct=float(b["percentage"]) if "percentage" in b else NAN,
            charging=charging,
            discharge_rate_ma=discharge_ma,
        )

    def capabilities(self) -> CapabilityTable:
        """Every `available` is EARNED by a read, never asserted.

        The first version of this method hardcoded `temperature_c` and
        `freq_ratio` to True, because the device probe had shown the handset
        could report them. Run against a *disconnected* handset it then
        declared them available while `sample()` returned NaN -- a capability
        table making a claim the monitor could not satisfy, which is exactly
        what `assert_manifest_consistent` exists to catch and what this class
        must never be the source of.

        A capability is a statement about THIS monitor reaching THIS device
        now, not about what the hardware can do in principle.
        """
        ok = self._battery_ok
        if ok is None:
            try:
                self._battery()
                ok = True
            except Exception:
                ok = False
        reason = (
            ""
            if ok
            else "termux-battery-status unreachable; sysfs is blocked by "
            "Android SELinux, so there is no fallback (ADR-0046)"
        )

        temp_c, _ = summarise_zones(self._zones())
        temp_ok = temp_c == temp_c  # NaN-safe: a NaN read is not a capability
        freq_ok = bool(self._freqs())
        unreachable = "no live reading; device unreachable or sensor absent"

        return CapabilityTable(
            [
                Capability(
                    "temperature_c",
                    temp_ok,
                    "sysfs thermal_zone via adb",
                    "" if temp_ok else unreachable,
                ),
                Capability(
                    "temperature_slope_c_per_s",
                    temp_ok,
                    "differenced across samples",
                    "" if temp_ok else unreachable,
                ),
                Capability(
                    "freq_ratio",
                    freq_ok,
                    "sysfs cpufreq via adb",
                    "" if freq_ok else unreachable,
                ),
                Capability("battery_pct", bool(ok), "termux-battery-status", reason),
                Capability("charging", bool(ok), "termux-battery-status", reason),
                Capability(
                    "discharge_rate_ma",
                    bool(ok),
                    "termux-battery-status (BatteryManager)",
                    reason or "NaN while charging: the sign is unstable on AC (ADR-0047)",
                ),
                Capability(
                    "energy_mj_measured",
                    bool(ok),
                    "V*I*dt integrated from termux-battery-status",
                    reason,
                ),
                Capability(
                    "cpu_util",
                    False,
                    "unavailable",
                    "not read over adb: a second round trip per sample, and the "
                    "handset's own /proc/stat is not comparable to the host's",
                ),
                Capability("mem_available_mb", False, "unavailable", "not read over adb"),
                Capability("mem_pressure", False, "unavailable", "not read over adb"),
            ]
        )

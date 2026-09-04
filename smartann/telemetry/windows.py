"""Telemetry on this development machine, which is Windows 11 on an i5-13420H.

What it can read: CPU utilisation, available memory, battery. What it cannot:
**temperature and energy**. WMI's thermal zone is access-denied without
elevation, there is no RAPL on Windows, and WSL2 exposes neither `thermal_zone*`
nor a populated `powercap` (ADR-0007). The Android phone is therefore the only
source of thermal and energy ground truth in this project (ADR-0008).

`freq_ratio` is probed rather than assumed: `psutil.cpu_freq()` exists on
Windows but frequently reports a constant, in which case it carries no
information about throttling and is declared unavailable rather than reported as
a number that never moves.
"""

from __future__ import annotations

import math
import time

import psutil

from smartann.telemetry.monitor import Capability, CapabilityTable, DeviceMonitor
from smartann.types import DeviceState

__all__ = ["WindowsMonitor", "probe_freq_ratio_usable"]


def probe_freq_ratio_usable(samples: int = 5, pause: float = 0.05) -> tuple[bool, str]:
    """Does cpu_freq() actually move on this machine?

    A field that reports the same value forever is not a measurement of
    frequency scaling; it is a constant wearing a sensor's name. Declaring it
    available would put a dead signal into the controller's feature vector.
    """
    try:
        readings = []
        for _ in range(samples):
            f = psutil.cpu_freq()
            if f is None or not f.max:
                return False, "psutil.cpu_freq() returned None or no max frequency"
            readings.append(f.current)
            time.sleep(pause)
    except Exception as exc:
        return False, f"psutil.cpu_freq() raised {type(exc).__name__}: {exc}"

    if len({round(r, 3) for r in readings}) <= 1:
        return False, f"cpu_freq() constant at {readings[0]:.0f} MHz across {samples} samples"
    return True, "psutil.cpu_freq()"


class WindowsMonitor(DeviceMonitor):
    """psutil-backed monitor. Reports NaN for everything it cannot read."""

    def __init__(self, hz: float = 10.0) -> None:
        super().__init__(hz=hz)
        self._freq_ok, self._freq_note = probe_freq_ratio_usable()
        self._freq_max = 0.0
        if self._freq_ok:
            f = psutil.cpu_freq()
            self._freq_max = float(f.max) if f and f.max else 0.0
        self._total_mb = psutil.virtual_memory().total / 2**20
        psutil.cpu_percent(interval=None)  # prime the delta-based counter

    def capabilities(self) -> CapabilityTable:
        return CapabilityTable(
            [
                Capability("cpu_util", True, "psutil.cpu_percent"),
                Capability("mem_available_mb", True, "psutil.virtual_memory"),
                Capability("mem_pressure", True, "psutil.virtual_memory"),
                Capability(
                    "freq_ratio",
                    self._freq_ok,
                    self._freq_note if self._freq_ok else "unavailable",
                    "" if self._freq_ok else self._freq_note,
                ),
                Capability(
                    "battery_pct",
                    psutil.sensors_battery() is not None,
                    "psutil.sensors_battery",
                    "" if psutil.sensors_battery() is not None else "no battery reported",
                ),
                Capability(
                    "temperature_c",
                    False,
                    "unavailable",
                    "WMI thermal zone is access-denied without elevation; psutil has no "
                    "sensors_temperatures on Windows. Phone only (ADR-0007, ADR-0008).",
                ),
                Capability(
                    "temperature_slope_c_per_s",
                    False,
                    "unavailable",
                    "derived from temperature_c, which is unavailable",
                ),
                Capability(
                    "throttled",
                    False,
                    "unavailable",
                    "requires frequency or thermal data this platform does not expose",
                ),
                Capability(
                    "energy_mj_measured",
                    False,
                    "unavailable",
                    "no RAPL on Windows and no powercap under WSL2; phone only (ADR-0008)",
                ),
                Capability(
                    "discharge_rate_ma",
                    False,
                    "unavailable",
                    "psutil.sensors_battery reports percentage and time-left, not current",
                ),
            ]
        )

    def sample(self) -> DeviceState:
        vm = psutil.virtual_memory()
        freq_ratio = math.nan
        if self._freq_ok and self._freq_max:
            f = psutil.cpu_freq()
            if f is not None:
                freq_ratio = float(f.current) / self._freq_max

        battery = psutil.sensors_battery()
        return DeviceState(
            timestamp=time.time(),
            temperature_c=math.nan,
            temperature_slope_c_per_s=math.nan,
            freq_ratio=freq_ratio,
            throttled=False,
            cpu_util=psutil.cpu_percent(interval=None) / 100.0,
            mem_available_mb=vm.available / 2**20,
            mem_pressure=1.0 - (vm.available / vm.total),
            battery_pct=float(battery.percent) if battery else math.nan,
            charging=bool(battery.power_plugged) if battery else False,
            discharge_rate_ma=math.nan,
        )

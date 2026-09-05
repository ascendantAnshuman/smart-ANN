"""The Android monitor: parsing, and capabilities that must be earned.

No device is required. `_shell` is the only thing that touches adb, so
overriding it exercises every decision this class makes against canned output
that is copied from the real handset (ADR-0046, ADR-0047) rather than invented.
"""

from __future__ import annotations

import math

import pytest

from smartann.telemetry.android import (
    AndroidMonitor,
    parse_battery_json,
    summarise_zones,
)

# Copied verbatim from the calibration run on the handset, discharging.
REAL_BATTERY = (
    '{"present":true,"technology":"Li-poly","health":"GOOD","plugged":"UNPLUGGED",'
    '"status":"DISCHARGING","temperature":31.4,"voltage":4280,"current":-334000,'
    '"percentage":96,"level":96,"scale":100,"charge_counter":1708000,"cycle":1158}'
)
CHARGING_BATTERY = REAL_BATTERY.replace("DISCHARGING", "CHARGING").replace(
    "UNPLUGGED", "PLUGGED_AC"
)


class FakeAndroid(AndroidMonitor):
    """AndroidMonitor with adb replaced by a lookup table."""

    def __init__(self, responses: dict[str, str]) -> None:
        super().__init__(serial="fake")
        self.responses = responses

    def _shell(self, cmd: str) -> str:  # type: ignore[override]
        for key, val in self.responses.items():
            if key in cmd:
                return val
        return ""


ZONES = "\n".join(
    [
        "skin-msm-therm:36600",
        "battery:31000",
        "cpuss-0:41000",
        "mmw3:-273000",  # disabled sensor
        "epm0:-273000",
    ]
)
FREQS = "\n".join(["2016000:2016000", "1228800:2016000", "864000:3187200"])


def monitor(**over: str) -> FakeAndroid:
    r = {"termux-battery-status": REAL_BATTERY, "thermal_zone": ZONES, "cpufreq": FREQS}
    r.update(over)
    return FakeAndroid(r)


class TestBatteryParsing:
    def test_parses_the_real_payload(self) -> None:
        got = parse_battery_json(REAL_BATTERY)
        assert got["current"] == -334000
        assert got["status"] == "DISCHARGING"

    def test_tolerates_noise_around_the_json(self) -> None:
        """adb, Termux and the shell may each prepend a warning."""
        noisy = f"Warning: something\n{REAL_BATTERY}\nbye"
        assert parse_battery_json(noisy)["percentage"] == 96

    def test_no_json_raises_rather_than_returning_empty(self) -> None:
        """An empty dict would read as a device with no battery."""
        with pytest.raises(ValueError, match="no JSON object"):
            parse_battery_json("command not found")

    def test_a_json_array_is_rejected(self) -> None:
        """No braces means no object, and that is caught before parsing.

        The `isinstance` guard behind this is defensive rather than reachable
        through this path -- the brace slice makes a bare array fail earlier.
        Recorded rather than deleted: it is one line and it is the only thing
        standing between a JSON scalar and a dict lookup on it.
        """
        with pytest.raises(ValueError, match="no JSON object"):
            parse_battery_json("[1, 2, 3]")

    def test_malformed_json_between_braces_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_battery_json("noise {not: valid} more")


class TestZoneSummary:
    def test_disabled_sensors_are_filtered(self) -> None:
        """-273000 is a sensor that is off, not a device at absolute zero.

        21 of the handset's 110 zones report this. An unfiltered aggregate
        reads as a cold device -- the direction that makes a thermal
        controller do nothing.
        """
        preferred, hottest = summarise_zones(
            [("skin-msm-therm", 36600), ("mmw3", -273000), ("cpuss-0", 41000)]
        )
        assert preferred == pytest.approx(36.6)
        assert hottest == pytest.approx(41.0)

    def test_preferred_zone_is_skin_not_the_hottest(self) -> None:
        """Skin swung 4.8 C under load where battery moved 1.2 C (ADR-0047)."""
        preferred, hottest = summarise_zones([("skin-msm-therm", 36600), ("cpuss-0", 41000)])
        assert preferred != hottest

    def test_all_sensors_disabled_yields_nan_not_zero(self) -> None:
        preferred, hottest = summarise_zones([("a", -273000), ("b", -273000)])
        assert math.isnan(preferred)
        assert math.isnan(hottest)

    def test_missing_preferred_zone_is_nan_not_a_substitute(self) -> None:
        """Silently substituting another zone would change what is measured."""
        preferred, hottest = summarise_zones([("cpuss-0", 41000)])
        assert math.isnan(preferred)
        assert hottest == pytest.approx(41.0)


class TestSampling:
    def test_reads_a_full_state(self) -> None:
        s = monitor().sample()
        assert s.temperature_c == pytest.approx(36.6)
        assert s.battery_pct == pytest.approx(96.0)
        assert s.charging is False
        assert s.discharge_rate_ma == pytest.approx(334.0)

    def test_discharge_rate_is_nan_while_charging(self) -> None:
        """The sign alternates on AC under constant load (ADR-0047).

        Reporting a magnitude whose sign cannot be defended is worse than
        reporting nothing, because the number survives into a manifest.
        """
        s = monitor(**{"termux-battery-status": CHARGING_BATTERY}).sample()
        assert s.charging is True
        assert math.isnan(s.discharge_rate_ma)

    def test_slope_is_nan_on_the_first_sample(self) -> None:
        m = monitor()
        assert math.isnan(m.sample().temperature_slope_c_per_s)

    def test_unreachable_device_yields_nan_not_zero(self) -> None:
        m = FakeAndroid({})
        s = m.sample()
        assert math.isnan(s.temperature_c)
        assert math.isnan(s.battery_pct)

    def test_freq_ratio_is_the_max_across_clusters(self) -> None:
        """Documented choice, not an accident: it answers 'is anything at speed'."""
        s = monitor().sample()
        assert s.freq_ratio == pytest.approx(1.0)


class TestCapabilitiesAreEarned:
    def test_a_reachable_device_reports_what_it_reads(self) -> None:
        caps = monitor().capabilities()
        assert caps.is_available("temperature_c")
        assert caps.is_available("energy_mj_measured")
        assert not caps.is_available("cpu_util")

    def test_an_unreachable_device_claims_nothing(self) -> None:
        """The bug this test exists for.

        The first version hardcoded temperature_c and freq_ratio to True
        because the probe had shown the handset could report them. Against a
        disconnected device it declared them available while sample() returned
        NaN -- a capability table asserting a claim the monitor could not
        satisfy, which is what the gate exists to catch.
        """
        caps = FakeAndroid({}).capabilities()
        assert not caps.is_available("temperature_c")
        assert not caps.is_available("freq_ratio")
        assert not caps.is_available("energy_mj_measured")
        assert "temperature_c" in caps.unavailable_fields()

    def test_thermal_without_battery_is_a_partial_capability(self) -> None:
        """Termux down but sysfs readable must not disable thermal too."""
        caps = FakeAndroid({"thermal_zone": ZONES, "cpufreq": FREQS}).capabilities()
        assert caps.is_available("temperature_c")
        assert not caps.is_available("energy_mj_measured")

    def test_all_zones_disabled_is_not_a_thermal_capability(self) -> None:
        dead = "\n".join(["mmw3:-273000", "epm0:-273000"])
        caps = FakeAndroid({"thermal_zone": dead, "cpufreq": FREQS}).capabilities()
        assert not caps.is_available("temperature_c")
        assert caps.is_available("freq_ratio")

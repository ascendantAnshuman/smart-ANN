"""Device telemetry. Fields a platform cannot measure are NaN, never modelled."""

from smartann.telemetry.monitor import (
    Capability,
    CapabilityTable,
    DeviceMonitor,
    assert_manifest_consistent,
    unavailable_state,
)

__all__ = [
    "Capability",
    "CapabilityTable",
    "DeviceMonitor",
    "assert_manifest_consistent",
    "unavailable_state",
]

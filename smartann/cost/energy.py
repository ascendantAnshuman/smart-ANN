"""Per-query energy: the model is specified and implemented here, fitted in 3.5.

MobileRAG's form (§3.4.3):

    E = V * [ I(t_s) * t_s + I(t_d) * t_d ]

with `t_s` the CPU-active time and `t_d` the disk-active time, at different
currents. **The coefficients are not fitted here and must not be.** This machine
can measure no energy at all -- no RAPL on Windows, no populated powercap under
WSL2 (ADR-0007, ADR-0008) -- so any coefficient produced here would be a guess
wearing a measurement's name. The phone is the only source, and Phase 3.5 fits
them against it.

**One substantive departure from the paper, and it is measured rather than
assumed.** MobileRAG assigns cluster access to `t_d` at the disk current. We
measured 95-99% of cluster-load time to be graph reconstruction and page-table
work rather than byte transfer (ADR-0026 amendment), so here it goes into `t_s`
at the CPU current. Putting it in `t_d` would misprice the term that dominates
~70% of query time, by roughly the ratio of the two currents. The paper's own
access model has no materialisation term at all, which is the contradiction
ADR-0031 records.

Because this project cannot measure energy, everything below refuses to pretend
otherwise: `EnergyModel.predict` works (it is arithmetic), but any attempt to
record a *fitted* or *measured* energy value on a platform whose capability
table says energy is unavailable raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "MOBILERAG_REFERENCE",
    "EnergyCoefficients",
    "EnergyModel",
    "assert_energy_measurable",
]


@dataclass(frozen=True)
class EnergyCoefficients:
    """Supply voltage and the two currents. Fitted only on a device."""

    voltage_v: float
    i_cpu_ua: float  # I(t_s): CPU-active current
    i_io_ua: float  # I(t_d): disk-active current
    fitted: bool
    source: str

    def __post_init__(self) -> None:
        if self.voltage_v <= 0 or self.i_cpu_ua <= 0 or self.i_io_ua <= 0:
            raise ValueError(
                f"non-physical coefficients: V={self.voltage_v}, "
                f"I_cpu={self.i_cpu_ua}, I_io={self.i_io_ua}"
            )


# The paper's reported values, carried as a REFERENCE POINT so a fitted result
# can be compared against them -- not as a substitute for fitting. Marked
# fitted=False so nothing downstream can mistake them for our measurement.
MOBILERAG_REFERENCE = EnergyCoefficients(
    voltage_v=3.85,
    i_cpu_ua=2300.0,
    i_io_ua=800.0,
    fitted=False,
    source=(
        "MobileRAG (arXiv:2507.01079) §3.4.3, reported for their device. "
        "NOT measured here and NOT fitted here -- a reference point only."
    ),
)


def assert_energy_measurable(capabilities: Any, action: str = "fit an energy model") -> None:
    """Refuse to fit or record energy on a platform that cannot measure it.

    The capability table is a gate, not a log (ADR-0020). Without this, Phase 3
    could close with an energy MAPE that was computed against predictions rather
    than measurements -- a number that looks like validation and is not.
    """
    if capabilities.is_available("energy_mj_measured"):
        return
    raise RuntimeError(
        f"cannot {action}: this platform reports energy_mj_measured as unavailable. "
        f"No RAPL on Windows and no populated powercap under WSL2 (ADR-0007, ADR-0008); "
        f"the phone is the only source and Phase 3.5 does the fitting. Specifying and "
        f"implementing the model is Phase 3's job; validating it is not."
    )


@dataclass(frozen=True)
class EnergyModel:
    """E = V * [I_cpu * t_cpu + I_io * t_io], in millijoules."""

    coefficients: EnergyCoefficients

    def predict_mj(self, t_cpu_ms: np.ndarray, t_io_ms: np.ndarray) -> np.ndarray:
        """Energy per query in mJ. Pure arithmetic; safe to call anywhere.

        Calling this does not claim a measurement. Recording its output as
        `energy_mj_model` is correct; recording it as `energy_mj_measured` is
        what `assert_energy_measurable` exists to prevent.
        """
        c = self.coefficients
        cpu_a = c.i_cpu_ua * 1e-6
        io_a = c.i_io_ua * 1e-6
        joules = c.voltage_v * (
            cpu_a * np.asarray(t_cpu_ms, dtype=np.float64) / 1000.0
            + io_a * np.asarray(t_io_ms, dtype=np.float64) / 1000.0
        )
        return joules * 1000.0

    def split_from_record(self, frame: Any) -> tuple[np.ndarray, np.ndarray]:
        """Split measured stage times into (t_cpu_ms, t_io_ms).

        `t_cluster_load` goes to the CPU side. That is the departure from the
        paper, and it is the measured one: 95-99% of it is reconstruction and
        mapping rather than byte transfer. Only the small genuinely-transfer
        fraction belongs in t_io, and this platform cannot separate it -- so it
        is attributed wholly to CPU here and the phone will refine the split.
        """
        cpu = (
            frame["t_centroid"].to_numpy(dtype=np.float64)
            + frame["t_cluster_load"].to_numpy(dtype=np.float64)
            + frame["t_cluster_search"].to_numpy(dtype=np.float64)
            + frame["t_merge"].to_numpy(dtype=np.float64)
        )
        # Warm page cache on this host, so measured I/O time is not separable
        # from the load term. Zero here is honest; the phone supplies the real
        # split in Phase 3.5.
        io = np.zeros_like(cpu)
        return cpu, io

    def describe(self) -> str:
        c = self.coefficients
        state = "FITTED" if c.fitted else "NOT FITTED (reference values)"
        return (
            f"E = {c.voltage_v} V * [{c.i_cpu_ua} uA * t_cpu + {c.i_io_ua} uA * t_io]  [{state}]\n"
            f"  source: {c.source}\n"
            f"  t_cluster_load is counted as CPU, not I/O (ADR-0026 amendment)"
        )

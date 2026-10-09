"""Why is this device hot? A decomposition of measured heat into named causes.

The question this answers is the one a user asks first and this project could
not previously answer: **the phone is warm and the cable is out, so what is
heating it?** Charging is the obvious suspect and it is absent, which leaves the
retrieval workload, the idle floor of the SoC, and heat that has not yet escaped
from queries that already ran.

**Everything here is attribution, not prediction.** Each term traces back to a
measurement: the idle floor was measured during real idle intervals on battery
(ADR-0052/0053), the per-cluster energy slope was fitted on a discharging trace,
and the temperature rise is read off the thermometer. Nothing is modelled into
existence, and where a term cannot be separated from the data this says so
rather than reporting a plausible split (Section 3, rule 1).

**Two models, fitted on two different power states, and that is the trap.**
`results/thermal_model.json` was fitted on a trace taken with the cable IN,
because Termux:API was unavailable and host-side power logging needed the USB
connection. Its constant term `c` therefore absorbs charging heat, idle power
and the ambient offset together, and using that constant as "idle heating" on an
unplugged trace would quietly import a charger that was not there.
`results/energy_model.json` was fitted on a discharging trace with 27.6% genuine
idle intervals, so it is the one that can separate idle from work. This module
takes the *work* coefficient from the thermal fit, the *idle floor* from the
energy fit, and refuses to take the baseline from the charging trace at all.

**The two fits share a work unit, which buys one derived number.** Both express
themselves per cluster loaded per second: the thermal fit in C/s, the energy fit
in W. Their ratio is the effective thermal capacitance in J/C, a property of the
handset that neither fit could produce alone. It is reported as derived, and the
equilibrium it implies is checked against the temperature actually observed,
because a capacitance that predicts a rise the trace never showed is evidence
about the model rather than about the phone.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "Attribution",
    "DeviceTrace",
    "EnergyModel",
    "PowerAttribution",
    "ThermalAttribution",
    "ThermalModel",
    "attribute",
    "load_trace",
]

# Below this the skin and the hottest zone are telling the same story and a
# policy reading either would act the same. Above it they are not, and which
# sensor the controller reads stops being a detail and becomes a design choice.
SENSOR_DIVERGENCE_C = 10.0


@dataclass(frozen=True)
class EnergyModel:
    """The measured power split: what the device draws idle, and what work adds.

    Only valid when the fit declared itself reportable. ADR-0052 is the reason
    that flag exists: the first on-battery trace never stopped working, so the
    idle intercept was extrapolated 95 units outside the data and came back as
    -0.209 W. A phone drawing negative power is not a small error to correct, it
    is a baseline that was never observed, so this refuses to load it.
    """

    idle_w: float
    slope_w_per_cluster_per_s: float
    r2: float
    idle_fraction: float
    measured_mean_power_w: float

    @classmethod
    def load(cls, path: Path) -> EnergyModel:
        raw = json.loads(path.read_text(encoding="utf-8"))
        attr = raw["attribution"]
        if not attr.get("reportable", False):
            raise ValueError(
                f"{path} reports the energy attribution as NOT reportable: "
                f"{attr.get('why_not_reportable', 'no reason given')}. "
                f"Splitting heat into idle and work needs an idle floor that was "
                f"actually observed; this refuses to invent one."
            )
        return cls(
            idle_w=float(attr["measured_idle_power_w"]),
            slope_w_per_cluster_per_s=float(attr["slope_w_per_cluster_per_s"]),
            r2=float(attr["r2"]),
            idle_fraction=float(attr["idle_fraction"]),
            measured_mean_power_w=float(raw["measured"]["mean_power_w"]),
        )


@dataclass(frozen=True)
class ThermalModel:
    """`dT/dt = a*work_rate + b*(T - T_ref) + c`, as fitted in Phase 4.

    `baseline_is_contaminated` is not a caveat bolted on afterwards. The fitting
    trace ran on AC, so `c` carries charger heat, and every consumer of this
    object has to decide what to do about that. Recording it on the model makes
    the decision explicit instead of leaving it to whoever reads the JSON.
    """

    a_per_cluster_per_s: float
    b_per_s: float
    c_c_per_s: float
    t_ref_c: float
    tau_s: float
    r2: float
    window_s: float
    baseline_is_contaminated: bool

    @classmethod
    def load(cls, path: Path) -> ThermalModel:
        raw = json.loads(path.read_text(encoding="utf-8"))
        fit = raw["fit"]
        src = raw.get("source_trace", {})
        # The fitting trace recorded zero discharging samples because the battery
        # API was down for all of it, which is the same outage that forced the
        # cable in. A trace with no discharging sample cannot vouch for its own
        # baseline, so absence of one is what marks `c` as suspect.
        contaminated = int(src.get("discharging_samples", 0) or 0) == 0
        return cls(
            a_per_cluster_per_s=float(fit["a_per_cluster_per_s"]),
            b_per_s=float(fit["b_per_s"]),
            c_c_per_s=float(fit["c_c_per_s"]),
            t_ref_c=float(fit["t_ref_c"]),
            tau_s=float(fit["tau_seconds"]),
            r2=float(fit["r2"]),
            window_s=float(fit["window_s"]),
            baseline_is_contaminated=contaminated,
        )


@dataclass(frozen=True)
class DeviceTrace:
    """One captured run: temperatures, power, and the work that drove them."""

    name: str
    t_s: np.ndarray
    skin_c: np.ndarray
    hottest_c: np.ndarray
    power_w: np.ndarray
    work_rate: np.ndarray
    n_queries: int
    status_counts: dict[str, int]

    @property
    def minutes(self) -> float:
        return float(self.t_s[-1] - self.t_s[0]) / 60.0

    @property
    def peak_excursion_c(self) -> float:
        """Coldest to hottest the skin got. Matches how `c5_summary.json` reports rise."""
        return float(self.skin_c.max() - self.skin_c.min())

    @property
    def net_change_c(self) -> float:
        """End minus start. Smaller than the excursion whenever the phone cooled back."""
        return float(self.skin_c[-1] - self.skin_c[0])

    @property
    def was_charging(self) -> bool:
        return self.status_counts.get("CHARGING", 0) > 0

    @property
    def power_state_known(self) -> bool:
        return sum(self.status_counts.values()) > 0

    @property
    def unplugged_fraction(self) -> float:
        total = sum(self.status_counts.values())
        if total == 0:
            return float("nan")
        return self.status_counts.get("DISCHARGING", 0) / total

    @property
    def mean_work_rate(self) -> float:
        # Interval zero has no preceding interval to accumulate work over, so it
        # is structurally zero rather than measured, and averaging it in would
        # understate the work by one sample's worth.
        return float(self.work_rate[1:].mean()) if self.work_rate.size > 1 else float("nan")

    @property
    def mean_power_w(self) -> float:
        ok = np.isfinite(self.power_w)
        return float(self.power_w[ok].mean()) if ok.any() else float("nan")

    @property
    def peak_divergence_c(self) -> float:
        """How far the hottest zone ran ahead of the skin, at its worst."""
        ok = np.isfinite(self.skin_c) & np.isfinite(self.hottest_c)
        return float((self.hottest_c[ok] - self.skin_c[ok]).max()) if ok.any() else float("nan")

    @property
    def hottest_max_c(self) -> float:
        ok = np.isfinite(self.hottest_c)
        return float(self.hottest_c[ok].max()) if ok.any() else float("nan")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Capture a trace with scripts/device_trace.py first; "
            f"this will not invent one."
        )
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def load_trace(trace_dir: Path, prefix: str) -> DeviceTrace:
    """Read `<prefix>telemetry.jsonl` and `<prefix>queries.jsonl` into arrays.

    Work is counted in clusters loaded, because that is the unit both fitted
    models speak. Traces that record `n_clusters` are used directly; the C5
    traces record only `nprobe`, which is the same number, verified equal on
    every one of the 19,144 queries in the C4 trace, so it stands in when the
    explicit count is absent and nothing else is accepted as a substitute.
    """
    telem = _read_jsonl(trace_dir / f"{prefix}telemetry.jsonl")
    queries = _read_jsonl(trace_dir / f"{prefix}queries.jsonl")
    if not queries:
        raise ValueError(f"{prefix}queries.jsonl is empty; no work drove this trace.")

    t_all = np.array([r["t"] for r in telem], dtype=np.float64)
    skin_all = np.array([r.get("skin_c", np.nan) for r in telem], dtype=np.float64)
    hot_all = np.array([r.get("hottest_c", np.nan) for r in telem], dtype=np.float64)

    ok = np.isfinite(skin_all)
    if ok.sum() < 10:
        raise ValueError(
            f"{prefix}telemetry.jsonl carries {int(ok.sum())} finite skin-temperature "
            f"samples. There is no thermal signal here to attribute."
        )
    t, skin, hottest = t_all[ok], skin_all[ok], hot_all[ok]
    telem = [r for r, keep in zip(telem, ok, strict=True) if keep]

    # Battery power, where the sensor gave both halves of it. Current is negative
    # while discharging; the magnitude is what heats the phone.
    power = np.full(t.size, np.nan)
    for i, r in enumerate(telem):
        cur, volt = r.get("current_ua"), r.get("voltage_mv")
        if cur is not None and volt is not None and math.isfinite(cur) and math.isfinite(volt):
            power[i] = abs(float(cur)) * 1e-6 * float(volt) * 1e-3

    status: dict[str, int] = {}
    for r in telem:
        s = r.get("status")
        if s:
            status[s] = status.get(s, 0) + 1

    # Query timestamps are relative to trace start; telemetry timestamps are
    # absolute. Rebase before bucketing, or every query lands in interval zero.
    qt = np.array([r["t_wall"] for r in queries], dtype=np.float64) + t_all[0]
    if "n_clusters" in queries[0]:
        qwork = np.array([r["n_clusters"] for r in queries], dtype=np.float64)
    elif "nprobe" in queries[0]:
        qwork = np.array([r["nprobe"] for r in queries], dtype=np.float64)
    else:
        raise ValueError(
            f"{prefix}queries.jsonl records neither n_clusters nor nprobe, so the "
            f"work that produced this heat cannot be counted."
        )

    order = np.argsort(qt)
    qt, qwork = qt[order], qwork[order]
    idx = np.searchsorted(qt, t)
    work_rate = np.zeros(t.size)
    for i in range(1, t.size):
        dt = t[i] - t[i - 1]
        lo, hi = idx[i - 1], idx[i]
        if hi > lo and dt > 0:
            work_rate[i] = qwork[lo:hi].sum() / dt

    return DeviceTrace(
        name=prefix.rstrip("_") or trace_dir.name,
        t_s=t - t[0],
        skin_c=skin,
        hottest_c=hottest,
        power_w=power,
        work_rate=work_rate,
        n_queries=len(queries),
        status_counts=status,
    )


@dataclass(frozen=True)
class PowerAttribution:
    """Where the watts went, from measured quantities only.

    This is the half of the answer that does not depend on the thermal fit, and
    it is the stronger half. Both terms come from a discharging trace: the idle
    floor was observed during real idle intervals, and the slope is the extra
    power each cluster load costs. Heat is dissipated power, so a split of the
    watts is a split of the heat.
    """

    idle_w: float
    work_w: float
    charging_w: float
    measured_mean_w: float
    unexplained_w: float

    @property
    def total_modelled_w(self) -> float:
        charging = self.charging_w if math.isfinite(self.charging_w) else 0.0
        return self.idle_w + self.work_w + charging

    @property
    def denominator_w(self) -> float:
        """Total heat the shares are taken against.

        The measured draw, not the sum of the terms, whenever the sensor gave
        one. On the C5 trace 0.177 W of a measured 1.407 W lands in neither
        bucket, and dividing by the terms alone would quietly hand that power to
        retrieval, lifting its share from 41% to 47%. Power nobody attributed is
        not power the controller can remove, so it dilutes the share instead.

        `max` guards the other direction: were the terms to exceed the measured
        draw, dividing by the smaller number would print shares above 100%.
        """
        measured = self.measured_mean_w
        if math.isfinite(measured) and measured > 0:
            return max(measured, self.total_modelled_w)
        return self.total_modelled_w

    @property
    def work_share(self) -> float:
        total = self.denominator_w
        return self.work_w / total if total > 0 else float("nan")

    @property
    def idle_share(self) -> float:
        total = self.denominator_w
        return self.idle_w / total if total > 0 else float("nan")

    @property
    def unexplained_share(self) -> float:
        total = self.denominator_w
        if not (math.isfinite(self.unexplained_w) and total > 0):
            return float("nan")
        return max(self.unexplained_w, 0.0) / total


@dataclass(frozen=True)
class ThermalAttribution:
    """The fitted model's account of the temperature change, term by term.

    **The terms are only meaningful together, which is why a missing one voids
    the rest.** Integrating `a*work_rate` over the C5 trace gives +216 C, and
    the fit balances that against a constant term of -0.142 C/s worth -188 C
    over the same span. Each is an artefact of where the regression put its
    operating point, not a quantity of heat anyone could measure. Drop the
    constant and the surviving terms do not become a conservative estimate, they
    become arithmetic on an unbalanced equation: +216 C of heating attributed to
    a phone whose thermometer moved 5.6 C.

    So when the constant cannot be used, `usable` is False and every term is
    NaN. Section 3 rule 1 says a value that was not measured is not reported,
    and a term-by-term split of an equation missing a term is exactly that.
    """

    retrieval_c: float
    dissipation_c: float
    baseline_c: float
    modelled_rise_c: float
    observed_rise_c: float
    residual_c: float
    usable: bool
    refusal_reason: str


@dataclass(frozen=True)
class Attribution:
    """The full answer, plus everything needed to judge how much to trust it."""

    trace: DeviceTrace
    power: PowerAttribution
    thermal: ThermalAttribution
    energy_model: EnergyModel
    thermal_model: ThermalModel
    c_th_j_per_c: float
    r_th_c_per_w: float
    predicted_equilibrium_rise_c: float

    @property
    def score(self) -> float:
        """0-100: the share of the device's heat that retrieval effort controls.

        Deliberately built on the power split rather than the thermal one. Both
        of its inputs were measured on battery, so the number does not inherit
        the charging contamination in the thermal fit's constant term, and it
        does not inherit that fit's R^2 of 0.30 either.

        Read it as a ceiling on what any retrieval controller can achieve here.
        A score of 35 means a controller that drove effort to zero would remove
        about a third of the heat and the other two thirds would remain, which
        is a statement about the device rather than about the policy.
        """
        return 100.0 * self.power.work_share

    @property
    def sensors_disagree(self) -> bool:
        div = self.trace.peak_divergence_c
        return math.isfinite(div) and div >= SENSOR_DIVERGENCE_C

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace": {
                "name": self.trace.name,
                "minutes": self.trace.minutes,
                "n_queries": self.trace.n_queries,
                "status_counts": self.trace.status_counts,
                "unplugged_fraction": self.trace.unplugged_fraction,
                "skin_min_c": float(self.trace.skin_c.min()),
                "skin_max_c": float(self.trace.skin_c.max()),
                "hottest_max_c": self.trace.hottest_max_c,
                "peak_excursion_c": self.trace.peak_excursion_c,
                "net_change_c": self.trace.net_change_c,
                "peak_sensor_divergence_c": self.trace.peak_divergence_c,
                "mean_work_rate_clusters_per_s": self.trace.mean_work_rate,
                "mean_power_w": self.trace.mean_power_w,
            },
            "power_attribution_w": {
                "idle": self.power.idle_w,
                "retrieval_work": self.power.work_w,
                "charging": self.power.charging_w,
                "measured_mean": self.power.measured_mean_w,
                "unexplained": self.power.unexplained_w,
                "work_share": self.power.work_share,
            },
            "thermal_attribution_c": {
                "retrieval": self.thermal.retrieval_c,
                "dissipation": self.thermal.dissipation_c,
                "baseline": self.thermal.baseline_c,
                "usable": self.thermal.usable,
                "refusal_reason": self.thermal.refusal_reason,
                "modelled_rise": self.thermal.modelled_rise_c,
                "observed_rise": self.thermal.observed_rise_c,
                "residual": self.thermal.residual_c,
                "model_r2": self.thermal_model.r2,
            },
            "derived": {
                "c_th_j_per_c": self.c_th_j_per_c,
                "r_th_c_per_w": self.r_th_c_per_w,
                "tau_s": self.thermal_model.tau_s,
                "predicted_equilibrium_rise_c": self.predicted_equilibrium_rise_c,
            },
            "score": self.score,
            "sensors_disagree": self.sensors_disagree,
        }


def attribute(
    trace: DeviceTrace,
    energy_model: EnergyModel,
    thermal_model: ThermalModel,
) -> Attribution:
    """Decompose one trace's heat into causes, using both fitted models.

    The power split leads because both of its terms were measured on battery.
    The thermal split follows because it explains only 30% of the variance, and
    its baseline term is dropped outright when the fit came from a charging
    trace rather than being silently reused.
    """
    work_rate = trace.mean_work_rate
    work_w = energy_model.slope_w_per_cluster_per_s * work_rate
    idle_w = energy_model.idle_w

    # Charging contributes nothing when the cable was out, and that zero is the
    # point of the whole exercise rather than a missing term. When a trace did
    # charge, the cable's contribution cannot be separated from the battery
    # reading at all, so it is left as NaN instead of being guessed at.
    charging_w = 0.0 if not trace.was_charging else float("nan")

    measured = trace.mean_power_w
    unexplained = measured - (idle_w + work_w) if math.isfinite(measured) else float("nan")

    power = PowerAttribution(
        idle_w=idle_w,
        work_w=work_w,
        charging_w=charging_w,
        measured_mean_w=measured,
        unexplained_w=unexplained,
    )

    observed = trace.net_change_c
    if thermal_model.baseline_is_contaminated:
        nan = float("nan")
        thermal = ThermalAttribution(
            retrieval_c=nan,
            dissipation_c=nan,
            baseline_c=nan,
            modelled_rise_c=nan,
            observed_rise_c=observed,
            residual_c=nan,
            usable=False,
            refusal_reason=(
                "the Phase 4 thermal fit was taken on AC, so its constant term carries "
                "charger heat that cannot be separated from idle power. The remaining "
                "terms do not stand alone: integrated over this trace the retrieval term "
                "alone reaches +216 C against a thermometer that moved 5.6 C, because the "
                "constant is what balances it. Re-fit on an unplugged trace to unlock this"
            ),
        )
    else:
        # Integrate each term of dT/dt over the trace. Trapezoid on the interior,
        # which is what the fit's own differencing implies.
        dt = np.diff(trace.t_s)
        wr_mid = 0.5 * (trace.work_rate[1:] + trace.work_rate[:-1])
        t_mid = 0.5 * (trace.skin_c[1:] + trace.skin_c[:-1])

        retrieval_c = float(np.sum(thermal_model.a_per_cluster_per_s * wr_mid * dt))
        dissipation_c = float(np.sum(thermal_model.b_per_s * (t_mid - thermal_model.t_ref_c) * dt))
        baseline_c = float(thermal_model.c_c_per_s * dt.sum())
        modelled = retrieval_c + dissipation_c + baseline_c

        thermal = ThermalAttribution(
            retrieval_c=retrieval_c,
            dissipation_c=dissipation_c,
            baseline_c=baseline_c,
            modelled_rise_c=modelled,
            observed_rise_c=observed,
            residual_c=observed - modelled,
            usable=True,
            refusal_reason="",
        )

    # Both fits are per cluster per second, so their ratio cancels the work unit
    # and leaves J/C. Neither fit could produce this alone.
    a = thermal_model.a_per_cluster_per_s
    c_th = energy_model.slope_w_per_cluster_per_s / a if a > 0 else float("nan")
    r_th = thermal_model.tau_s / c_th if math.isfinite(c_th) and c_th > 0 else float("nan")
    total_w = idle_w + work_w
    eq_rise = r_th * total_w if math.isfinite(r_th) else float("nan")

    return Attribution(
        trace=trace,
        power=power,
        thermal=thermal,
        energy_model=energy_model,
        thermal_model=thermal_model,
        c_th_j_per_c=c_th,
        r_th_c_per_w=r_th,
        predicted_equilibrium_rise_c=eq_rise,
    )

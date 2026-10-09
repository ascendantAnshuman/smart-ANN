"""The attribution must refuse more readily than it reports.

Three refusals are the point of these tests, and each one exists because the
alternative is a number that looks fine and is wrong.

An energy fit that never observed idle time returns a negative intercept
(ADR-0052), so loading one must fail rather than propagate a phone that consumes
negative power. A thermal fit taken on AC carries charger heat in its constant
term, so a term-by-term split using the other two terms must be refused, not
approximated: integrated over a real trace the retrieval term alone reaches
+216 C beside a thermometer that moved 5.6 C. And power the split cannot account
for must lower the score rather than be handed to retrieval, because the
controller cannot remove heat nobody attributed to it.

The known-answer test runs the opposite way: a synthetic trace built from a
chosen idle floor and a chosen slope must score what those two numbers imply. A
refusal suite alone would pass on a module that refused everything.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from smartann.thermal.attribution import (
    EnergyModel,
    ThermalModel,
    attribute,
    load_trace,
)
from smartann.thermal.report import render


def write_models(
    d: Path,
    *,
    reportable: bool = True,
    idle_w: float = 0.60,
    slope: float = 0.004,
    discharging_samples: int = 0,
    a: float = 0.0009,
) -> tuple[Path, Path]:
    """Both fitted models on disk, with the two flags the module keys off.

    `discharging_samples = 0` is what marks a thermal fit as taken on AC, which
    is the real state of `results/thermal_model.json`: its battery API failed
    for all 478 samples, which is why the cable was in.
    """
    energy = d / "energy_model.json"
    energy.write_text(
        json.dumps(
            {
                "measured": {"mean_power_w": 1.0},
                "attribution": {
                    "reportable": reportable,
                    "measured_idle_power_w": idle_w,
                    "slope_w_per_cluster_per_s": slope,
                    "r2": 0.5,
                    "idle_fraction": 0.3,
                    "why_not_reportable": "" if reportable else "no idle intervals in the trace",
                },
            }
        ),
        encoding="utf-8",
    )
    thermal = d / "thermal_model.json"
    thermal.write_text(
        json.dumps(
            {
                "fit": {
                    "a_per_cluster_per_s": a,
                    "b_per_s": -0.015,
                    "c_c_per_s": -0.14,
                    "t_ref_c": 34.0,
                    "tau_seconds": 66.7,
                    "r2": 0.30,
                    "window_s": 30.0,
                },
                "source_trace": {"discharging_samples": discharging_samples},
            }
        ),
        encoding="utf-8",
    )
    return energy, thermal


def write_trace(
    d: Path,
    *,
    prefix: str = "t_",
    work_per_query: int = 8,
    queries_per_interval: int = 10,
    n: int = 40,
    idle_w: float = 0.60,
    slope: float = 0.004,
    volts: float = 4.0,
    skin_start: float = 30.0,
    skin_rise: float = 4.0,
    hottest_offset: float = 5.0,
    use_nprobe: bool = False,
    extra_w: float = 0.0,
    status: str = "DISCHARGING",
) -> None:
    """A trace whose power is `idle_w + slope*work_rate` by construction.

    Built so the attribution has a right answer to find. `extra_w` adds draw the
    split cannot explain, which is how the dilution test creates unattributed
    power without disturbing the two terms that are supposed to be recovered.

    Query timestamps are relative to trace start while telemetry timestamps are
    absolute, exactly as the device traces record them, so this fixture also
    pins that the loader rebases rather than dropping every query into the first
    interval.
    """
    t0, period = 1_700_000_000.0, 3.0
    work_rate = work_per_query * queries_per_interval / period

    telem, queries = [], []
    qid = 0
    for i in range(n):
        t = t0 + i * period
        skin = skin_start + skin_rise * (i / max(n - 1, 1))
        power = idle_w + slope * (work_rate if i > 0 else 0.0) + extra_w
        telem.append(
            {
                "t": t,
                "skin_c": round(skin, 3),
                "hottest_c": round(skin + hottest_offset, 3),
                # Discharging current is negative on the handset; magnitude is
                # what heats it, and the loader must take the absolute value.
                "current_ua": -(power / volts) * 1e6,
                "voltage_mv": volts * 1e3,
                "status": status,
            }
        )
        # Work is attributed to the interval ENDING at a sample, so a sample's
        # queries must fall before it. Emitting them after puts power and work
        # one interval apart and halves the recovered slope.
        if i > 0:
            for j in range(queries_per_interval):
                rec = {
                    "t_wall": (i - 1) * period + (j + 0.5) * period / queries_per_interval,
                    "query_id": qid,
                    "t_total_ms": 10.0,
                }
                rec["nprobe" if use_nprobe else "n_clusters"] = work_per_query
                queries.append(rec)
                qid += 1

    (d / f"{prefix}telemetry.jsonl").write_text(
        "\n".join(json.dumps(r) for r in telem), encoding="utf-8"
    )
    (d / f"{prefix}queries.jsonl").write_text(
        "\n".join(json.dumps(r) for r in queries), encoding="utf-8"
    )


def test_energy_model_refuses_an_unreportable_fit(tmp_path: Path) -> None:
    """A fit that never saw idle time cannot supply an idle floor."""
    energy, _ = write_models(tmp_path, reportable=False)
    with pytest.raises(ValueError, match="NOT reportable"):
        EnergyModel.load(energy)


def test_thermal_split_is_refused_when_the_fit_came_from_a_charging_trace(
    tmp_path: Path,
) -> None:
    """Every term NaN, and a reason, rather than two terms of an unbalanced equation."""
    energy, thermal = write_models(tmp_path, discharging_samples=0)
    write_trace(tmp_path)
    a = attribute(load_trace(tmp_path, "t_"), EnergyModel.load(energy), ThermalModel.load(thermal))

    assert a.thermal.usable is False
    assert "AC" in a.thermal.refusal_reason or "charger" in a.thermal.refusal_reason
    for term in (a.thermal.retrieval_c, a.thermal.dissipation_c, a.thermal.baseline_c):
        assert math.isnan(term), "a refused split must report no terms at all"
    # The thermometer is still a measurement and survives the refusal.
    assert math.isfinite(a.thermal.observed_rise_c)


def test_thermal_split_is_computed_when_the_fit_came_from_battery(tmp_path: Path) -> None:
    """The refusal is conditional, not a permanent disabling of the section."""
    energy, thermal = write_models(tmp_path, discharging_samples=200)
    write_trace(tmp_path)
    a = attribute(load_trace(tmp_path, "t_"), EnergyModel.load(energy), ThermalModel.load(thermal))

    assert a.thermal.usable is True
    assert math.isfinite(a.thermal.retrieval_c)
    assert a.thermal.retrieval_c > 0, "work can only add heat"
    # The residual is defined as what the terms failed to explain.
    assert a.thermal.residual_c == pytest.approx(
        a.thermal.observed_rise_c - a.thermal.modelled_rise_c
    )


def test_score_recovers_a_known_split(tmp_path: Path) -> None:
    """Built from a chosen idle floor and slope, it must score what they imply."""
    idle_w, slope, work_per_query, qpi = 0.60, 0.004, 8, 10
    energy, thermal = write_models(tmp_path, idle_w=idle_w, slope=slope)
    write_trace(
        tmp_path,
        idle_w=idle_w,
        slope=slope,
        work_per_query=work_per_query,
        queries_per_interval=qpi,
    )
    a = attribute(load_trace(tmp_path, "t_"), EnergyModel.load(energy), ThermalModel.load(thermal))

    work_rate = work_per_query * qpi / 3.0
    expected = 100.0 * (slope * work_rate) / (idle_w + slope * work_rate)
    assert a.score == pytest.approx(expected, rel=0.05)
    assert a.power.work_share + a.power.idle_share == pytest.approx(1.0, abs=0.05)


def test_unattributed_power_lowers_the_score(tmp_path: Path) -> None:
    """Heat nobody explained is not heat the controller can remove.

    Dividing by the sum of the terms instead of the measured draw lifts the C5
    score from 41 to 47 on exactly this difference, which is the direction that
    flatters the contribution.
    """
    energy, thermal = write_models(tmp_path)
    write_trace(tmp_path, prefix="clean_")
    write_trace(tmp_path, prefix="noisy_", extra_w=0.40)

    clean = attribute(
        load_trace(tmp_path, "clean_"), EnergyModel.load(energy), ThermalModel.load(thermal)
    )
    noisy = attribute(
        load_trace(tmp_path, "noisy_"), EnergyModel.load(energy), ThermalModel.load(thermal)
    )

    assert noisy.power.unexplained_w > 0.3
    assert noisy.score < clean.score
    assert noisy.power.work_share <= 1.0


def test_nprobe_stands_in_for_n_clusters(tmp_path: Path) -> None:
    """The C5 traces record only nprobe; it is the same count, verified on C4."""
    energy, thermal = write_models(tmp_path)
    write_trace(tmp_path, prefix="a_", use_nprobe=False)
    write_trace(tmp_path, prefix="b_", use_nprobe=True)

    em, tm = EnergyModel.load(energy), ThermalModel.load(thermal)
    by_clusters = attribute(load_trace(tmp_path, "a_"), em, tm)
    by_nprobe = attribute(load_trace(tmp_path, "b_"), em, tm)
    assert by_clusters.score == pytest.approx(by_nprobe.score)


def test_a_trace_with_no_work_column_is_refused(tmp_path: Path) -> None:
    """Neither count present means the heat cannot be tied to work at all."""
    write_trace(tmp_path)
    q = tmp_path / "t_queries.jsonl"
    rows = [json.loads(line) for line in q.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        r.pop("n_clusters", None)
    q.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="neither n_clusters nor nprobe"):
        load_trace(tmp_path, "t_")


def test_a_trace_without_a_thermal_signal_is_refused(tmp_path: Path) -> None:
    """No thermometer means nothing to attribute; NaN is not 'cold'."""
    write_trace(tmp_path)
    tel = tmp_path / "t_telemetry.jsonl"
    rows = [json.loads(line) for line in tel.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        r["skin_c"] = None
    tel.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="no thermal signal"):
        load_trace(tmp_path, "t_")


def test_missing_trace_names_the_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"telemetry\.jsonl"):
        load_trace(tmp_path, "absent_")


def test_sensor_divergence_is_detected(tmp_path: Path) -> None:
    """The C5 finding in miniature: the silicon runs hot while the skin does not."""
    energy, thermal = write_models(tmp_path)
    write_trace(tmp_path, prefix="near_", hottest_offset=2.0)
    write_trace(tmp_path, prefix="far_", hottest_offset=30.0)

    em, tm = EnergyModel.load(energy), ThermalModel.load(thermal)
    near = attribute(load_trace(tmp_path, "near_"), em, tm)
    far = attribute(load_trace(tmp_path, "far_"), em, tm)

    assert near.sensors_disagree is False
    assert far.sensors_disagree is True
    assert "SENSOR DISAGREEMENT" in render(far)
    assert "SENSOR DISAGREEMENT" not in render(near)


def test_charging_contribution_is_nan_not_zero(tmp_path: Path) -> None:
    """On AC the cable's share cannot be separated, and NaN says so; 0.0 would lie."""
    energy, thermal = write_models(tmp_path)
    write_trace(tmp_path, status="CHARGING")
    a = attribute(load_trace(tmp_path, "t_"), EnergyModel.load(energy), ThermalModel.load(thermal))
    assert a.trace.was_charging is True
    assert math.isnan(a.power.charging_w)


def test_derived_capacitance_comes_from_both_fits(tmp_path: Path) -> None:
    """C_th is the ratio of the two work coefficients and exists only jointly."""
    slope, a_coef = 0.004, 0.0009
    energy, thermal = write_models(tmp_path, slope=slope, a=a_coef)
    write_trace(tmp_path, slope=slope)
    res = attribute(
        load_trace(tmp_path, "t_"), EnergyModel.load(energy), ThermalModel.load(thermal)
    )
    assert res.c_th_j_per_c == pytest.approx(slope / a_coef)
    assert res.r_th_c_per_w == pytest.approx(res.thermal_model.tau_s / res.c_th_j_per_c)


def test_report_renders_without_a_power_sensor(tmp_path: Path) -> None:
    """A trace with no battery reading must still print, with n/a where it belongs."""
    energy, thermal = write_models(tmp_path)
    write_trace(tmp_path)
    tel = tmp_path / "t_telemetry.jsonl"
    rows = [json.loads(line) for line in tel.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        r.pop("current_ua", None)
    tel.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    res = attribute(
        load_trace(tmp_path, "t_"), EnergyModel.load(energy), ThermalModel.load(thermal)
    )
    text = render(res)
    assert "n/a" in text
    assert math.isnan(res.power.measured_mean_w)
    # Falls back to the terms, so the score survives a missing sensor.
    assert math.isfinite(res.score)


def test_as_dict_is_json_serialisable(tmp_path: Path) -> None:
    """The JSON output feeds figures and the paper, so it must round-trip."""
    energy, thermal = write_models(tmp_path)
    write_trace(tmp_path)
    res = attribute(
        load_trace(tmp_path, "t_"), EnergyModel.load(energy), ThermalModel.load(thermal)
    )
    text = json.dumps(res.as_dict())
    back = json.loads(text)
    assert back["score"] == pytest.approx(res.score)
    assert back["thermal_attribution_c"]["usable"] is False


def test_work_rate_ignores_the_structurally_empty_first_interval(tmp_path: Path) -> None:
    """Interval zero has no predecessor to accumulate over, so it is not a measurement."""
    write_models(tmp_path)
    write_trace(tmp_path, work_per_query=8, queries_per_interval=10)
    trace = load_trace(tmp_path, "t_")
    assert trace.work_rate[0] == 0.0
    assert trace.mean_work_rate == pytest.approx(np.mean(trace.work_rate[1:]))
    assert trace.mean_work_rate > 0

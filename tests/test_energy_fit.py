"""The energy fit must refuse an intercept it cannot measure.

The first on-battery trace ran queries continuously, so `work_rate` never
approached zero and the regression put the idle-power intercept 95 units
outside the data. It returned **-0.209 W** -- a phone consuming negative power
(ADR-0052). No estimator fixes that; the baseline simply was not observed.

These tests drive the real script over synthetic traces, because the property
worth pinning is end-to-end: given a trace with no idle time it must refuse,
and given one with idle time it must report.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def write_trace(
    d: Path,
    *,
    idle_fraction: float,
    idle_w: float = 0.80,
    w_per_cluster: float = 0.004,
    n: int = 120,
    prefix: str = "t_",
) -> None:
    """A synthetic trace with a known idle power and a known slope.

    Power is constructed as `idle_w + w_per_cluster * work`, so a fit that
    recovers those two numbers is doing its job and one that does not is not.

    **Alignment matters and is the easy thing to get wrong.** The fit attributes
    the work done in `(t[i-1], t[i]]` to the power sampled at `t[i]`, so this
    builds the work rates first and sets each row's power from the work of the
    interval ENDING at it. The first draft emitted a row's queries *after* its
    own timestamp, which put power and work one interval apart and made a
    perfectly linear dataset fit at a third of its true slope -- a fixture bug
    that would have read as the estimator being broken.
    """
    t0 = 1_700_000_000.0
    period = 5.0
    volts = 4.0
    tel: list[dict] = []
    qs: list[dict] = []

    rates = [
        0.0 if (i % 10) < round(idle_fraction * 10) else 100.0 + (i % 5) * 30.0 for i in range(n)
    ]

    for i in range(n):
        t_abs = t0 + i * period
        # Power at row i reflects the work in the interval ending at row i.
        work_here = rates[i - 1] if i > 0 else 0.0
        power = idle_w + w_per_cluster * work_here
        tel.append(
            {
                "t": t_abs,
                "skin_c": 33.0 + i * 0.01,
                "hottest_c": 45.0,
                "current_ua": -round(power / volts * 1e6),
                "voltage_mv": int(volts * 1000),
                "battery_c": 30.0,
                "pct": 80,
                "status": "DISCHARGING",
                "charge_counter": 1_000_000,
            }
        )

    # Queries for interval i fall inside (i*period, (i+1)*period], carrying
    # exactly rates[i] clusters per second.
    for i, rate in enumerate(rates):
        if rate <= 0:
            continue
        n_q = 25
        per_query = rate * period / n_q
        for j in range(n_q):
            qs.append(
                {
                    "t_wall": i * period + (j + 1) * period / n_q,
                    "nprobe": 8,
                    "query_id": 0,
                    "t_total_ms": 20.0,
                    "t_centroid_ms": 1.0,
                    "n_clusters": per_query,
                    "bytes_read": 1000,
                }
            )

    d.mkdir(parents=True, exist_ok=True)
    (d / f"{prefix}telemetry.jsonl").write_text(
        "\n".join(json.dumps(r) for r in tel) + "\n", encoding="utf-8"
    )
    (d / f"{prefix}queries.jsonl").write_text(
        "\n".join(json.dumps(r) for r in qs) + "\n", encoding="utf-8"
    )


def run_fit(trace_dir: Path, out: Path) -> dict:
    script = Path(__file__).resolve().parent.parent / "scripts" / "fit_energy_model.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--trace-dir",
            str(trace_dir),
            "--prefix",
            "t_",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(out.read_text(encoding="utf-8"))


class TestBaselineIsRequired:
    def test_no_idle_time_is_not_reportable(self, tmp_path: Path) -> None:
        """The exact shape of the real failure: continuous work, no baseline."""
        write_trace(tmp_path / "tr", idle_fraction=0.0)
        got = run_fit(tmp_path / "tr", tmp_path / "out.json")
        assert got["attribution"]["reportable"] is False
        assert got["attribution"]["idle_fraction"] == 0.0
        assert "extrapolated" in got["attribution"]["why_not_reportable"]

    def test_idle_time_makes_it_reportable(self, tmp_path: Path) -> None:
        write_trace(tmp_path / "tr", idle_fraction=0.3)
        got = run_fit(tmp_path / "tr", tmp_path / "out.json")
        a = got["attribution"]
        assert a["reportable"] is True
        assert a["idle_fraction"] >= 0.05
        assert a["mj_per_cluster"] is not None

    def test_recovers_the_planted_idle_power(self, tmp_path: Path) -> None:
        """A fit that cannot recover a known baseline is not measuring one."""
        write_trace(tmp_path / "tr", idle_fraction=0.3, idle_w=0.80)
        got = run_fit(tmp_path / "tr", tmp_path / "out.json")
        assert abs(got["attribution"]["intercept_w"] - 0.80) < 0.05
        assert abs(got["attribution"]["measured_idle_power_w"] - 0.80) < 0.05

    def test_recovers_the_planted_slope(self, tmp_path: Path) -> None:
        write_trace(tmp_path / "tr", idle_fraction=0.3, w_per_cluster=0.004)
        got = run_fit(tmp_path / "tr", tmp_path / "out.json")
        assert abs(got["attribution"]["slope_w_per_cluster_per_s"] - 0.004) < 5e-4

    def test_too_little_idle_is_still_refused(self, tmp_path: Path) -> None:
        """A couple of idle points is not a baseline.

        The threshold exists because an intercept resting on three samples out
        of 120 fails the same way as one resting on none, just less obviously.
        """
        write_trace(tmp_path / "tr", idle_fraction=0.0)
        got = run_fit(tmp_path / "tr", tmp_path / "out.json")
        assert got["attribution"]["reportable"] is False


class TestChargingTracesAreRefused:
    def test_a_charging_sample_stops_the_fit(self, tmp_path: Path) -> None:
        """ADR-0047: while charging the current's sign alternates.

        Integrating over it yields approximately zero, so a charging trace is
        not a weak measurement -- it is not one.
        """
        d = tmp_path / "tr"
        write_trace(d, idle_fraction=0.3)
        rows = [json.loads(x) for x in (d / "t_telemetry.jsonl").read_text().splitlines()]
        rows[5]["status"] = "CHARGING"
        (d / "t_telemetry.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        script = Path(__file__).resolve().parent.parent / "scripts" / "fit_energy_model.py"
        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                "--trace-dir",
                str(d),
                "--prefix",
                "t_",
                "--out",
                str(tmp_path / "o.json"),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.returncode != 0
        assert "CHARGING" in proc.stdout + proc.stderr

    def test_a_positive_current_stops_the_fit(self, tmp_path: Path) -> None:
        d = tmp_path / "tr"
        write_trace(d, idle_fraction=0.3)
        rows = [json.loads(x) for x in (d / "t_telemetry.jsonl").read_text().splitlines()]
        rows[7]["current_ua"] = +50_000
        (d / "t_telemetry.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        script = Path(__file__).resolve().parent.parent / "scripts" / "fit_energy_model.py"
        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                "--trace-dir",
                str(d),
                "--prefix",
                "t_",
                "--out",
                str(tmp_path / "o.json"),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.returncode != 0
        assert "positive current" in (proc.stdout + proc.stderr)

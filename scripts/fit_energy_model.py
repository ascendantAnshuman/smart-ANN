#!/usr/bin/env python
"""C4: measured energy from the on-battery device trace.

The first measured energy in this project. ADR-0007 and ADR-0008 recorded in
Phase 1 that no energy could be measured on the development laptop; ADR-0046
found the route (Termux:API, not sysfs); ADR-0047 established the sign
convention. This consumes a trace taken with all 209 samples DISCHARGING and
every current negative.

**It reports what is identifiable and refuses what is not.**

Identifiable: total energy, mean power, and the relationship between retrieval
effort and power. Those need no decomposition -- they are integrals and
orderings over measured samples.

NOT identifiable from this trace: per-query energy attributable to retrieval.
The regression `power ~ a*work_rate + b` returns **b = -0.209 W**, and a phone
idling at negative 209 milliwatts does not exist. The cause is visible in the
data: `work_rate` spans 95-251 clusters/s and never approaches zero, so the
intercept is an extrapolation 95 units beyond the nearest observation. The
arithmetic confirms it -- at the trace's mean effort of 12.2 clusters the
marginal term predicts 115 mJ/query against a measured all-in average of
99.3 mJ/query, the marginal exceeding the total with a negative intercept
absorbing the difference.

So the marginal coefficient is not reported as a result. A number that implies
negative idle power is not a smaller version of the right answer.

**What would fix it:** idle periods inside the trace. The workload must stop
for long enough to measure the baseline the screen and radios draw, rather than
leaving it to be extrapolated. That is a change to `device_trace.py`, not to
this fit.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run scripts/device_trace.py on the handset, "
            f"UNPLUGGED; this script will not invent an energy trace."
        )
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", default="results/device")
    # c4_ is the DUTY-CYCLED trace (ADR-0053/0054), the one whose baseline is
    # measured rather than extrapolated. The default used to be e2_, the earlier
    # continuous trace, so running the documented command regenerated the
    # SUPERSEDED "not reportable" result and overwrote the good one.
    ap.add_argument("--prefix", default="c4_")
    ap.add_argument("--out", default="results/energy_model.json")
    args = ap.parse_args()

    d = Path(args.trace_dir)
    tel = load_jsonl(d / f"{args.prefix}telemetry.jsonl")
    qs = load_jsonl(d / f"{args.prefix}queries.jsonl")

    status = Counter(r.get("status") for r in tel)
    if status.get("CHARGING"):
        raise ValueError(
            f"{status['CHARGING']} of {len(tel)} samples are CHARGING. While "
            f"charging the current's sign alternates between samples under "
            f"constant load, so no energy can be integrated (ADR-0047). This "
            f"trace cannot produce an energy result."
        )

    currents = [r["current_ua"] for r in tel if r.get("current_ua") is not None]
    if any(c > 0 for c in currents):
        raise ValueError(
            f"{sum(1 for c in currents if c > 0)} positive currents in a "
            f"DISCHARGING trace. The sign convention (negative = discharge, "
            f"ADR-0047) does not hold here and the trace must not be integrated."
        )

    t = np.array([r["t"] for r in tel], dtype=np.float64)
    t -= t[0]
    volts = np.array([r["voltage_mv"] / 1000.0 for r in tel], dtype=np.float64)
    amps = np.array([abs(r["current_ua"]) / 1e6 for r in tel], dtype=np.float64)
    # V*I, not I scaled by a nominal voltage: voltage sags under load (ADR-0047).
    watts = volts * amps
    skin = np.array([r.get("skin_c", np.nan) for r in tel], dtype=np.float64)

    qt = np.array([r["t_wall"] for r in qs], dtype=np.float64)
    qw = np.array([r.get("n_clusters", 0) for r in qs], dtype=np.float64)

    idx = np.searchsorted(qt, t)
    wr = np.zeros(t.size)
    for i in range(1, t.size):
        lo, hi = idx[i - 1], idx[i]
        dt = t[i] - t[i - 1]
        if hi > lo and dt > 0:
            wr[i] = qw[lo:hi].sum() / dt
    w, p = wr[1:], watts[1:]
    good = w > 0
    w, p = w[good], p[good]

    energy_j = float(np.trapezoid(watts, t))
    mean_w = energy_j / float(t[-1])

    # The effort/power relationship, as quintiles rather than a fitted slope.
    # Binning needs no intercept, so it says what the data supports without
    # extrapolating to a work rate never observed.
    edges = np.percentile(w, [0, 20, 40, 60, 80, 100])
    quintiles = []
    for i in range(5):
        m = (w >= edges[i]) & (w <= edges[i + 1])
        if m.sum():
            quintiles.append(
                {
                    "work_lo": float(edges[i]),
                    "work_hi": float(edges[i + 1]),
                    "power_median_w": float(np.median(p[m])),
                    "n": int(m.sum()),
                }
            )

    # The intercept is only meaningful if the data reaches toward zero work.
    # With duty cycling the trace contains genuine idle intervals, so the
    # baseline is INTERPOLATED from measured samples; without them it is
    # extrapolated, and extrapolation is what produced -0.209 W (ADR-0052).
    #
    # `w` above excludes zero-work intervals so the quintiles describe working
    # periods. The regression needs them back, precisely because they are the
    # anchor.
    w_all, p_all = wr[1:], watts[1:]
    n_idle = int((w_all == 0).sum())
    idle_fraction = float(n_idle / w_all.size) if w_all.size else 0.0
    # 5% of intervals at zero work is the line. Below that the intercept is
    # still leaning on a handful of points and the old failure mode is a
    # rounding error away.
    has_baseline = idle_fraction >= 0.05

    design = np.column_stack([w_all, np.ones(w_all.size)])
    coef, *_ = np.linalg.lstsq(design, p_all, rcond=None)
    r2 = float(1.0 - np.sum((p_all - design @ coef) ** 2) / np.sum((p_all - p_all.mean()) ** 2))
    # BOTH conditions. A positive intercept fitted without idle data is a lucky
    # extrapolation, not a measurement, and would be reported with the same
    # confidence as a real one.
    intercept_physical = bool(coef[1] >= 0 and has_baseline)
    measured_idle_w = float(np.median(p_all[w_all == 0])) if n_idle else float("nan")

    result = {
        "trace": {
            "n_queries": len(qs),
            "n_samples": len(tel),
            "minutes": float(t[-1] / 60.0),
            "status": dict(status),
            "all_currents_negative": True,
        },
        "measured": {
            "total_energy_j": energy_j,
            "mean_power_w": mean_w,
            "power_median_w": float(np.median(watts)),
            "power_p5_w": float(np.percentile(watts, 5)),
            "power_p95_w": float(np.percentile(watts, 95)),
            "mj_per_query_allin": energy_j / len(qs) * 1000.0,
            "voltage_range_v": [float(volts.min()), float(volts.max())],
            "skin_range_c": [float(np.nanmin(skin)), float(np.nanmax(skin))],
        },
        "effort_vs_power": {
            "quintiles": quintiles,
            "correlation": float(np.corrcoef(w, p)[0, 1]),
            "work_rate_range": [float(w.min()), float(w.max())],
        },
        "attribution": {
            "reportable": intercept_physical,
            "slope_w_per_cluster_per_s": float(coef[0]),
            "intercept_w": float(coef[1]),
            "r2": r2,
            "idle_intervals": n_idle,
            "idle_fraction": idle_fraction,
            "measured_idle_power_w": measured_idle_w,
            "mj_per_cluster": float(coef[0] * 1000.0) if intercept_physical else None,
            "why_not_reportable": (
                (
                    f"Only {idle_fraction * 100:.1f}% of intervals had zero work "
                    f"(need 5%+), so the intercept is extrapolated rather than "
                    f"measured. Re-run device_trace.py with --idle-s to duty-cycle "
                    f"the workload. "
                    if not has_baseline
                    else ""
                )
                + "The fitted intercept is negative, which would be a phone idling "
                "at negative power. work_rate spans "
                f"{w.min():.0f}-{w.max():.0f} clusters/s and never approaches "
                "zero, so the intercept extrapolates far outside the data. At the "
                "trace's mean effort the marginal term predicts more energy per "
                "query than the measured all-in average, with the negative "
                "intercept absorbing the difference. Per-query attribution needs "
                "IDLE PERIODS inside the trace so the screen-and-radio baseline is "
                "measured rather than extrapolated."
            )
            if not intercept_physical
            else "",
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    m = result["measured"]
    print(f"trace: {len(qs):,} queries, {len(tel)} samples, {t[-1] / 60:.1f} min", flush=True)
    print(f"  status: {dict(status)}, every current negative", flush=True)
    print("\nMEASURED (no model needed):", flush=True)
    print(f"  total energy      {m['total_energy_j']:.1f} J", flush=True)
    print(f"  mean power        {m['mean_power_w']:.3f} W", flush=True)
    print(f"  power p5/p95      {m['power_p5_w']:.3f} / {m['power_p95_w']:.3f} W", flush=True)
    print(f"  all-in per query  {m['mj_per_query_allin']:.2f} mJ", flush=True)
    print("\nEFFORT vs POWER (binned, no intercept required):", flush=True)
    for q in quintiles:
        print(
            f"  work {q['work_lo']:6.1f}-{q['work_hi']:6.1f} clusters/s -> "
            f"{q['power_median_w']:.3f} W  (n={q['n']})",
            flush=True,
        )
    print(f"  correlation {result['effort_vs_power']['correlation']:+.4f}", flush=True)
    if not intercept_physical:
        print(
            f"\nPER-QUERY ATTRIBUTION: NOT REPORTABLE.\n"
            f"  fit gives {coef[0]:.3e} W per cluster/s with intercept "
            f"{coef[1]:+.3f} W (R2 {r2:.3f}).\n"
            f"  A negative idle power is not a smaller version of the right "
            f"answer. The trace needs idle periods.",
            flush=True,
        )
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

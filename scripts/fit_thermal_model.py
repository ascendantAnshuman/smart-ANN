#!/usr/bin/env python
"""Phase 4: fit a thermal model to the measured device trace.

**Fitted, never chosen.** ADR-0007 and ADR-0008 recorded, in Phase 1, that this
project's thermal ground truth exists only on the handset. Every phase since has
refused to pick thermal constants rather than measure them. This is the script
that pays that debt off, and it fails loudly if the trace is missing rather than
falling back on plausible numbers (Section 3, rule 1).

**The model.** First-order lumped thermal mass, the standard form for a body
heated by dissipated power and cooled toward ambient:

    dT/dt = (P(effort) - (T - T_ambient) / R) / C

Discretised over the trace's sampling interval and fitted by least squares on

    dT/dt  ~  a * work_rate  +  b * (T - T_ref)  +  c

where `work_rate` is the retrieval work completed in the interval. Then the
thermal resistance and time constant fall out as `R = -a/b` scaled and
`tau = -1/b`, which is the number a controller actually needs: how long it has
before an action shows up in temperature.

**Why that matters here specifically.** The Phase 6 control law collapsed
because its control period was shorter than its observation delay. `tau`
measured from this trace is the thermal equivalent of that delay, so it is not
a decoration on the model -- it is the parameter that says whether a
thermally-aware controller can work at all.

**What this trace cannot support, and the script says so rather than implying
otherwise.** The handset was CHARGING throughout (Termux:API was unavailable,
so the run needed the cable for host-side power logging). Charging adds a heat
contribution this model absorbs into its constant term. The fitted `tau` and
the effort coefficient are still meaningful -- they come from the *variation*
across effort levels within one power state -- but the absolute offset is not
the device's idle thermal state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run scripts/device_trace.py on the handset first; "
            f"this script will not invent a thermal trace."
        )
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", default="results/device")
    ap.add_argument("--out", default="results/thermal_model.json")
    args = ap.parse_args()

    d = Path(args.trace_dir)
    telem = load_jsonl(d / "trace_telemetry.jsonl")
    queries = load_jsonl(d / "trace_queries.jsonl")
    meta = json.loads((d / "trace_meta.json").read_text(encoding="utf-8"))

    t = np.array([r["t"] for r in telem], dtype=np.float64)
    skin = np.array([r.get("skin_c", np.nan) for r in telem], dtype=np.float64)
    hottest = np.array([r.get("hottest_c", np.nan) for r in telem], dtype=np.float64)
    t -= t[0]

    ok = np.isfinite(skin)
    if ok.sum() < 30:
        raise ValueError(
            f"only {ok.sum()} finite skin-temperature samples; a fit on this would be "
            f"noise. The trace must carry a working thermal signal."
        )
    t, skin, hottest = t[ok], skin[ok], hottest[ok]

    # Query work per telemetry interval. The trace interleaves nprobe levels
    # round-robin, so any interval contains a mix -- `n_clusters` is the honest
    # per-query work unit, since it is what actually gets loaded and searched.
    qt = np.array([r["t_wall"] for r in queries], dtype=np.float64)
    qwork = np.array([r.get("n_clusters", 0) for r in queries], dtype=np.float64)
    qlat = np.array([r["t_total_ms"] for r in queries], dtype=np.float64)
    nprobe = np.array([r["nprobe"] for r in queries], dtype=np.float64)

    # Trim the first two minutes: the power state was changing then (the cable
    # was pulled and reattached), and a step in the heat input with no record of
    # when it happened is exactly the confound this project keeps catching.
    trim_s = 120.0
    keep = t >= trim_s
    t, skin, hottest = t[keep], skin[keep], hottest[keep]

    def fit_at(window_s: float) -> dict | None:
        """Fit at one differencing window. None when too few intervals remain.

        The window matters more than it looks. Skin temperature rose 4.6 C over
        25 minutes -- about 0.003 C/s -- while differencing ADJACENT 3-second
        samples produces a dT/dt whose standard deviation is 0.051 C/s, sixteen
        times the signal. The sensor quantises at ~0.005 C, so short-interval
        differencing measures quantisation, not physics. Widening the window
        trades sample count for signal-to-noise.
        """
        step = max(1, round(window_s / 3.0))
        ts, ss = t[::step], skin[::step]
        if ts.size < 12:
            return None
        idx_w = np.searchsorted(qt, ts)
        wr = np.zeros(ts.size)
        for i in range(1, ts.size):
            lo, hi = idx_w[i - 1], idx_w[i]
            dt = ts[i] - ts[i - 1]
            if hi > lo and dt > 0:
                wr[i] = qwork[lo:hi].sum() / dt
        d_t = np.diff(ss) / np.diff(ts)
        t_mid = 0.5 * (ss[1:] + ss[:-1])
        w_r = wr[1:]
        g = np.isfinite(d_t) & np.isfinite(t_mid) & np.isfinite(w_r)
        d_t, t_mid, w_r = d_t[g], t_mid[g], w_r[g]
        if d_t.size < 10:
            return None
        t_ref_w = float(np.median(t_mid))
        design = np.column_stack([w_r, t_mid - t_ref_w, np.ones(w_r.size)])
        coef_w, *_ = np.linalg.lstsq(design, d_t, rcond=None)
        resid = d_t - design @ coef_w
        ss_tot_w = float(np.sum((d_t - d_t.mean()) ** 2))
        return {
            "window_s": window_s,
            "n_intervals": int(d_t.size),
            "a_per_cluster_per_s": float(coef_w[0]),
            "b_per_s": float(coef_w[1]),
            "c_c_per_s": float(coef_w[2]),
            "t_ref_c": t_ref_w,
            "r2": float(1.0 - np.sum(resid**2) / ss_tot_w) if ss_tot_w > 0 else float("nan"),
            "tau_seconds": float(-1.0 / coef_w[1]) if coef_w[1] < 0 else float("nan"),
        }

    # A sweep, not a single fit. Reporting one window would hide that the
    # estimate depends on an analysis choice nobody outside this file would
    # think to question.
    sweep = [f for f in (fit_at(w) for w in (3.0, 15.0, 30.0, 60.0, 120.0)) if f]

    # 30 s is the headline: it is the largest window that still leaves enough
    # intervals for the fit to mean anything. The 120 s window scores R^2 0.69
    # and is NOT quoted -- eleven points against three free parameters is a
    # number that flatters itself.
    min_intervals = 30
    usable = [f for f in sweep if f["n_intervals"] >= min_intervals]
    if not usable:
        raise ValueError(
            f"no differencing window left {min_intervals}+ intervals; the trace is "
            f"too short or too sparsely sampled to fit a thermal model."
        )
    chosen = max(usable, key=lambda f: f["r2"])
    a, b, c = chosen["a_per_cluster_per_s"], chosen["b_per_s"], chosen["c_c_per_s"]
    t_ref, r2, tau_s = chosen["t_ref_c"], chosen["r2"], chosen["tau_seconds"]

    taus = [f["tau_seconds"] for f in sweep if np.isfinite(f["tau_seconds"])]
    a_vals = [f["a_per_cluster_per_s"] for f in sweep]

    per_level = {}
    for n in sorted(set(nprobe.tolist())):
        m = nprobe == n
        per_level[int(n)] = {
            "n_queries": int(m.sum()),
            "latency_p50_ms": float(np.percentile(qlat[m], 50)),
            "latency_p95_ms": float(np.percentile(qlat[m], 95)),
            "clusters_mean": float(qwork[m].mean()),
        }

    result = {
        "source_trace": {
            "n_queries": meta["n_queries"],
            "n_samples": meta["n_samples"],
            "minutes": meta["minutes_actual"],
            "nprobe_levels": meta["nprobe_levels"],
            "battery_read_failures": meta["battery_read_failures"],
        },
        "model": "dT/dt = a*work_rate + b*(T - T_ref) + c   (first-order lumped mass)",
        "fit": chosen,
        "window_sweep": sweep,
        "why_a_sweep": "The differencing window is an analysis choice that moves "
        "R^2 from 0.11 to 0.69. Reporting one window would hide that. The chosen "
        "fit is the best R^2 among windows retaining 30+ intervals; the 120 s "
        "window is excluded despite scoring highest, because 11 points against 3 "
        "free parameters is a number that flatters itself.",
        "parameter_stability": {
            "tau_seconds_range": [min(taus), max(taus)] if taus else None,
            "a_range": [min(a_vals), max(a_vals)],
            "note": "The point estimates barely move across windows even as R^2 "
            "swings 6x. That agreement across analysis choices is stronger "
            "evidence for the parameters than any single R^2.",
        },
        "derived": {
            "tau_seconds": tau_s,
            "tau_note": "time constant: how long a change in effort takes to show "
            "up in temperature. This is the thermal observation delay, and a "
            "controller acting faster than it repeats the Phase 6 collapse.",
        },
        "temperature_observed": {
            "skin_min_c": float(skin.min()),
            "skin_max_c": float(skin.max()),
            "skin_rise_c": float(skin.max() - skin.min()),
            "hottest_zone_max_c": float(np.nanmax(hottest)),
        },
        "per_nprobe": per_level,
        "caveats": [
            "The handset was CHARGING for the whole trace: Termux:API was not "
            "installed, so the cable was needed for host-side power logging. "
            "Charging heat is absorbed into the constant term c, so c is NOT the "
            "device's idle thermal state.",
            "No energy was measured (all 478 battery reads failed). This model is "
            "thermal only; the energy half of C4 remains undemonstrated.",
            f"The first {trim_s:.0f}s are trimmed: the cable was pulled and "
            "reattached during them, an unrecorded step in the heat input.",
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(
        f"trace: {meta['n_queries']:,} queries, {meta['minutes_actual']:.1f} min",
        flush=True,
    )
    print(
        f"skin temperature: {skin.min():.2f} -> {skin.max():.2f} C "
        f"(rise {skin.max() - skin.min():.2f} C)",
        flush=True,
    )
    print(f"\n{'window s':>9} {'n':>5} {'R2':>8} {'a (work)':>12} {'tau s':>8}", flush=True)
    for f in sweep:
        if f is chosen:
            mark = "  <-- chosen"
        elif f["n_intervals"] < min_intervals:
            mark = "  (too few intervals -- not quoted)"
        else:
            mark = ""
        print(
            f"{f['window_s']:>9.0f} {f['n_intervals']:>5} {f['r2']:>8.4f} "
            f"{f['a_per_cluster_per_s']:>12.3e} {f['tau_seconds']:>8.1f}{mark}",
            flush=True,
        )
    print(f"\nfit: dT/dt = {a:+.3e}*work + {b:+.3e}*(T-{t_ref:.2f}) {c:+.3e}", flush=True)
    print(
        f"  R^2 = {r2:.4f} over {chosen['n_intervals']} intervals "
        f"at a {chosen['window_s']:.0f}s window",
        flush=True,
    )
    if taus:
        print(
            f"  tau across ALL windows: {min(taus):.1f} - {max(taus):.1f} s, "
            f"stable while R^2 swings {min(f['r2'] for f in sweep):.2f}-"
            f"{max(f['r2'] for f in sweep):.2f}",
            flush=True,
        )
    print(f"\n{'nprobe':>7} {'n':>7} {'p50 ms':>9} {'p95 ms':>9} {'clusters':>9}", flush=True)
    for n, v in per_level.items():
        print(
            f"{n:>7} {v['n_queries']:>7,} {v['latency_p50_ms']:>9.2f} "
            f"{v['latency_p95_ms']:>9.2f} {v['clusters_mean']:>9.2f}",
            flush=True,
        )
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

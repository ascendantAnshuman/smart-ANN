#!/usr/bin/env python
"""Phase 3.5: the device trace. Runs ON the handset, in Termux.

The first measurement in this project taken on the target device rather than a
laptop standing in for one. It runs the real EcoVector index with the real
query stream and samples energy and temperature alongside every query, so that
Phase 4's thermal constants can be **fitted** from a trace rather than chosen
(ADR-0007, ADR-0008).

**Why it is self-contained.** It logs to `/sdcard` and needs no network and no
adb connection while running, so the phone can be unplugged the moment it
starts. That is not a convenience: energy cannot be measured through a charging
cable, and the current's sign is unstable on AC (ADR-0047).

**Sampling is decoupled from the query loop.** A background thread samples the
battery every `--sample-period` seconds; the query loop never waits on it.
`termux-battery-status` takes hundreds of milliseconds, so sampling inline
would make the controller's own telemetry the dominant cost of every query --
measuring the instrument instead of the system.

**Effort sweeps so the trace covers the space a controller would use.** A trace
at one nprobe teaches a thermal model nothing about how effort maps to heat.
Levels are interleaved round-robin rather than blocked, because a blocked sweep
measures drift as much as configuration (ADR-0030) and this run is long enough
and hot enough for drift to be the point rather than a nuisance.

Run it as:

    python /sdcard/smartann/device_trace.py --minutes 25
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path("/sdcard/smartann")
sys.path.insert(0, str(ROOT))

BATTERY_BIN = "/data/data/com.termux/files/usr/bin/termux-battery-status"
_SENSOR_DISABLED = -273000
# Chosen from measurement: skin swung 4.8 C over 60 s of load where the battery
# zone moved 1.2 C (ADR-0047). The battery zone is not a substitute.
_PREFERRED = ("skin-msm-therm", "shell_front", "skin-therm")


def read_zones() -> tuple[float, float]:
    """(preferred zone C, hottest live zone C). Disabled sensors excluded."""
    live: dict[str, float] = {}
    for z in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            t = (z / "type").read_text().strip()
            v = int((z / "temp").read_text().strip())
        except (OSError, ValueError):
            continue
        # -273000 is a sensor that is OFF, not a device at absolute zero. 21 of
        # this handset's 110 zones report it, and an unfiltered aggregate reads
        # as a cold device -- the direction that makes a controller do nothing.
        if v != _SENSOR_DISABLED:
            live[t] = v / 1000.0
    if not live:
        return float("nan"), float("nan")
    pref = next((live[z] for z in _PREFERRED if z in live), float("nan"))
    return pref, max(live.values())


def read_battery() -> dict:
    out = subprocess.run(
        [BATTERY_BIN], capture_output=True, text=True, timeout=25, check=False
    ).stdout
    lo, hi = out.find("{"), out.rfind("}")
    if lo < 0 or hi <= lo:
        raise ValueError("no JSON in battery output")
    return json.loads(out[lo : hi + 1])


class Sampler(threading.Thread):
    """Background telemetry. The query loop never blocks on it."""

    def __init__(self, period_s: float, out_path: Path) -> None:
        super().__init__(daemon=True)
        self.period_s = period_s
        self.out_path = out_path
        self.rows: list[dict] = []
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            t = time.time()
            pref, hottest = read_zones()
            row = {"t": t, "skin_c": pref, "hottest_c": hottest}
            try:
                b = read_battery()
                row.update(
                    current_ua=b.get("current"),
                    voltage_mv=b.get("voltage"),
                    battery_c=b.get("temperature"),
                    pct=b.get("percentage"),
                    status=b.get("status"),
                    charge_counter=b.get("charge_counter"),
                )
            except Exception as exc:
                # A failed read is missing data, not zero power. Recorded as
                # such so a gap in the trace is visible rather than imputed.
                row["battery_error"] = str(exc)[:120]
            self.rows.append(row)
            self._stop.wait(self.period_s)

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=30)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=25.0)
    ap.add_argument("--sample-period", type=float, default=3.0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="/sdcard/smartann/trace")
    ap.add_argument("--nprobe", type=int, nargs="*", default=[1, 4, 8, 16, 32])
    args = ap.parse_args()

    from smartann.index.ecovector import EcoVectorIndex
    from smartann.index.storage import IndexPaths
    from smartann.types import AnnConfig

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    index = EcoVectorIndex.open(IndexPaths(ROOT / "index" / "squad").root)
    queries = np.load(ROOT / "data" / "queries.npy").astype(np.float32)
    print(f"index open, {queries.shape[0]} queries", flush=True)

    cfgs = {
        n: AnnConfig(nprobe=n, ef_centroid=128, ef_list=64, k=args.k, per_cluster_k=32)
        for n in args.nprobe
    }

    # Warm every level before the trace starts, so the first samples are not a
    # cold-start artifact attributed to whichever level happened to run first.
    for n in args.nprobe:
        for i in range(5):
            p = index.probe(queries[i], cfgs[n])
            index.finish(queries[i], p, cfgs[n])
    print("warmed", flush=True)

    sampler = Sampler(args.sample_period, out / "telemetry.jsonl")
    sampler.start()

    rows = []
    t_end = time.time() + args.minutes * 60.0
    qi = 0
    t0_wall = time.time()
    while time.time() < t_end:
        nprobe = args.nprobe[qi % len(args.nprobe)]
        row_i = qi % queries.shape[0]
        cfg = cfgs[nprobe]
        t0 = time.perf_counter()
        probe = index.probe(queries[row_i], cfg)
        res = index.finish(queries[row_i], probe, cfg)
        dt = (time.perf_counter() - t0) * 1000.0
        rows.append(
            {
                "t_wall": time.time() - t0_wall,
                "nprobe": nprobe,
                "query_id": row_i,
                "t_total_ms": dt,
                "t_centroid_ms": probe.t_centroid_ms,
                "n_clusters": int(res.ids.size > 0) and len(probe.cluster_ids),
                "bytes_read": int(getattr(res, "bytes_read", 0)),
            }
        )
        qi += 1
        if qi % 500 == 0:
            el = time.time() - t0_wall
            print(f"  {qi} queries, {el / 60:.1f} min, {len(sampler.rows)} samples", flush=True)

    sampler.stop()

    (out / "queries.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    (out / "telemetry.jsonl").write_text(
        "\n".join(json.dumps(r) for r in sampler.rows) + "\n", encoding="utf-8"
    )
    plugged = [r.get("status") for r in sampler.rows if r.get("status")]
    (out / "meta.json").write_text(
        json.dumps(
            {
                "minutes_requested": args.minutes,
                "minutes_actual": (time.time() - t0_wall) / 60.0,
                "n_queries": len(rows),
                "n_samples": len(sampler.rows),
                "nprobe_levels": args.nprobe,
                "sample_period_s": args.sample_period,
                "k": args.k,
                # If ANY sample says CHARGING the energy trace is not usable
                # for a discharge integration, and that must be visible in the
                # metadata rather than discovered during analysis.
                "charging_samples": sum(1 for s in plugged if s == "CHARGING"),
                "discharging_samples": sum(1 for s in plugged if s == "DISCHARGING"),
                "battery_read_failures": sum(1 for r in sampler.rows if "battery_error" in r),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}: {len(rows)} queries, {len(sampler.rows)} telemetry samples", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

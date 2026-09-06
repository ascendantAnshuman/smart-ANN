#!/usr/bin/env python
"""C5: a device-state-aware policy against a static baseline, ON the handset.

Runs in Termux. This is the experiment that moves C5 from *built* to
*demonstrated*: `DeviceStatePolicy` has passed 25 unit tests against synthetic
state, and unit tests prove logic, not behaviour. Until it reads a real
thermometer on a warming phone and the reading changes what it does, the claim
is about code.

**The disturbance is the device itself.** Earlier controller experiments needed
CPU burners to manufacture pressure. Here the workload generates its own: 20+
minutes of continuous retrieval raises skin temperature ~4.6 C (ADR-0051), and
the policy's job is to notice and back off. Nothing is simulated.

**Interleaved, not blocked.** Static and adaptive alternate every query, so the
thermal ramp both arms experience is identical. A blocked design would give the
second arm a hotter phone and measure the ramp rather than the policy --
ADR-0030, and the mistake that manufactured a false C3 verdict (ADR-0052).

**The static baseline is tuned, not crippled** (Section 3, rule 3). It is set
to the highest effort level whose DEVICE-measured p95 fits the budget, which is
exactly what a competent engineer would ship after profiling. Beating a
deliberately bad baseline would prove nothing.

**What would falsify C5.** If the adaptive arm derates but its violation rate
is no better than static's, device state is not carrying useful information and
C5 fails. That outcome is reportable and the script does not steer away from
it: it records both arms' violations, recall and chosen effort, and the
temperature at every decision.
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
_PREFERRED = ("skin-msm-therm", "shell_front", "skin-therm")


def read_zones() -> tuple[float, float]:
    live: dict[str, float] = {}
    for z in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            t = (z / "type").read_text().strip()
            v = int((z / "temp").read_text().strip())
        except (OSError, ValueError):
            continue
        if v != _SENSOR_DISABLED:
            live[t] = v / 1000.0
    if not live:
        return float("nan"), float("nan")
    pref = next((live[z] for z in _PREFERRED if z in live), float("nan"))
    return pref, max(live.values())


def read_battery() -> dict:
    out = subprocess.run(
        [BATTERY_BIN], capture_output=True, text=True, timeout=8, check=False
    ).stdout
    lo, hi = out.find("{"), out.rfind("}")
    if lo < 0 or hi <= lo:
        raise ValueError("no JSON in battery output")
    return json.loads(out[lo : hi + 1])


class Telemetry(threading.Thread):
    """Background device state. The query loop reads `latest` without blocking.

    Sampling inline would put a several-hundred-millisecond battery read inside
    every controller decision, making the instrument the dominant cost of the
    thing it measures.
    """

    def __init__(self, period_s: float = 5.0) -> None:
        super().__init__(daemon=True)
        self.period_s = period_s
        self.rows: list[dict] = []
        self.latest: dict = {}
        self._prev: tuple[float, float] | None = None
        self._fails = 0
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            skin, hottest = read_zones()
            slope = float("nan")
            if self._prev is not None:
                pt, pv = self._prev
                dt = now - pt
                if dt > 0 and skin == skin and pv == pv:
                    slope = (skin - pv) / dt
            self._prev = (now, skin)
            row = {"t": now, "skin_c": skin, "hottest_c": hottest, "slope": slope}
            if self._fails < 3:
                try:
                    b = read_battery()
                    self._fails = 0
                    row.update(
                        current_ua=b.get("current"),
                        voltage_mv=b.get("voltage"),
                        pct=b.get("percentage"),
                        status=b.get("status"),
                    )
                except Exception as exc:
                    self._fails += 1
                    row["battery_error"] = str(exc)[:100]
            self.rows.append(row)
            self.latest = row
            self._stop.wait(self.period_s)

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=20)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=22.0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--budget-ms", type=float, default=110.0)
    ap.add_argument("--sample-period", type=float, default=5.0)
    ap.add_argument(
        "--headroom",
        type=float,
        default=1.0,
        help="DeviceStatePolicy headroom. MUST be 1.0 for a fair comparison: at "
        "0.9 the adaptive arm silently gets a 99 ms budget against static's 110 ms, "
        "which excludes nprobe=8 (device p95 101.5 ms) before device state is "
        "consulted at all. The first run of this experiment did exactly that and "
        "measured headroom rather than device awareness (ADR-0055).",
    )
    ap.add_argument("--out", default="/sdcard/smartann/c5")
    args = ap.parse_args()

    from smartann.controller.policies.device import DeviceStatePolicy
    from smartann.controller.profile import ProfileTable
    from smartann.index.ecovector import EcoVectorIndex
    from smartann.index.storage import IndexPaths
    from smartann.types import (
        AnnConfig,
        Constraints,
        ControllerState,
        DeviceState,
        PerfHistory,
        QueryFeatures,
        ScrConfig,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    table = ProfileTable.read(ROOT / "device_profile_table.json")
    index = EcoVectorIndex.open(IndexPaths(ROOT / "index" / "squad").root)
    queries = np.load(ROOT / "data" / "queries.npy").astype(np.float32)
    gt = np.load(ROOT / "data" / "gt_ids.npy")

    scr_off = ScrConfig(1, 0, 0, 3, 0.0, enabled=False)
    base = AnnConfig(nprobe=8, ef_centroid=128, ef_list=64, k=args.k, per_cluster_k=32)
    constraints = Constraints(args.budget_ms, args.budget_ms * 3, float("inf"), None)

    # The static arm is the best DEVICE-measured level that fits the budget --
    # what a competent engineer ships after profiling, not a strawman.
    # BOTH arms start from the same effective budget. Without this the
    # comparison is between two budgets, and device state never gets a vote.
    static_entry = table.best_within(args.budget_ms * args.headroom)
    print(
        f"budget {args.budget_ms:.0f} ms x headroom {args.headroom:.2f} "
        f"-> static baseline nprobe="
        f"{static_entry.nprobe} (device p95 {static_entry.p95_ms:.1f} ms, "
        f"recall {static_entry.recall:.4f})",
        flush=True,
    )

    policy = DeviceStatePolicy(table, base, scr_off, headroom=args.headroom)
    policy.reset()

    telem = Telemetry(args.sample_period)
    telem.start()
    time.sleep(args.sample_period + 1)  # one real sample before the first decision

    def state_now() -> ControllerState:
        r = telem.latest
        pct = r.get("pct")
        status = str(r.get("status", "")).upper()
        dev = DeviceState(
            timestamp=r.get("t", time.time()),
            temperature_c=float(r.get("skin_c", float("nan"))),
            temperature_slope_c_per_s=float(r.get("slope", float("nan"))),
            freq_ratio=float("nan"),
            throttled=False,
            cpu_util=float("nan"),
            mem_available_mb=float("nan"),
            mem_pressure=float("nan"),
            battery_pct=float(pct) if pct is not None else float("nan"),
            charging=(status == "CHARGING"),
            discharge_rate_ma=float("nan"),
        )
        return ControllerState(
            dev, QueryFeatures(0.0, 0.0, 0.0, 1.0, 0), PerfHistory(0, 0, 0, 0, 0, 0, 0, 0)
        )

    cfgs: dict[int, AnnConfig] = {}

    def cfg_for(n: int) -> AnnConfig:
        if n not in cfgs:
            cfgs[n] = AnnConfig(nprobe=n, ef_centroid=128, ef_list=64, k=args.k, per_cluster_k=32)
        return cfgs[n]

    # Warm every level so the first queries are not a cold-start artifact.
    for n in (e.nprobe for e in table.entries):
        for i in range(5):
            c = cfg_for(n)
            index.finish(queries[i], index.probe(queries[i], c), c)
    print("warmed", flush=True)

    rows = []
    t0 = time.time()
    t_end = t0 + args.minutes * 60.0
    qi = 0
    while time.time() < t_end:
        # Interleave, so both arms climb the same thermal ramp.
        arm = "static" if qi % 2 == 0 else "adaptive"
        row_i = qi % queries.shape[0]
        st = state_now()

        t_dec = time.perf_counter()
        if arm == "static":
            nprobe = static_entry.nprobe
            reasons: list[str] = []
        else:
            nprobe = policy.decide(st, constraints).ann.nprobe
            reasons = policy.reasons()
        overhead_us = (time.perf_counter() - t_dec) * 1e6

        c = cfg_for(nprobe)
        t_q = time.perf_counter()
        probe = index.probe(queries[row_i], c)
        res = index.finish(queries[row_i], probe, c)
        lat = (time.perf_counter() - t_q) * 1000.0

        hits = len(set(res.ids[: args.k].tolist()) & set(gt[row_i, : args.k].tolist()))
        rows.append(
            {
                "t_wall": time.time() - t0,
                "arm": arm,
                "nprobe": nprobe,
                "query_id": row_i,
                "t_total_ms": lat,
                "recall_at_k": hits / args.k,
                "violation": bool(lat > args.budget_ms),
                "overhead_us": overhead_us,
                "skin_c": st.device.temperature_c,
                "slope": st.device.temperature_slope_c_per_s,
                "battery_pct": st.device.battery_pct,
                "reasons": reasons,
            }
        )
        qi += 1
        if qi % 1000 == 0:
            el = (time.time() - t0) / 60.0
            print(
                f"  {qi} queries, {el:.1f} min, skin "
                f"{st.device.temperature_c:.1f} C, {len(telem.rows)} samples",
                flush=True,
            )

    telem.stop()

    (out / "queries.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    (out / "telemetry.jsonl").write_text(
        "\n".join(json.dumps(r) for r in telem.rows) + "\n", encoding="utf-8"
    )

    summary = {}
    print(f"\n{'arm':<10} {'n':>6} {'p95 ms':>9} {'viol %':>8} {'recall':>8} {'nprobe p50':>11}")
    for arm in ("static", "adaptive"):
        a = [r for r in rows if r["arm"] == arm]
        lat = np.array([r["t_total_ms"] for r in a])
        summary[arm] = {
            "n": len(a),
            "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95)),
            "violation_pct": float(np.mean([r["violation"] for r in a]) * 100),
            "recall": float(np.mean([r["recall_at_k"] for r in a])),
            "nprobe_p50": float(np.percentile([r["nprobe"] for r in a], 50)),
            "overhead_p95_us": float(np.percentile([r["overhead_us"] for r in a], 95)),
        }
        s = summary[arm]
        print(
            f"{arm:<10} {s['n']:>6} {s['p95_ms']:>9.2f} {s['violation_pct']:>7.1f}% "
            f"{s['recall']:>8.4f} {s['nprobe_p50']:>11.0f}",
            flush=True,
        )

    skins = [r["skin_c"] for r in rows if r["skin_c"] == r["skin_c"]]
    derated = [r for r in rows if r["arm"] == "adaptive" and r["reasons"]]
    (out / "summary.json").write_text(
        json.dumps(
            {
                "budget_ms": args.budget_ms,
                "headroom": args.headroom,
                "same_effective_budget": True,
                "minutes": (time.time() - t0) / 60.0,
                "static_baseline": {
                    "nprobe": static_entry.nprobe,
                    "device_p95_ms": static_entry.p95_ms,
                    "recall": static_entry.recall,
                    "selection": "highest-recall level whose DEVICE-measured p95 fits "
                    "the budget -- tuned, not crippled (Section 3, rule 3)",
                },
                "arms": summary,
                "thermal": {
                    "skin_min_c": min(skins) if skins else None,
                    "skin_max_c": max(skins) if skins else None,
                    "rise_c": (max(skins) - min(skins)) if skins else None,
                },
                "adaptive_decisions_with_a_reason": len(derated),
                "adaptive_decisions_total": sum(1 for r in rows if r["arm"] == "adaptive"),
                "note": "The adaptive arm reads a real thermometer. Decisions with an "
                "empty reason list are decisions taken blind -- if that count is the "
                "total, device state carried no information and C5 fails.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"\nskin {min(skins):.2f} -> {max(skins):.2f} C" if skins else "\nno thermal signal",
        flush=True,
    )
    print(
        f"adaptive decisions citing device state: {len(derated)} / "
        f"{sum(1 for r in rows if r['arm'] == 'adaptive')}",
        flush=True,
    )
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""E-DEMO: adaptive retrieval effort under CPU contention.

The demonstration the review needs, and the one the phone's absence does not
block. The AIMD governor observes p95 latency against a target and moves
`nprobe`. It does not know *why* latency rose -- contention, thermal throttling
and a hard query are the same signal to it -- which is exactly why showing it
against contention shows the mechanism that will later face thermal throttling.

Structure: quiet, contended, quiet. Sized in SECONDS, not queries, because
under contention the adaptive arm answers more queries in the same wall-clock
window precisely because it cut its effort. A fixed query count would give the
two arms different amounts of time and hide the effect.

Everything measured. No simulated thermal data, no modelled energy.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_acceptance import index_root, load_all

from smartann.bench.runner import RollingWindow, RunSpec, run_queries
from smartann.bench.writer import RunWriter
from smartann.config import resolve_paths
from smartann.controller import AimdConfig, AimdPolicy, StaticPolicy
from smartann.index.ecovector import EcoVectorIndex
from smartann.index.storage import IndexMeta
from smartann.telemetry.windows import WindowsMonitor
from smartann.types import AnnConfig, Constraints, ScrConfig

SCR_OFF = ScrConfig(3, 1, 1, 2, 0.0, enabled=False)


def _burn(stop: mp.Event) -> None:  # type: ignore[name-defined]
    """Spin until told to stop. Deliberately trivial: the point is CPU demand."""
    x = 0.0
    while not stop.is_set():
        for _ in range(100_000):
            x += 1.000000001
    _ = x


class Burners:
    """A block of spinning processes, started and stopped around a phase."""

    def __init__(self, n: int) -> None:
        self.n = n
        self._stop: mp.Event | None = None  # type: ignore[name-defined]
        self._procs: list[mp.Process] = []

    def start(self) -> None:
        self._stop = mp.Event()
        self._procs = [mp.Process(target=_burn, args=(self._stop,)) for _ in range(self.n)]
        for p in self._procs:
            p.start()

    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        self._procs = []


def tune_static(index, queries, gt_ids, gt_dists, candidates, seconds=6.0):  # type: ignore[no-untyped-def]
    """Pick the best static nprobe under NO contention, and record how.

    Section 3 rule 3: the static baseline must be tuned, not crippled. The
    criterion is the largest nprobe whose quiet p95 stays under the budget --
    i.e. the most recall a static config can buy while meeting its SLO when
    nothing is wrong. That is the strongest honest baseline: a static policy
    cannot know contention is coming.
    """
    from smartann.bench.metrics import recall_at_k_batch

    rows = []
    for npb in candidates:
        cfg = AnnConfig(nprobe=npb, ef_centroid=128, ef_list=64, k=10, per_cluster_k=32)
        lat: list[float] = []
        t_end = time.time() + seconds
        collected_i, collected_d, rows_used = [], [], []
        qi = 0
        while time.time() < t_end:
            t0 = time.perf_counter()
            r = index.search(queries[qi % queries.shape[0]], cfg)
            lat.append((time.perf_counter() - t0) * 1000.0)
            pad_i = np.full(10, -1, dtype=np.int64)
            pad_d = np.full(10, np.inf, dtype=np.float64)
            m = min(r.ids.size, 10)
            pad_i[:m], pad_d[:m] = r.ids[:m], r.dists[:m]
            collected_i.append(pad_i)
            collected_d.append(pad_d)
            rows_used.append(qi % queries.shape[0])
            qi += 1
        se, _ = recall_at_k_batch(
            np.stack(collected_i),
            np.stack(collected_d),
            gt_ids[rows_used],
            gt_dists[rows_used],
            k=10,
        )
        arr = np.asarray(lat)
        rows.append((npb, float(np.percentile(arr, 95)), float(se.mean()), len(lat)))
        print(
            f"    nprobe={npb:>3}  quiet p95 {rows[-1][1]:7.2f} ms  recall {se.mean():.4f} "
            f"({len(lat)} queries)",
            flush=True,
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift1m")
    ap.add_argument("--quiet-s", type=float, default=60.0)
    ap.add_argument("--contended-s", type=float, default=120.0)
    ap.add_argument("--recovery-s", type=float, default=60.0)
    ap.add_argument("--burners", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = resolve_paths()
    ip = index_root(args.dataset, "ecovector")
    if ip is None:
        print(f"no ecovector index built for {args.dataset}", file=sys.stderr)
        return 1
    meta = IndexMeta.from_json(ip.meta)
    index = EcoVectorIndex.open(ip.root)
    _, queries, gt_ids, gt_dists = load_all(args.dataset)
    queries = np.ascontiguousarray(queries[:2000], dtype=np.float32)
    gt_ids, gt_dists = gt_ids[:2000], gt_dists[:2000]

    out_root = Path(args.out) if args.out else paths.results_root / "E_DEMO"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"=== tuning the static baseline on {args.dataset}, quiet ===", flush=True)
    tuning = tune_static(index, queries, gt_ids, gt_dists, candidates=(4, 8, 16, 32, 64))

    # Target at ~1.5x the best quiet p95, so a tuned static config sits inside
    # the budget when idle and outside it when contended. Fixed HERE, before the
    # contention run, and not touched afterwards.
    best_npb, best_p95, _best_recall, _ = max(tuning, key=lambda r: r[0])
    target_ms = 1.5 * best_p95
    aimd_cfg = AimdConfig(
        nprobe_min=1,
        nprobe_max=64,
        additive_increase=0.05,
        multiplicative_decrease=0.5,
        low_water=0.8,
        warmup_queries=20,
        effort_init=1.0,
    )
    note = (
        f"static nprobe={best_npb} chosen as the largest candidate whose quiet p95 "
        f"({best_p95:.2f} ms) is met by the target; target = 1.5 * that = {target_ms:.2f} ms. "
        f"AIMD parameters fixed before the contention run and unchanged after."
    )
    print(f"\n  {note}\n", flush=True)

    constraints = Constraints(
        latency_target_ms=target_ms,
        latency_hard_ms=3.0 * best_p95,
        energy_budget_mj=float("inf"),
        min_recall=None,
    )
    base_ann = AnnConfig(nprobe=best_npb, ef_centroid=128, ef_list=64, k=10, per_cluster_k=32)

    policies = {
        "static": StaticPolicy(base_ann, SCR_OFF, tuning_note=note),
        "aimd": AimdPolicy(aimd_cfg, base_ann, SCR_OFF),
    }

    for name, policy in policies.items():
        print(f"=== running {name} ===", flush=True)
        policy.reset()
        monitor = WindowsMonitor(hz=10.0).start()
        writer = RunWriter(
            out_root / name,
            run_id=f"{name}",
            experiment_id="E_DEMO",
            capabilities=monitor.capabilities(),
        )
        spec = RunSpec(args.dataset, meta.index_variant, meta.config_hash)
        window = RollingWindow(size=50)
        burners = Burners(args.burners)
        try:
            writer.event("phase_quiet_start")
            n1 = run_queries(
                index,
                queries,
                policy,
                constraints,
                monitor,
                writer,
                spec,
                gt_ids=gt_ids,
                gt_dists=gt_dists,
                duration_s=args.quiet_s,
                window=window,
            )
            writer.event("burners_start", workers=args.burners)
            burners.start()
            n2 = run_queries(
                index,
                queries,
                policy,
                constraints,
                monitor,
                writer,
                spec,
                gt_ids=gt_ids,
                gt_dists=gt_dists,
                duration_s=args.contended_s,
                window=window,
            )
            burners.stop()
            writer.event("burners_stop")
            n3 = run_queries(
                index,
                queries,
                policy,
                constraints,
                monitor,
                writer,
                spec,
                gt_ids=gt_ids,
                gt_dists=gt_dists,
                duration_s=args.recovery_s,
                window=window,
            )
            writer.event("phase_recovery_end")
        finally:
            burners.stop()
            monitor.stop()

        parquet, _manifest = writer.write(
            extra={
                "policy": name,
                "tuning_note": note,
                "latency_target_ms": target_ms,
                "burners": args.burners,
                "phase_seconds": {
                    "quiet": args.quiet_s,
                    "contended": args.contended_s,
                    "recovery": args.recovery_s,
                },
                "aimd": aimd_cfg.__dict__ if name == "aimd" else None,
                "static_tuning": [
                    {"nprobe": r[0], "quiet_p95_ms": r[1], "recall": r[2], "n": r[3]}
                    for r in tuning
                ],
            }
        )
        print(f"  {n1} quiet + {n2} contended + {n3} recovery = {n1 + n2 + n3} queries", flush=True)
        print(f"  {parquet}", flush=True)

    print(json.dumps({"out": str(out_root), "target_ms": target_ms}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

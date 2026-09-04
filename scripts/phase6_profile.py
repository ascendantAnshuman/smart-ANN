#!/usr/bin/env python
"""Phase 6: build the profile table, then compare three policies against it.

Two steps, in order, because the second depends on the first.

**Step 1 — profile.** Sweep `nprobe` and record what each level actually costs
and returns. This is a table of measurements, not a fitted model: Phase 3's
latency model holds at 14.75% MAPE inside its fitted regime and fails at 35.87%
outside it because `cpu_util` is censored (ADR-0033), so a policy that predicted
from it would be most confident where it is most wrong.

**Step 2 — compare.** Static, AIMD and the budget policy against one declared
latency budget, on the same query stream, in the same process, interleaved by
arm order. The budget is fixed from the profile table before any policy runs, so
no policy is judged against a target derived from itself.

Energy is absent from all of this and that is deliberate: no energy can be
measured here (ADR-0007/0008) and `BudgetPolicy.honours_energy_budget()` returns
False, which every manifest records.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_acceptance import index_root, load_all

from smartann.bench.metrics import recall_at_k_batch
from smartann.controller import (
    AimdConfig,
    AimdPolicy,
    BudgetPolicy,
    StaticPolicy,
    build_profile_table,
)
from smartann.index.ecovector import EcoVectorIndex, ProbeResult
from smartann.types import AnnConfig, Constraints, ControllerState, PerfHistory, ScrConfig

SCR_OFF = ScrConfig(3, 1, 1, 2, 0.0, enabled=False)
NPROBE_GRID = (1, 2, 4, 8, 12, 16, 24, 32)


def _history(lat: list[float], target: float) -> PerfHistory:
    if not lat:
        return PerfHistory(0, 0, 0, 0, 0, 0, 0, 0)
    w = np.asarray(lat[-50:])
    return PerfHistory(
        float(w.mean()),
        float(np.percentile(w, 50)),
        float(np.percentile(w, 95)),
        float(np.percentile(w, 99)),
        0.0,
        float(np.mean(w > target)),
        len(w),
        0.0,
    )


def _run_once(index, q, cfg, gt_ids, gt_dists, row, k):  # type: ignore[no-untyped-def]
    probe = index.probe(q, cfg)
    want = min(cfg.nprobe, probe.cluster_ids.size)
    narrowed = ProbeResult(
        probe.cluster_ids[:want],
        probe.centroid_dists[:want],
        probe.features,
        probe.t_centroid_ms,
        probe.n_dist_modeled,
    )
    r = index.finish(q, narrowed, cfg)
    ids = np.full((1, k), -1, dtype=np.int64)
    d = np.full((1, k), np.inf, dtype=np.float64)
    m = min(r.ids.size, k)
    ids[0, :m], d[0, :m] = r.ids[:m], r.dists[:m]
    se, _ = recall_at_k_batch(ids, d, gt_ids[row : row + 1], gt_dists[row : row + 1], k=k)
    return r, probe, float(se[0])


def profile(index, queries, gt_ids, gt_dists, seconds: float, k: int) -> pd.DataFrame:
    """Sweep nprobe INTERLEAVED, recording every query.

    Round-robin across effort levels rather than one block per level, per
    ADR-0030. A blocked sweep measures the machine's drift as much as the code,
    and the first blocked attempt proved it: nprobe=1 recorded a 38.8 ms p50
    against nprobe=4's 4.8 ms, because the first level ran against a cold index
    and the later ones did not. That inverted the latency ordering entirely, and
    the resulting Pareto front declared the four cheapest levels "dominated" --
    a table the budget policy would then have used to rule out exactly the
    configurations it should reach for under pressure.

    Interleaving spreads any drift across every level instead of loading it onto
    whichever ran first, and a warm-up pass absorbs the cold start before any
    measurement is taken.
    """
    cfgs = {
        n: AnnConfig(nprobe=n, ef_centroid=128, ef_list=64, k=k, per_cluster_k=32)
        for n in NPROBE_GRID
    }
    # Warm every level before timing any of them.
    for n in NPROBE_GRID:
        for i in range(20):
            _run_once(index, queries[i], cfgs[n], gt_ids, gt_dists, i, k)

    rows = []
    counts = dict.fromkeys(NPROBE_GRID, 0)
    t_end = time.time() + seconds * len(NPROBE_GRID)
    qi = 0
    while time.time() < t_end:
        nprobe = NPROBE_GRID[qi % len(NPROBE_GRID)]
        row = qi % queries.shape[0]
        r, probe, se = _run_once(index, queries[row], cfgs[nprobe], gt_ids, gt_dists, row, k)
        rows.append(
            {
                "cfg_ann_nprobe": nprobe,
                "t_total": r.t_total + probe.t_centroid_ms,
                "recall_at_k": se,
                "context_tokens": 0.0,
            }
        )
        counts[nprobe] += 1
        qi += 1
    for n, c in counts.items():
        print(f"  nprobe={n:>3}: {c:>6,} queries", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift1m")
    ap.add_argument("--profile-seconds", type=float, default=25.0)
    ap.add_argument("--compare-seconds", type=float, default=90.0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="results/phase6")
    args = ap.parse_args()

    ip = index_root(args.dataset, "ecovector")
    if ip is None:
        print(f"no ecovector index for {args.dataset}", file=sys.stderr)
        return 1
    index = EcoVectorIndex.open(ip.root)
    _, queries, gt_ids, gt_dists = load_all(args.dataset)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"=== step 1: profiling {args.dataset} ===", flush=True)
    frame = profile(index, queries, gt_ids, gt_dists, args.profile_seconds, args.k)
    table = build_profile_table(frame, args.dataset, measured_under="quiet machine, warm cache")
    table.write(out / "profile_table.json")
    print(f"\n{'nprobe':>7} {'p50 ms':>9} {'p95 ms':>9} {'recall':>8} {'n':>8}", flush=True)
    for e in table.entries:
        print(
            f"{e.nprobe:>7} {e.p50_ms:>9.3f} {e.p95_ms:>9.3f} {e.recall:>8.4f} {e.n_samples:>8,}",
            flush=True,
        )
    front = [e.nprobe for e in table.pareto_front()]
    print(f"\npareto front (nprobe): {front}", flush=True)
    print(
        f"dominated and never worth choosing: "
        f"{[e.nprobe for e in table.entries if e.nprobe not in front]}",
        flush=True,
    )

    # Budget fixed from the TABLE before any policy runs, so no policy is judged
    # against a target derived from its own behaviour.
    mid = table.entries[len(table.entries) // 2]
    budget_ms = mid.p95_ms
    print(
        f"\nlatency budget = p95 of the median swept level (nprobe={mid.nprobe}) "
        f"= {budget_ms:.2f} ms\n",
        flush=True,
    )
    constraints = Constraints(budget_ms, 3 * budget_ms, float("inf"), None)
    base = AnnConfig(nprobe=mid.nprobe, ef_centroid=128, ef_list=64, k=args.k, per_cluster_k=32)

    policies = {
        "static": StaticPolicy(
            base, SCR_OFF, tuning_note=f"nprobe={mid.nprobe}, the budget anchor"
        ),
        "aimd": AimdPolicy(
            AimdConfig(nprobe_min=1, nprobe_max=32, warmup_queries=20, control_period_s=4.05),
            base,
            SCR_OFF,
        ),
        "budget": BudgetPolicy(table, base, SCR_OFF),
    }

    print("=== step 2: three policies, one budget ===", flush=True)
    summary = []
    for name, policy in policies.items():
        policy.reset()
        lat, rec, npb, ovh = [], [], [], []
        t_end = time.time() + args.compare_seconds
        qi = 0
        while time.time() < t_end:
            row = qi % queries.shape[0]
            state = ControllerState(
                __import__(
                    "smartann.telemetry", fromlist=["unavailable_state"]
                ).unavailable_state(),
                _null_features(),
                _history(lat, budget_ms),
            )
            d = policy.decide_timed(state, constraints)
            r, probe, se = _run_once(
                index, queries[row], d.config.ann, gt_ids, gt_dists, row, args.k
            )
            lat.append(r.t_total + probe.t_centroid_ms)
            rec.append(se)
            npb.append(d.config.ann.nprobe)
            ovh.append(d.overhead_us)
            qi += 1
        a = np.asarray(lat)
        s = {
            "policy": name,
            "n": len(lat),
            "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)),
            "violation_pct": float(np.mean(a > budget_ms) * 100),
            "recall": float(np.mean(rec)),
            "nprobe_p50": float(np.percentile(npb, 50)),
            "overhead_p95_pct": float(np.percentile(np.asarray(ovh) / 1000.0 / a * 100, 95)),
            "honours_energy_budget": getattr(policy, "honours_energy_budget", lambda: False)(),
        }
        summary.append(s)
        print(
            f"  {name:<7} n={s['n']:>6,}  p95 {s['p95_ms']:7.2f} ms  "
            f"viol {s['violation_pct']:5.1f}%  recall {s['recall']:.4f}  "
            f"nprobe_p50 {s['nprobe_p50']:.0f}  overhead p95 {s['overhead_p95_pct']:.3f}%",
            flush=True,
        )

    (out / "policy_comparison.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "budget_ms": budget_ms,
                "budget_derivation": f"p95 of the median swept level (nprobe={mid.nprobe}), "
                f"fixed from the profile table before any policy ran",
                "energy": "NOT evaluated: no energy is measurable on this platform "
                "(ADR-0007/0008), so BudgetPolicy.honours_energy_budget() is False",
                "arms": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out}/profile_table.json and {out}/policy_comparison.json", flush=True)
    return 0


def _null_features():  # type: ignore[no-untyped-def]
    from smartann.types import QueryFeatures

    return QueryFeatures(0.0, 0.0, 0.0, 1.0, 0)


if __name__ == "__main__":
    sys.exit(main())

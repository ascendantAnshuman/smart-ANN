#!/usr/bin/env python
"""Phase 9: does per-query effort allocation beat uniform effort at equal cost?

**The gap this closes.** `probe()` already computes query-difficulty signals --
how far the nearest centroid is, how much it beats the second, the entropy over
all of them -- and hands them to every controller decision. Until now no policy
read any of them. Every query at a given budget got identical effort, which is
the uniform allocation this project exists to argue against, applied to the one
axis where the information was already free.

**The claim under test.** Effort spent on an easy query is wasted: if one
cluster clearly dominates, opening sixteen more finds nothing. Moving that
effort to an ambiguous query should buy recall the uniform policy cannot. The
claim is therefore not "adaptive is better" but the sharper, falsifiable one:
*at equal mean device cost, allocating effort by measured difficulty yields
higher mean recall than spending it uniformly.*

**Why this can be measured here.** Recall is a property of the index and the
query, not of the machine -- the same query at the same nprobe returns the same
neighbours on the laptop and on the handset. Latency is not, so every cost in
this experiment is taken from the measured device profile table rather than
timed here. The allocation is therefore optimised against handset milliseconds
while the recall behind it is measured locally, and neither number is invented.

**Fitted, never chosen.** The bucket edges are quantiles of the measured gap
distribution. The level assigned to each bucket is whatever maximises measured
mean recall under the cost constraint, found by exhaustive search over all
combinations -- small enough to enumerate, so no optimiser and no tolerance to
defend. Nothing here is a round number someone picked.

**Held out.** Buckets and allocation are fitted on one half of the queries and
scored on the other. Fitting difficulty buckets and then reporting the gain on
the same queries would measure memorisation, and the gain would be real-looking
and worthless.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_acceptance import index_root, load_all

from smartann.bench.metrics import recall_at_k_batch
from smartann.controller.profile import ProfileTable
from smartann.index.ecovector import EcoVectorIndex, ProbeResult
from smartann.pipeline.ask import load_profile_table
from smartann.types import AnnConfig

# The signal. Named once so the report, the policy and this script cannot
# disagree about which of the three probe features was actually used.
SIGNAL = "centroid_dist_gap"
SIGNAL_MEANING = (
    "how far the second nearest centroid is beyond the nearest, as a distance "
    "difference rather than a ratio -- a ratio would divide by a distance that "
    "can be zero or negative for duplicate vectors. A large difference means one "
    "cluster clearly dominates, so the query is easy; a difference near zero "
    "means the query sits between clusters and its neighbours are scattered."
)


def measure(
    index: EcoVectorIndex,
    queries: np.ndarray,
    gt_ids: np.ndarray,
    gt_dists: np.ndarray,
    levels: list[int],
    k: int,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Per-query difficulty, and per-query recall at every effort level.

    The probe runs once per query at the widest level and the result is narrowed
    for each level below it, which is exactly what `ask` and the bench runner do.
    Re-probing per level would measure a different search.
    """
    widest = max(levels)
    n = len(queries)
    gaps = np.zeros(n, dtype=np.float64)
    ids_at: dict[int, np.ndarray] = {lv: np.zeros((n, k), dtype=np.int64) for lv in levels}
    dists_at: dict[int, np.ndarray] = {lv: np.zeros((n, k), dtype=np.float64) for lv in levels}

    for i, q in enumerate(queries):
        vec = np.asarray(q, dtype=np.float32)
        probe_cfg = AnnConfig(nprobe=widest, ef_centroid=128, ef_list=64, k=k, per_cluster_k=32)
        probe = index.probe(vec, probe_cfg)
        assert probe.features is not None
        gaps[i] = getattr(probe.features, SIGNAL)

        for lv in levels:
            want = min(lv, probe.cluster_ids.size)
            narrowed = ProbeResult(
                cluster_ids=probe.cluster_ids[:want],
                centroid_dists=probe.centroid_dists[:want],
                features=probe.features,
                t_centroid_ms=probe.t_centroid_ms,
                n_dist_modeled=probe.n_dist_modeled,
            )
            cfg = AnnConfig(nprobe=lv, ef_centroid=128, ef_list=64, k=k, per_cluster_k=32)
            res = index.finish(vec, narrowed, cfg)
            got = min(k, res.ids.size)
            ids_at[lv][i, :got] = res.ids[:got]
            dists_at[lv][i, :got] = res.dists[:got]
            if got < k:  # a level too small to return k; pad beyond any threshold
                dists_at[lv][i, got:] = np.inf
                ids_at[lv][i, got:] = -1

        if (i + 1) % 100 == 0:
            print(f"  measured {i + 1}/{n} queries", flush=True)

    recall_at: dict[int, np.ndarray] = {}
    for lv in levels:
        _, overlap = recall_at_k_batch(ids_at[lv], dists_at[lv], gt_ids, gt_dists, k)
        recall_at[lv] = overlap
    return gaps, recall_at


def best_allocation(
    recall_by_bucket: np.ndarray,  # [n_buckets, n_levels]
    weights: np.ndarray,  # [n_buckets], share of queries
    costs: np.ndarray,  # [n_levels], measured device ms
    budget_ms: float,
) -> tuple[tuple[int, ...], float, float]:
    """The assignment of a level to each bucket that maximises mean recall.

    Exhaustive over every combination. With five buckets and five levels that is
    3,125 evaluations, so enumerating is both exact and faster to defend than an
    optimiser whose tolerance would need justifying.
    """
    n_buckets, n_levels = recall_by_bucket.shape
    best: tuple[int, ...] = tuple([0] * n_buckets)
    best_recall = -1.0
    best_cost = float("inf")
    for combo in itertools.product(range(n_levels), repeat=n_buckets):
        cost = float(sum(weights[b] * costs[combo[b]] for b in range(n_buckets)))
        if cost > budget_ms:
            continue
        rec = float(sum(weights[b] * recall_by_bucket[b, combo[b]] for b in range(n_buckets)))
        if rec > best_recall or (rec == best_recall and cost < best_cost):
            best, best_recall, best_cost = combo, rec, cost
    return best, best_recall, best_cost


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="squad")
    ap.add_argument("--variant", default="ecovector")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--buckets", type=int, default=5)
    ap.add_argument("--n-queries", type=int, default=0, help="0 means all of them.")
    ap.add_argument("--table", type=Path, default=Path("results/device_profile_table.json"))
    ap.add_argument("--out", type=Path, default=Path("results/phase9/query_adaptive.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--reuse",
        action="store_true",
        help="Re-derive the allocation from the cached measurement instead of searching again.",
    )
    args = ap.parse_args()

    paths = index_root(args.dataset, args.variant)
    if paths is None:
        print(
            f"no {args.variant} index for {args.dataset}; run `smartann index build`",
            file=sys.stderr,
        )
        return 2

    table: ProfileTable = load_profile_table(args.table, args.dataset)
    levels = sorted(e.nprobe for e in table.entries)
    costs = np.array([table.entry_for(lv).p95_ms for lv in levels], dtype=np.float64)  # type: ignore[union-attr]

    _, queries, gt_ids, gt_dists = load_all(args.dataset)
    if args.n_queries:
        queries = queries[: args.n_queries]
        gt_ids, gt_dists = gt_ids[: args.n_queries], gt_dists[: args.n_queries]

    # The measurement is the expensive part and is a durable artifact in its own
    # right: seventeen minutes of index reads that nothing about the analysis
    # changes. Caching it means the allocation and the verdict can be re-derived
    # without re-running the search, and re-derived from exactly the same
    # numbers rather than from a second run that might differ.
    cache = args.out.with_name("measurement.npz")
    if args.reuse and cache.exists():
        blob = np.load(cache)
        if list(blob["levels"]) != levels or len(blob["gaps"]) != len(queries):
            print(f"{cache} does not match this run; delete it to re-measure", file=sys.stderr)
            return 2
        gaps = blob["gaps"]
        recall_at = {lv: blob[f"recall_{lv}"] for lv in levels}
        print(f"reusing the measurement in {cache} ({len(gaps):,} queries)", flush=True)
    else:
        index = EcoVectorIndex.open(paths.root)
        print(f"measuring {len(queries):,} queries at levels {levels} ...", flush=True)
        gaps, recall_at = measure(index, queries, gt_ids, gt_dists, levels, args.k)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache,
            gaps=gaps,
            levels=np.array(levels),
            **{f"recall_{lv}": recall_at[lv] for lv in levels},
        )

    # Fit and test halves, fixed by seed so the split is reproducible.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(queries))
    fit_idx, test_idx = order[: len(order) // 2], order[len(order) // 2 :]

    # Bucket edges from the FIT half only. Using every query's gap would let the
    # test half influence the buckets it is then scored in.
    qs = np.linspace(0, 100, args.buckets + 1)[1:-1]
    edges = np.percentile(gaps[fit_idx], qs)

    def bucket_of(values: np.ndarray) -> np.ndarray:
        return np.digitize(values, edges)

    fit_b, test_b = bucket_of(gaps[fit_idx]), bucket_of(gaps[test_idx])

    recall_fit = np.zeros((args.buckets, len(levels)), dtype=np.float64)
    weights_fit = np.zeros(args.buckets, dtype=np.float64)
    for b in range(args.buckets):
        m = fit_b == b
        weights_fit[b] = m.mean()
        for j, lv in enumerate(levels):
            recall_fit[b, j] = float(recall_at[lv][fit_idx][m].mean()) if m.any() else np.nan

    # One comparison per uniform level: give the adaptive policy exactly the
    # mean device cost that level would spend, and see what recall it returns.
    comparisons = []
    for j, lv in enumerate(levels):
        budget = float(costs[j])
        combo, _, fitted_cost = best_allocation(recall_fit, weights_fit, costs, budget)

        # Held out: apply the fitted assignment to queries it was never fitted on.
        chosen_levels = np.array([levels[combo[b]] for b in range(args.buckets)])
        adaptive_recall = np.array(
            [recall_at[int(chosen_levels[b])][test_idx][i] for i, b in enumerate(test_b)]
        )
        adaptive_cost = float(np.mean([costs[combo[b]] for b in test_b]))
        uniform_recall = recall_at[lv][test_idx]

        comparisons.append(
            {
                "uniform_nprobe": lv,
                "budget_ms": budget,
                "assignment": [int(x) for x in chosen_levels],
                "fitted_mean_cost_ms": fitted_cost,
                "held_out": {
                    "n": len(test_idx),
                    "adaptive_recall": float(adaptive_recall.mean()),
                    "uniform_recall": float(uniform_recall.mean()),
                    "delta_recall": float(adaptive_recall.mean() - uniform_recall.mean()),
                    "adaptive_mean_cost_ms": adaptive_cost,
                    "uniform_mean_cost_ms": float(costs[j]),
                },
            }
        )

    # The smallest difference this many held-out queries can resolve. One query
    # gaining one neighbour moves mean recall by exactly this much, so a delta
    # of a few multiples of it is a handful of queries, not an effect.
    resolution = 1.0 / (len(test_idx) * args.k)

    deltas = [c["held_out"]["delta_recall"] for c in comparisons]
    savings = [
        1.0 - c["held_out"]["adaptive_mean_cost_ms"] / c["held_out"]["uniform_mean_cost_ms"]
        for c in comparisons
    ]
    wins = [d for d in deltas if d > 3 * resolution]
    losses = [d for d in deltas if d < -3 * resolution]
    mean_saving = float(np.mean(savings))

    # The allocation was constrained to the budget on the FIT half. The held-out
    # half's difficulty mix is not identical, so the cost it actually spends
    # drifts. "At equal cost" is only true within that drift, and the largest
    # overspend is what a reader needs in order to judge whether a recall gain
    # was bought with time rather than with better allocation.
    worst_overspend = max((-s for s in savings), default=0.0)

    # A majority of budgets must improve before this is called support. One win
    # in five is the distribution of noise, and reporting it as a result is how
    # a project talks itself into a contribution it does not have.
    if len(wins) > len(comparisons) / 2:
        verdict = (
            f"C6 SUPPORTED: allocating effort by {SIGNAL} beat uniform effort on "
            f"{len(wins)} of {len(comparisons)} budgets, by up to {max(deltas):+.4f} recall "
            f"on {len(test_idx):,} held-out queries. The allocation was constrained to each "
            f"budget on the fit half; on held-out queries it spent at most "
            f"{100 * worst_overspend:.1f}% more device time than the uniform arm it beat, so "
            f"the comparison is at near-equal cost rather than exactly equal cost."
        )
    elif mean_saving > 0.05 and not losses:
        verdict = (
            f"C6 PARTIALLY SUPPORTED: recall did not improve at matched budgets, but the "
            f"allocation reached the same recall for {100 * mean_saving:.0f}% less measured "
            f"device time on average, with no budget losing more than the "
            f"{3 * resolution:.4f} this query set can resolve. The gain is cost, not quality."
        )
    else:
        verdict = (
            f"C6 NOT demonstrated: allocating by {SIGNAL} improved recall on {len(wins)} of "
            f"{len(comparisons)} budgets and lost on {len(losses)}, against a resolution of "
            f"{resolution:.4f} recall per query-neighbour. The signal is computed and free, "
            f"but on this index it does not predict which queries need more clusters well "
            f"enough to beat spending the same time uniformly."
        )

    # How much the signal separates easy from hard at all, reported whether or
    # not the allocation won. A flat column here explains a null result.
    separation = [
        {
            "bucket": b,
            "gap_lo": float(gaps[fit_idx][fit_b == b].min()) if (fit_b == b).any() else None,
            "gap_hi": float(gaps[fit_idx][fit_b == b].max()) if (fit_b == b).any() else None,
            "share": float(weights_fit[b]),
            "recall_by_level": {
                str(levels[j]): float(recall_fit[b, j]) for j in range(len(levels))
            },
        }
        for b in range(args.buckets)
    ]

    out = {
        "dataset": args.dataset,
        "k": args.k,
        "signal": SIGNAL,
        "signal_meaning": SIGNAL_MEANING,
        "n_queries": len(queries),
        "n_fit": len(fit_idx),
        "n_test": len(test_idx),
        "seed": args.seed,
        "levels": levels,
        "level_cost_ms": [float(c) for c in costs],
        "cost_source": table.measured_under,
        "bucket_edges": [float(e) for e in edges],
        "separation": separation,
        "comparisons": comparisons,
        "recall_resolution": resolution,
        "mean_cost_saving": mean_saving,
        "worst_overspend": worst_overspend,
        "verdict": verdict,
        "method": (
            "Recall measured locally because it is a property of the index and the query, "
            "not of the machine. Every cost is the handset's measured p95 from the profile "
            "table. Buckets and allocation fitted on half the queries, scored on the other "
            "half."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    print(verdict)
    for c in comparisons:
        h = c["held_out"]
        print(
            f"  uniform nprobe={c['uniform_nprobe']:<3} recall {h['uniform_recall']:.4f}"
            f"  ->  adaptive {h['adaptive_recall']:.4f}  ({h['delta_recall']:+.4f})"
            f"   cost {h['uniform_mean_cost_ms']:.1f} -> {h['adaptive_mean_cost_ms']:.1f} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

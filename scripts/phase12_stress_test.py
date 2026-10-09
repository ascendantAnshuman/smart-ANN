#!/usr/bin/env python
"""12.4 -- fixed depth against the controller, over one recorded device trace.

Three hundred queries, twice: once at a fixed maximum depth, once with the
controller reading the handset's recorded condition and choosing a level from
the measured profile table. Same queries, same order, same trace, same index.
The only difference between the arms is the policy, which is the only way the
comparison means anything.

**The device state is replayed, not invented.** `ReplayFeed` walks the C5
telemetry trace, so temperature and battery across the run are what the phone
actually reported. The trace warms as it goes, so the controller's choice has
something to follow.

**The latency is this laptop's, and the recall is exact.** Retrieval runs here,
so the milliseconds are this machine's and are labelled that way. Recall@10 is
scored against brute-force nearest neighbours over the same 255 vectors, which
is exact rather than approximate, so a depth that loses recall is visible rather
than argued.

**The energy column is an estimate and says so in the header.** Clusters opened
are counted here; the joules per cluster come from the handset's measured
power-against-cluster-rate fit. It answers "what would the phone have spent on
this retrieval", not "what did this laptop spend".

**Three arms, because the thermometer turns out to decide the result.** The
handset exposes two sensors that disagree by tens of degrees, and the policy's
thresholds (warm 42 C, hot 46 C) were anchored on SKIN measurements. Fed the
hottest internal zone, which never drops below 47 C anywhere in the C5 trace,
the policy is derated on every single query. Fed the skin sensor, which never
reaches 42 C, it never derates at all. Neither produces a controller that
visibly tracks the device, and running only one of them would have reported that
as either a success or a failure depending on which was picked.

So both are run. The difference between them is ADR-0056's finding -- the
thermal signal is real but not actionable through these thresholds -- made
quantitative instead of argued.

**What this can and cannot show.** It can show whether the controller gives up
recall, how much, and what it buys. It cannot show a thermal benefit: the heat
in the trace was produced by the phone's own run, and nothing done here changes
it. Claiming the controller kept the phone cooler would require running both
arms on the phone, which is C5's unfinished business (ADR-0056).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smartann.config import resolve_paths
from smartann.controller.policies.device import DeviceStatePolicy
from smartann.controller.policies.static import StaticPolicy
from smartann.controller.profile import ProfileTable
from smartann.index.ecovector import EcoVectorIndex
from smartann.index.storage import resolve_index_root
from smartann.pipeline.embedder import Embedder
from smartann.telemetry.replay import ReplayFeed, load_trace, trace_path
from smartann.types import (
    AnnConfig,
    Constraints,
    ControllerState,
    PerfHistory,
    ScrConfig,
)

DATASET = "demo_phone_care"
K = 10
SCR_OFF = ScrConfig(1, 0, 0, 1, 0.0, enabled=False)
COLD_HISTORY = PerfHistory(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)


def load_queries(spec_path: Path, docs_path: Path) -> list[str]:
    """The evaluation queries plus every document title, for variety.

    Twenty queries cycled fifteen times would measure one small set of
    neighbourhoods repeatedly. Titles are natural questions a reader of this
    corpus might ask and spread the load across the whole index.
    """
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    docs = json.loads(docs_path.read_text(encoding="utf-8"))
    queries = list(spec["paraphrases"])
    queries += [item["query"] for item in spec["off_topic"]]
    queries += [d["title"] for d in docs]
    return queries


def mj_per_cluster(path: Path) -> tuple[float, str]:
    if not path.exists():
        return float("nan"), "no energy model on disk; energy is not estimated"
    model = json.loads(path.read_text(encoding="utf-8"))
    att = model["attribution"]
    return float(att["mj_per_cluster"]), (
        f"estimate: clusters opened x {float(att['mj_per_cluster']):.3f} mJ/cluster, "
        f"the handset's measured power-vs-cluster-rate fit (r2 = {float(att['r2']):.3f})"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--trace", default="c5")
    ap.add_argument("--table", type=Path, default=Path("results/demo_profile_table.json"))
    ap.add_argument("--out-csv", type=Path, default=Path("results/phase12_fixed_vs_adaptive.csv"))
    ap.add_argument("--out-png", type=Path, default=Path("results/phase12_fixed_vs_adaptive.png"))
    ap.add_argument("--out-json", type=Path, default=Path("results/phase12_fixed_vs_adaptive.json"))
    ap.add_argument("--allow-busy", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from phase2_acceptance import assert_machine_quiet

    busy_note = ""
    try:
        assert_machine_quiet()
    except RuntimeError as exc:
        if not args.allow_busy:
            print(exc)
            return 2
        busy_note = " TAKEN ON A BUSY MACHINE (--allow-busy): latencies are not quotable."
        print(f"WARNING:{busy_note}")

    ip = resolve_index_root(DATASET, "ecovector")
    if ip is None:
        print("no demo index. Run scripts/phase11_build_demo_index.py first.")
        return 2
    if not args.table.exists():
        print(f"no profile table at {args.table}. Run scripts/phase11_demo_eval.py first.")
        return 2

    paths = resolve_paths()
    proc = Path(paths.data_root) / "processed" / DATASET
    base = np.load(proc / "base.npy").astype(np.float32)
    index = EcoVectorIndex.open(ip.root)
    table = ProfileTable.read(args.table)
    if table.dataset != DATASET:
        print(f"{args.table} was measured on '{table.dataset}', not '{DATASET}'.")
        return 2

    device_dir = Path(paths.results_root) / "device"
    if not device_dir.is_dir():
        device_dir = Path("results") / "device"
    traces = {
        sensor: load_trace(trace_path(device_dir, args.trace), sensor=sensor)
        for sensor in ("hottest", "skin")
    }
    trace = traces["hottest"]

    embedder = Embedder(cache_path=Path(paths.data_root) / "cache" / "demo.sqlite")
    _ = embedder.model
    queries = load_queries(
        Path("data/demo_phone_care/eval_queries.json"),
        Path("data/demo_phone_care/documents.json"),
    )
    vecs = np.asarray(embedder.encode(queries), dtype=np.float32)
    exact = [{int(i) for i in np.argsort(-(base @ v))[:K]} for v in vecs]

    n_clusters = int(np.count_nonzero(index.sizes > 0))
    max_nprobe = n_clusters
    median_level = table.entries[len(table.entries) // 2]
    budget_ms = 1.5 * median_level.p95_ms
    mj, energy_method = mj_per_cluster(Path("results/energy_model.json"))

    print(f"{DATASET}: {base.shape[0]} chunks, {n_clusters} clusters, k={K}")
    print(f"  {len(queries)} distinct queries, cycled to {args.queries} per arm")
    print(f"  trace {args.trace}: {len(trace)} samples, peak {trace.peak_c:.1f} C")
    print(
        f"  budget = 1.5 x measured p95 at the median level "
        f"(nprobe={median_level.nprobe}, {median_level.p95_ms:.2f} ms) = {budget_ms:.2f} ms"
    )

    base_ann = AnnConfig(nprobe=max_nprobe, ef_centroid=128, ef_list=64, k=K, per_cluster_k=32)
    constraints = Constraints(
        latency_target_ms=budget_ms,
        latency_hard_ms=3.0 * budget_ms,
        energy_budget_mj=float("inf"),
        min_recall=None,
    )

    # (policy, which thermometer it reads). The fixed arm reads none, and is
    # given the hottest trace only so that its rows carry the same temperature
    # column as the others.
    arms = {
        "fixed": (
            StaticPolicy(base_ann, SCR_OFF, tuning_note=f"fixed nprobe={max_nprobe}"),
            "hottest",
        ),
        "adaptive-hottest": (DeviceStatePolicy(table, base_ann, SCR_OFF), "hottest"),
        "adaptive-skin": (DeviceStatePolicy(table, base_ann, SCR_OFF), "skin"),
    }

    rows: list[dict[str, Any]] = []
    for arm_name, (policy, sensor) in arms.items():
        policy.reset()
        # A fresh feed per arm, so both arms see the SAME device states in the
        # same order. Sharing one feed would give the second arm the tail of the
        # trace and the comparison would be between two different runs.
        feed = ReplayFeed(traces[sensor], over_queries=args.queries)
        # Warm every level once before timing, as the device trace itself does.
        for npb in (1, max_nprobe):
            index.search(vecs[0], AnnConfig(npb, 128, 64, K, 32))

        for i in range(args.queries):
            qi = i % len(queries)
            state, label = feed.next()
            cstate = ControllerState(
                device=state,
                query=index.probe(vecs[qi], base_ann).features,
                history=COLD_HISTORY,
            )
            decision = policy.decide_timed(cstate, constraints)
            ann = decision.config.ann

            t0 = time.perf_counter()
            res = index.search(vecs[qi], ann)
            elapsed = (time.perf_counter() - t0) * 1000.0

            got = {int(x) for x in res.ids[:K].tolist()}
            rows.append(
                {
                    "arm": arm_name,
                    "i": i,
                    "query": queries[qi],
                    "sensor": sensor,
                    "t_trace_s": state.timestamp - traces[sensor].samples[0].t,
                    "temperature_c": state.temperature_c,
                    "battery_pct": state.battery_pct,
                    "nprobe": ann.nprobe,
                    "latency_ms": elapsed,
                    "recall_at_10": len(got & exact[qi]) / K,
                    "n_cluster_loads": int(res.n_cluster_loads),
                    "energy_mj_estimate": int(res.n_cluster_loads) * mj,
                    "overhead_us": decision.overhead_us,
                    "reasons": "; ".join(getattr(policy, "reasons", lambda: [])()),
                    "device_source": label,
                }
            )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, Any] = {}
    print()
    for arm_name in arms:
        a = [r for r in rows if r["arm"] == arm_name]
        lat = np.array([r["latency_ms"] for r in a])
        rec = np.array([r["recall_at_10"] for r in a])
        npb = np.array([r["nprobe"] for r in a], dtype=float)
        energy = np.array([r["energy_mj_estimate"] for r in a])
        summary[arm_name] = {
            "n": len(a),
            "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95)),
            "recall_at_10": float(rec.mean()),
            "energy_mj_total_estimate": float(energy.sum()),
            "energy_mj_per_query_estimate": float(energy.mean()),
            "nprobe_p50": float(np.median(npb)),
            "nprobe_min": int(npb.min()),
            "nprobe_max": int(npb.max()),
            "clusters_opened_total": int(sum(r["n_cluster_loads"] for r in a)),
            "over_budget_pct": 100.0 * float(np.mean(lat > budget_ms)),
            "overhead_p95_us": float(np.percentile([r["overhead_us"] for r in a], 95)),
        }
        s = summary[arm_name]
        print(
            f"  {arm_name:9s} p50 {s['p50_ms']:6.2f} ms  p95 {s['p95_ms']:6.2f} ms  "
            f"recall@10 {s['recall_at_10']:.4f}  nprobe p50 {s['nprobe_p50']:4.0f}  "
            f"energy {s['energy_mj_per_query_estimate']:6.2f} mJ/query  "
            f"over budget {s['over_budget_pct']:5.1f}%"
        )

    fixed = summary["fixed"]
    deltas = {}
    for arm_name in ("adaptive-hottest", "adaptive-skin"):
        a = summary[arm_name]
        deltas[arm_name] = {
            "recall_given_up": fixed["recall_at_10"] - a["recall_at_10"],
            "p95_saved_ms": fixed["p95_ms"] - a["p95_ms"],
            "p95_saved_pct": 100.0 * (1.0 - a["p95_ms"] / fixed["p95_ms"]),
            "energy_saved_pct_estimate": 100.0
            * (1.0 - a["energy_mj_total_estimate"] / fixed["energy_mj_total_estimate"]),
            "clusters_avoided": fixed["clusters_opened_total"] - a["clusters_opened_total"],
        }
    print()
    for arm_name, d in deltas.items():
        print(
            f"  {arm_name:17s} gives up {d['recall_given_up']:.4f} recall@10 for "
            f"{d['p95_saved_pct']:5.1f}% lower p95 and "
            f"{d['energy_saved_pct_estimate']:5.1f}% less estimated retrieval energy"
        )
    print()
    print(
        "  the two adaptive arms differ ONLY in which thermometer they read, which is "
        "why the sensor is reported beside every number."
    )

    _plot(rows, summary, budget_ms, trace, args.out_png, busy_note)

    args.out_json.write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "n_queries_per_arm": args.queries,
                "n_distinct_queries": len(queries),
                "k": K,
                "trace": args.trace,
                "budget_ms": budget_ms,
                "budget_derivation": (
                    f"1.5 x the measured p95 of the median level in {args.table}"
                ),
                "arms": summary,
                "deltas": deltas,
                "energy_method": energy_method,
                "provenance": {
                    "latency": "measured on THIS LAPTOP, retrieval only" + busy_note,
                    "recall": f"exact: recall@{K} against brute force over the same vectors",
                    "device_state": f"replayed from the measured handset trace {args.trace}",
                    "energy": energy_method,
                    "not_shown": (
                        "no thermal benefit is claimed. The heat in the trace was produced "
                        "by the phone's own recorded run; nothing done here changed it."
                    ),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out_csv}\nwrote {args.out_png}\nwrote {args.out_json}")
    return 0


def _plot(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    budget_ms: float,
    trace: Any,
    out: Path,
    busy_note: str,
) -> None:
    plt.rcParams.update(
        {"figure.dpi": 130, "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3}
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    colours = {"fixed": "#b3382c", "adaptive-hottest": "#1f7a4d", "adaptive-skin": "#2c6fb3"}
    names = list(colours)

    ax = axes[0][0]
    for arm, c in colours.items():
        a = [r for r in rows if r["arm"] == arm]
        ax.step(
            [r["t_trace_s"] for r in a],
            [r["nprobe"] for r in a],
            where="post",
            color=c,
            lw=1.4,
            label=f"{arm} (median {summary[arm]['nprobe_p50']:.0f})",
        )
    ax.set_ylabel("retrieval depth (nprobe)")
    ax.set_xlabel("time along the recorded trace (s)")
    ax.set_ylim(0, 17)
    ax.set_title("Which thermometer the policy reads decides the level it holds", fontsize=10)
    ax.legend(fontsize=8, loc="center right")

    ax = axes[0][1]
    for sensor, c, lab in (
        ("hottest", "#c2410c", "hottest zone inside"),
        ("skin", "#0369a1", "skin (the case)"),
    ):
        a = [r for r in rows if r["sensor"] == sensor and r["arm"] != "fixed"]
        ax.plot(
            [r["t_trace_s"] for r in a], [r["temperature_c"] for r in a], color=c, lw=1.3, label=lab
        )
    ax.axhline(46.0, color="black", ls="--", lw=1.1, label="policy hot threshold, 46 C")
    ax.axhline(42.0, color="grey", ls=":", lw=1.1, label="policy warm threshold, 42 C")
    ax.set_ylabel("temperature (C), measured on the handset")
    ax.set_xlabel("time along the recorded trace (s)")
    ax.set_title("Why: one sensor is always above both thresholds, one always below", fontsize=10)
    ax.legend(fontsize=7.5, loc="center right")

    x = np.arange(len(names))
    ax = axes[1][0]
    ax.bar(x, [summary[n]["recall_at_10"] for n in names], color=[colours[n] for n in names])
    for i, n in enumerate(names):
        ax.text(
            i,
            summary[n]["recall_at_10"] + 0.012,
            f"{summary[n]['recall_at_10']:.4f}",
            ha="center",
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("recall@10 vs exact search")
    ax.set_title("What the saving costs", fontsize=10)

    ax = axes[1][1]
    width = 0.38
    p95 = [summary[n]["p95_ms"] for n in names]
    energy = [summary[n]["energy_mj_per_query_estimate"] for n in names]
    ax.bar(x - width / 2, p95, width, color="#64748b", label="p95 latency (ms), this laptop")
    ax.bar(
        x + width / 2, energy, width, color="#a16207", label="retrieval energy (mJ/query), ESTIMATE"
    )
    ax.axhline(budget_ms, color="black", ls="--", lw=1.1, label=f"budget {budget_ms:.1f} ms")
    for i in range(len(names)):
        ax.text(i - width / 2, p95[i] * 1.02, f"{p95[i]:.1f}", ha="center", fontsize=8)
        ax.text(i + width / 2, energy[i] * 1.02, f"{energy[i]:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_title("What the saving buys", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right")

    fig.suptitle(
        "Fixed depth against the SMART-ANN controller, over one recorded handset trace\n"
        "300 queries per arm, same queries and same device states; the two adaptive arms "
        "differ only in which thermometer they read\n"
        f"latency measured on this laptop{busy_note}",
        fontsize=10.5,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())

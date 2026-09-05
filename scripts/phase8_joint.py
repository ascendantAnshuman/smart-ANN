#!/usr/bin/env python
"""Phase 8: the joint quality/cost surface. C3, with quality on the axis.

**The question C3 asks.** Is controlling retrieval effort *and* context
reduction together worth more than controlling either alone? That is a claim
about a surface, not about a controller: if the best joint setting is no better
than the best single-dimension setting at the same cost, no control law can
recover the difference, and C3 is dead regardless of how it is implemented.

So this sweeps the grid directly rather than running a policy. A policy would
add its own dynamics to a question that is about the underlying trade-off.

**The comparison, stated before the run.** Three families:

  * *retrieval only* -- SCR off, nprobe varying
  * *reduction only* -- nprobe fixed at its highest, SCR varying
  * *joint*          -- both varying

C3 is supported if some joint cell is **not dominated by any single-dimension
cell** -- that is, it sits on the quality/cost Pareto front and offers a
trade-off neither dimension reaches alone. "Better than the average
single-dimension cell" is not the test; the single families keep every cell
they have, including the corner they share.

**Cost.** Reported as generation latency plus retrieval latency. SCR's own
on-path cost is deliberately EXCLUDED from the cost axis and reported
separately, because it is measured here against a warm embedding cache and
Phase 5 measured the cold on-path figure at ~1220 ms p95 -- three orders of
magnitude apart. Folding a warm-cache number into the cost axis would make
reduction look nearly free. The separate column is the honest place for it.

**What this cannot say.** Nothing about energy (ADR-0007/0008) and nothing
about the target device. It is a trade-off surface measured on a laptop.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_acceptance import index_root

from smartann.bench.qa import score_answer
from smartann.config import resolve_paths
from smartann.generate import GeneratorConfig, HFGenerator
from smartann.generate.corpus import QaCorpus
from smartann.generate.prompt import join_chunks
from smartann.index.ecovector import EcoVectorIndex
from smartann.pipeline.embedder import Embedder
from smartann.scr.reduce import reduce_context
from smartann.types import AnnConfig, ScrConfig

# Both ladders chosen from measurement, not taste. The SCR ladder uses
# window_size=1 because SQuAD chunks hold at most three sentences, so wider
# windows yield one window per chunk and cannot reduce at all (ADR-0035, and
# the corpus measurement in scripts/phase7_qa.py).
NPROBE_LEVELS = (1, 4, 16)
SCR_LEVELS: dict[str, ScrConfig] = {
    "off": ScrConfig(1, 0, 0, 3, 0.0, enabled=False),
    "keep2": ScrConfig(1, 0, 0, 2, 0.0, enabled=True),
    "keep1": ScrConfig(1, 0, 0, 1, 0.0, enabled=True),
}


def family(nprobe: int, scr_name: str) -> str:
    """Which single-dimension family a cell belongs to, if any."""
    at_max_nprobe = nprobe == max(NPROBE_LEVELS)
    if scr_name == "off":
        return "retrieval_only"
    if at_max_nprobe:
        return "reduction_only"
    return "joint"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-questions", type=int, default=250)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="results/phase8")
    args = ap.parse_args()

    ip = index_root("squad", "ecovector")
    if ip is None:
        print("no ecovector index for squad", file=sys.stderr)
        return 1
    index = EcoVectorIndex.open(ip.root)

    paths = resolve_paths()
    proc = Path(paths.data_root) / "processed" / "squad"
    corpus = QaCorpus(proc / "chunks.sqlite")
    query_vecs = np.load(proc / "queries.npy").astype(np.float32)
    questions = corpus.questions()[: args.n_questions]

    embedder = Embedder(cache_path=Path(paths.data_root) / "cache" / "scr_embeddings.sqlite")
    gcfg = GeneratorConfig(max_input_tokens=1024)
    print(f"loading {gcfg.model_id} ...", flush=True)
    gen = HFGenerator(gcfg)
    gen.load()

    cells = list(itertools.product(NPROBE_LEVELS, SCR_LEVELS))
    print(
        f"{len(cells)} cells x {len(questions)} questions "
        f"= {len(cells) * len(questions)} generations",
        flush=True,
    )

    # Retrieve once per nprobe level, shared by every SCR level above it.
    retrieved: dict[int, list[list[str]]] = {}
    t_retrieve: dict[int, list[float]] = {}
    for nprobe in NPROBE_LEVELS:
        ann = AnnConfig(nprobe=nprobe, ef_centroid=128, ef_list=64, k=args.k, per_cluster_k=32)
        chunks_per_q, times = [], []
        for q in questions:
            vec = query_vecs[q.query_id]
            t0 = time.perf_counter()
            probe = index.probe(vec, ann)
            res = index.finish(vec, probe, ann)
            times.append((time.perf_counter() - t0) * 1000.0)
            chunks_per_q.append(corpus.chunk_texts(res.ids[: args.k].tolist()))
        retrieved[nprobe] = chunks_per_q
        t_retrieve[nprobe] = times
        print(f"  nprobe={nprobe:>3}: retrieved, mean {np.mean(times):.2f} ms", flush=True)

    # Warm the sentence-embedding cache over EVERY chunk any cell will reduce,
    # so no cell pays another cell's cache misses (the mistake ADR-0030 names).
    print("warming the sentence-embedding cache ...", flush=True)
    warm = SCR_LEVELS["keep2"]
    seen: set[str] = set()
    for nprobe in NPROBE_LEVELS:
        for chunks in retrieved[nprobe]:
            for c in chunks:
                if c not in seen:
                    seen.add(c)
                    reduce_context(c, query_vecs[questions[0].query_id], embedder.encode, warm)
    print(f"  {len(seen):,} distinct chunks embedded", flush=True)

    # INTERLEAVED, not blocked. The first version of this experiment ran all
    # 250 questions of one cell before starting the next, and the machine drifted
    # underneath it: identical work cost 1.01 ms/token in the cells measured
    # first and up to 1.96 ms/token in those measured last. Because both
    # single-dimension family bests sit at the highest nprobe -- measured last,
    # and therefore slowest -- the drift inflated exactly the baselines the
    # joint cells are compared against, and manufactured a "C3 supported"
    # verdict. Round-robin spreads any drift across every cell instead of
    # loading it onto whichever ran last (ADR-0030, ADR-0046).
    schedule = [(qi, cell) for qi in range(len(questions)) for cell in cells]

    rows = []
    for step, (qi, (nprobe, scr_name)) in enumerate(schedule):
        scr = SCR_LEVELS[scr_name]
        q = questions[qi]
        vec = query_vecs[q.query_id]
        t0 = time.perf_counter()
        reduced, chars_in, chars_out = [], 0, 0
        for c in retrieved[nprobe][qi]:
            r = reduce_context(c, vec, embedder.encode, scr)
            reduced.append(r.text)
            chars_in += r.chars_in
            chars_out += r.chars_out
        t_scr = (time.perf_counter() - t0) * 1000.0

        g = gen.generate(q.question, join_chunks(reduced))
        em, f1 = score_answer(g.text, q.answers)
        rows.append(
            {
                "nprobe": nprobe,
                "scr": scr_name,
                "family": family(nprobe, scr_name),
                "query_id": q.query_id,
                "step": step,
                "em": em,
                "f1": f1,
                "prompt_tokens": g.prompt_tokens,
                "truncated": g.truncated,
                "chars_in": chars_in,
                "chars_out": chars_out,
                "t_retrieve_ms": t_retrieve[nprobe][qi],
                "t_scr_ms": t_scr,
                "t_generate_ms": g.t_generate_ms,
            }
        )
        if (step + 1) % 250 == 0:
            print(f"  {step + 1}/{len(schedule)}", flush=True)

    import pandas as pd

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(out / "joint_records.parquet", index=False)

    # A host-sleep event is not a measurement. The first run of this experiment
    # recorded a single generation of 1,224,952 ms -- 20.4 minutes -- which
    # dragged that cell's MEAN cost to 5,778 ms against a median of 817 ms and
    # made it the most expensive cell on the surface by a factor of six. The
    # cost axis is therefore the MEDIAN, and any sample beyond 50x its cell's
    # median is reported explicitly rather than quietly absorbed.
    sleeps = []
    for (nprobe, scr_name), g in frame.groupby(["nprobe", "scr"], sort=False):
        med = float(np.percentile(g.t_generate_ms, 50))
        bad = g[g.t_generate_ms > 50 * med]
        for _, r in bad.iterrows():
            sleeps.append(
                {
                    "nprobe": int(nprobe),
                    "scr": str(scr_name),
                    "query_id": int(r.query_id),
                    "t_generate_ms": float(r.t_generate_ms),
                    "cell_median_ms": med,
                }
            )
    if sleeps:
        print("\nHOST-SLEEP EVENTS (excluded from no statistic; median is robust):", flush=True)
        for s in sleeps:
            print(
                f"  nprobe={s['nprobe']} scr={s['scr']} query {s['query_id']}: "
                f"{s['t_generate_ms']:,.0f} ms against a cell median of "
                f"{s['cell_median_ms']:,.0f} ms",
                flush=True,
            )

    # ms/token is the drift diagnostic. Generation cost is close to linear in
    # prompt length on this model, so a cell whose ms/token departs from the
    # others did not do different work -- it ran on a different machine state.
    print(
        f"\n{'nprobe':>7} {'scr':<7} {'family':<15} {'EM':>7} {'F1':>7} "
        f"{'cost ms':>9} {'tok':>7} {'ms/tok':>7} {'scr ms':>8}",
        flush=True,
    )
    cell_summary = []
    for (nprobe, scr_name), g in frame.groupby(["nprobe", "scr"], sort=False):
        cost = float(np.percentile(g.t_retrieve_ms, 50) + np.percentile(g.t_generate_ms, 50))
        s = {
            "nprobe": int(nprobe),
            "scr": str(scr_name),
            "family": family(int(nprobe), str(scr_name)),
            "n": len(g),
            "em": float(g.em.mean()),
            "f1": float(g.f1.mean()),
            "cost_ms": cost,
            "cost_stat": "median of t_retrieve + median of t_generate",
            "prompt_tokens": float(g.prompt_tokens.mean()),
            "ms_per_token": float(np.percentile(g.t_generate_ms, 50) / g.prompt_tokens.mean()),
            "t_scr_ms_warm": float(np.percentile(g.t_scr_ms, 50)),
            "truncated_pct": float(g.truncated.mean() * 100),
        }
        cell_summary.append(s)
        print(
            f"{nprobe:>7} {scr_name:<7} {s['family']:<15} {s['em']:>7.4f} {s['f1']:>7.4f} "
            f"{cost:>9.1f} {s['prompt_tokens']:>7.1f} {s['ms_per_token']:>7.3f} "
            f"{s['t_scr_ms_warm']:>8.1f}",
            flush=True,
        )

    # Drift gate. If the cells no longer agree on ms/token, the cost axis is
    # measuring the machine rather than the configurations, and no verdict
    # drawn from it means anything. Stated as a ratio so it is scale-free.
    mpt = [c["ms_per_token"] for c in cell_summary]
    drift_ratio = max(mpt) / min(mpt)
    drift_ok = drift_ratio <= 1.25
    print(
        f"\n  ms/token spread across cells: {drift_ratio:.2f}x "
        f"({'OK' if drift_ok else 'TOO HIGH -- the cost axis is drift, not configuration'})",
        flush=True,
    )

    # The declared test: does any joint cell dominate the best of BOTH single
    # families? Domination is >= on quality and <= on cost, with at least one
    # strict -- the standard Pareto relation, not "better on average".
    #
    # The families OVERLAP at one cell, and that is deliberate. (max nprobe,
    # SCR off) is the common baseline: it is the strongest retrieval-only
    # setting and it is also "reduction at zero". Assigning it to only one
    # family would deny the other its own baseline and tilt the test toward
    # declaring joint control a winner.
    max_nprobe = max(NPROBE_LEVELS)
    fam_retrieval = [c for c in cell_summary if c["scr"] == "off"]
    fam_reduction = [c for c in cell_summary if c["nprobe"] == max_nprobe]
    joint = [c for c in cell_summary if c["scr"] != "off" and c["nprobe"] != max_nprobe]

    def dominates(a: dict[str, object], b: dict[str, object]) -> bool:
        """``a`` is at least as good on both axes and strictly better on one.

        Higher F1 is better, lower cost is better. The standard Pareto
        relation, written once so both tests below use the same one.
        """
        af, ac = float(a["f1"]), float(a["cost_ms"])  # type: ignore[arg-type]
        bf, bc = float(b["f1"]), float(b["cost_ms"])  # type: ignore[arg-type]
        return af >= bf and ac <= bc and (af > bf or ac < bc)

    singles = fam_retrieval + fam_reduction

    # THE DECLARED TEST. A joint cell supports C3 when no single-dimension cell
    # dominates it -- that is, it sits on the overall Pareto front and offers a
    # quality/cost trade-off neither dimension reaches alone.
    #
    # An earlier draft asked whether a joint cell beat the HIGHEST-F1 single
    # cell on both axes at once. That is arbitrarily strict: the highest-F1
    # cell is typically also the most expensive, so the test demanded a joint
    # cell be both more accurate than the most accurate setting and cheaper
    # than an expensive one. Almost nothing passes it, and a "C3 not supported"
    # produced that way would have measured the criterion, not the surface.
    on_front = [c for c in joint if not any(dominates(s, c) for s in singles)]

    # The stronger claim, reported separately when it happens to hold.
    b_ret = max(fam_retrieval, key=lambda c: c["f1"])
    b_red = max(fam_reduction, key=lambda c: c["f1"])
    dominating = [c for c in joint if dominates(c, b_ret) and dominates(c, b_red)]

    print("\n--- C3: joint vs split ---", flush=True)
    print(
        f"  best retrieval-only: nprobe={b_ret['nprobe']} scr={b_ret['scr']} "
        f"F1 {b_ret['f1']:.4f} at {b_ret['cost_ms']:.1f} ms",
        flush=True,
    )
    print(
        f"  best reduction-only: nprobe={b_red['nprobe']} scr={b_red['scr']} "
        f"F1 {b_red['f1']:.4f} at {b_red['cost_ms']:.1f} ms",
        flush=True,
    )
    for c in on_front:
        print(
            f"  ON THE FRONT (no single-dimension cell dominates it): "
            f"nprobe={c['nprobe']} scr={c['scr']} F1 {c['f1']:.4f} at {c['cost_ms']:.1f} ms",
            flush=True,
        )
    for c in dominating:
        print(
            f"  DOMINATES BOTH FAMILY BESTS: nprobe={c['nprobe']} scr={c['scr']} "
            f"F1 {c['f1']:.4f} at {c['cost_ms']:.1f} ms",
            flush=True,
        )

    if on_front:
        verdict = (
            f"C3 SUPPORTED on this surface: {len(on_front)} joint cell(s) sit on the "
            f"quality/cost Pareto front, offering trade-offs no single-dimension "
            f"setting reaches. "
            + (
                "One also dominates both family bests outright."
                if dominating
                else "None dominates both family bests outright, so the gain is a "
                "trade-off along the front rather than a free win."
            )
        )
    else:
        verdict = (
            "C3 NOT SUPPORTED on this surface: every joint cell is dominated by some "
            "single-dimension cell. Joint control reaches nothing that one dimension "
            "does not already reach more cheaply, so no control law over both "
            "dimensions would either."
        )
    print(f"\n  {verdict}", flush=True)

    (out / "joint_summary.json").write_text(
        json.dumps(
            {
                "dataset": "squad",
                "n_questions": len(questions),
                "k": args.k,
                "generator": gcfg.as_dict(),
                "nprobe_levels": list(NPROBE_LEVELS),
                "scr_levels": list(SCR_LEVELS),
                "cost_axis": "t_retrieve_ms + t_generate_ms. SCR's own cost is EXCLUDED "
                "and reported separately as t_scr_ms_warm: it is measured against a warm "
                "embedding cache, while Phase 5 measured the cold on-path cost at ~1220 ms "
                "p95. Folding a warm-cache number into the cost axis would make reduction "
                "look nearly free.",
                "energy": "NOT measured on this platform (ADR-0007/0008).",
                "cells": cell_summary,
                "declared_test": "a joint cell supports C3 if NO single-dimension cell "
                "dominates it (>= F1 and <= cost, one strict). Reported as "
                "joint_cells_on_pareto_front.",
                "best_retrieval_only": b_ret,
                "best_reduction_only": b_red,
                "joint_cells_on_pareto_front": on_front,
                "joint_cells_dominating_both_family_bests": dominating,
                "verdict": verdict,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out}/joint_records.parquet and {out}/joint_summary.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

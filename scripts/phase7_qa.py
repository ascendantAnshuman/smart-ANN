#!/usr/bin/env python
"""Phase 7: end-to-end RAG on SQuAD, scored with EM and F1.

**What this closes.** Until now every SCR setting could be priced only in
characters, tokens and milliseconds. On those axes a reduction that deletes the
answer is indistinguishable from a free one -- both look like a win. C2 claims
*adaptive* reduction is worth having, and that claim is empty without a quality
number on the other side of the trade. This is that number.

**The arms.** One retrieval configuration, held fixed, so every difference
between arms is attributable to context reduction alone. `off` is the
byte-identical passthrough (ADR-0035), and the reduction arms tighten from
there. The same questions, in the same order, against the same index.

**Retrieval runs once.** It is identical in every arm by construction, so
re-running it per arm would add noise to a comparison meant to isolate
reduction, and would triple the runtime for nothing.

**Determinism.** Greedy decoding and a fixed question order, so a re-run
reproduces answer-for-answer. The only stochastic element would be sampling,
which is disabled.

**What it cannot do.** Nothing here measures energy; no arm claims to. Latency
is measured, but this machine is a laptop and not the target device, so the
generation cost below is reported for the ratio between arms rather than as a
device number.
"""

from __future__ import annotations

import argparse
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

# Aggressiveness ladder, chosen from the corpus rather than guessed.
#
# SQuAD chunks are short: 1.4% hold one sentence, 22.6% two, 76.0% three
# (3,000 sampled). The number of windows a chunk yields therefore caps how far
# any setting can reduce it, and most settings cannot reduce it at all:
#
#   w=3,o=1  ->  100.0% of chunks yield exactly ONE window   (ADR-0035)
#   w=2,o=1  ->   24.0% yield one; the rest yield two, so only top_windows=1 cuts
#   w=1,o=0  ->   98.6% yield two or three -- the only setting with a real ladder
#
# The first draft of this experiment used w=2 with top_windows 4/3/2/1 and
# produced three arms byte-identical to `off`, because keeping 3 of at most 2
# windows keeps everything. The reduction was not broken; the ladder was.
ARMS: dict[str, ScrConfig] = {
    "off": ScrConfig(1, 0, 0, 3, 0.0, enabled=False),
    "keep2": ScrConfig(1, 0, 0, 2, 0.0, enabled=True),
    "keep1_ext": ScrConfig(1, 0, 1, 1, 0.0, enabled=True),
    "keep1": ScrConfig(1, 0, 0, 1, 0.0, enabled=True),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-questions", type=int, default=300)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--nprobe", type=int, default=8)
    ap.add_argument("--out", default="results/phase7")
    args = ap.parse_args()

    ip = index_root("squad", "ecovector")
    if ip is None:
        print("no ecovector index for squad; run `smartann index build`", file=sys.stderr)
        return 1
    index = EcoVectorIndex.open(ip.root)

    paths = resolve_paths()
    proc = Path(paths.data_root) / "processed" / "squad"
    corpus = QaCorpus(proc / "chunks.sqlite")
    query_vecs = np.load(proc / "queries.npy").astype(np.float32)

    questions = corpus.questions()[: args.n_questions]
    print(f"{len(questions)} questions, k={args.k}, nprobe={args.nprobe}", flush=True)

    embedder = Embedder(cache_path=Path(paths.data_root) / "cache" / "scr_embeddings.sqlite")

    # 1024, not the model's nominal 512. At k=5 the unreduced context reaches
    # ~590 tokens, and at 512 two thirds of the `off` arm was silently
    # truncated -- so the baseline would have been "the first 512 tokens",
    # making reduction look free because the comparison never happened. T5 uses
    # relative position bias and extrapolates past its training length; the
    # `truncated` field reports any residual so this cannot hide again.
    gcfg = GeneratorConfig(max_input_tokens=1024)
    print(f"loading {gcfg.model_id} (max_input_tokens={gcfg.max_input_tokens}) ...", flush=True)
    t0 = time.perf_counter()
    gen = HFGenerator(gcfg)
    gen.load()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    ann = AnnConfig(nprobe=args.nprobe, ef_centroid=128, ef_list=64, k=args.k, per_cluster_k=32)

    print(f"retrieving top-{args.k} once, shared by every arm ...", flush=True)
    retrieved: list[list[str]] = []
    t_retrieve_ms: list[float] = []
    for q in questions:
        vec = query_vecs[q.query_id]
        t0 = time.perf_counter()
        probe = index.probe(vec, ann)
        res = index.finish(vec, probe, ann)
        t_retrieve_ms.append((time.perf_counter() - t0) * 1000.0)
        retrieved.append(corpus.chunk_texts(res.ids[: args.k].tolist()))

    # Warm the sentence-embedding cache before any SCR timing. The first
    # reducing arm otherwise pays every cache miss and every model load, and
    # reports it as its own cost: in the first run that arm measured 1700 ms
    # against 8 ms for the arms behind it, for identical work. Same lesson as
    # ADR-0030, in a different stage.
    warm = ScrConfig(1, 0, 0, 2, 0.0, enabled=True)
    for chunks in retrieved[: min(30, len(retrieved))]:
        for c in chunks:
            reduce_context(c, query_vecs[questions[0].query_id], embedder.encode, warm)

    rows = []
    for arm, scr in ARMS.items():
        print(f"\n=== arm: {arm} ===", flush=True)
        for qi, q in enumerate(questions):
            vec = query_vecs[q.query_id]

            t0 = time.perf_counter()
            reduced: list[str] = []
            chars_in = chars_out = 0
            for c in retrieved[qi]:
                r = reduce_context(c, vec, embedder.encode, scr)
                reduced.append(r.text)
                chars_in += r.chars_in
                chars_out += r.chars_out
            t_scr = (time.perf_counter() - t0) * 1000.0

            g = gen.generate(q.question, join_chunks(reduced))
            em, f1 = score_answer(g.text, q.answers)

            rows.append(
                {
                    "arm": arm,
                    "query_id": q.query_id,
                    "em": em,
                    "f1": f1,
                    "answer": g.text,
                    "prompt_tokens": g.prompt_tokens,
                    "truncated": g.truncated,
                    "chars_in": chars_in,
                    "chars_out": chars_out,
                    "t_retrieve_ms": t_retrieve_ms[qi],
                    "t_scr_ms": t_scr,
                    "t_generate_ms": g.t_generate_ms,
                }
            )
            if (qi + 1) % 50 == 0:
                print(f"  {qi + 1}/{len(questions)}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    frame = pd.DataFrame(rows)
    frame.to_parquet(out / "qa_records.parquet", index=False)

    header = (
        f"\n{'arm':<11} {'EM':>7} {'F1':>7} {'tok':>7} "
        f"{'chars':>8} {'red%':>7} {'scr ms':>8} {'gen ms':>8} {'trunc%':>7}"
    )
    print(header, flush=True)
    summary = []
    for arm in ARMS:
        g = frame[frame.arm == arm]
        s = {
            "arm": arm,
            "n": len(g),
            "em": float(g.em.mean()),
            "f1": float(g.f1.mean()),
            "prompt_tokens_mean": float(g.prompt_tokens.mean()),
            "chars_out_mean": float(g.chars_out.mean()),
            "reduction_pct": float((1.0 - g.chars_out.sum() / g.chars_in.sum()) * 100),
            "t_scr_ms_mean": float(g.t_scr_ms.mean()),
            "t_generate_ms_mean": float(g.t_generate_ms.mean()),
            "truncated_pct": float(g.truncated.mean() * 100),
            "scr": {
                "window": ARMS[arm].sliding_window_size,
                "overlap": ARMS[arm].overlap_size,
                "extension": ARMS[arm].context_extension_size,
                "top_windows": ARMS[arm].top_windows,
                "enabled": ARMS[arm].enabled,
            },
        }
        summary.append(s)
        print(
            f"{arm:<11} {s['em']:>7.4f} {s['f1']:>7.4f} {s['prompt_tokens_mean']:>7.1f} "
            f"{s['chars_out_mean']:>8.0f} {s['reduction_pct']:>6.1f}% "
            f"{s['t_scr_ms_mean']:>8.1f} {s['t_generate_ms_mean']:>8.1f} "
            f"{s['truncated_pct']:>6.1f}%",
            flush=True,
        )

    # Paired bootstrap against `off`. The arms answer the SAME questions in the
    # same order, so the per-question difference is the statistic and an
    # unpaired interval would throw away that pairing and widen every CI for
    # nothing. A CI straddling zero means this experiment cannot distinguish
    # the arm from the baseline -- which is a result, not a failure to report.
    print("\npaired bootstrap vs `off`, 10,000 resamples, 95% CI:", flush=True)
    rng = np.random.default_rng(0)
    base = frame[frame.arm == "off"].sort_values("query_id")
    paired = []
    for arm in ARMS:
        if arm == "off":
            continue
        g = frame[frame.arm == arm].sort_values("query_id")
        assert (g.query_id.to_numpy() == base.query_id.to_numpy()).all(), "arms are not paired"
        row: dict[str, object] = {"arm": arm}
        for metric in ("em", "f1"):
            d = g[metric].to_numpy() - base[metric].to_numpy()
            idx = rng.integers(0, len(d), size=(10_000, len(d)))
            boot = d[idx].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            row[f"d_{metric}"] = float(d.mean())
            row[f"d_{metric}_ci"] = [float(lo), float(hi)]
            row[f"d_{metric}_significant"] = bool(lo > 0 or hi < 0)
        paired.append(row)
        print(
            f"  {arm:<11} dEM {row['d_em']:+.4f} "
            f"[{row['d_em_ci'][0]:+.4f}, {row['d_em_ci'][1]:+.4f}]"  # type: ignore[index]
            f"{'  *' if row['d_em_significant'] else '   '}"
            f"   dF1 {row['d_f1']:+.4f} "
            f"[{row['d_f1_ci'][0]:+.4f}, {row['d_f1_ci'][1]:+.4f}]"  # type: ignore[index]
            f"{'  *' if row['d_f1_significant'] else ''}",
            flush=True,
        )

    (out / "qa_summary.json").write_text(
        json.dumps(
            {
                "dataset": "squad",
                "n_questions": len(questions),
                "k": args.k,
                "nprobe": args.nprobe,
                "generator": GeneratorConfig().as_dict(),
                "retrieval": "run once and shared across arms; identical in every arm "
                "by construction",
                "energy": "NOT measured: this platform cannot (ADR-0007/0008). Latency is "
                "this laptop's, reported for the ratio between arms.",
                "t_scr_ms_caveat": "WARM CACHE, and NOT comparable across arms. The "
                "sentence-embedding cache fills during the first reducing arm, so later "
                "arms read what it paid for. It also understates the real on-path cost "
                "by orders of magnitude: Phase 5 measured SCR at ~1220 ms p95 on the "
                "query path, because it embeds every window of every retrieved chunk. "
                "Use that number, not this one, when asking whether SCR pays for itself. "
                "t_generate_ms is unaffected -- no cache is involved in generation.",
                "arms": summary,
                "paired_bootstrap_vs_off": paired,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out}/qa_records.parquet and {out}/qa_summary.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

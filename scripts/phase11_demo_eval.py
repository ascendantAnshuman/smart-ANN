#!/usr/bin/env python
"""11.4 -- does retrieval on the demonstration corpus actually discriminate?

Two questions, measured rather than asserted:

1. **Does the right document come back whatever the wording?** Ten paraphrases
   of one question, scored against documents hand-labelled as relevant before
   any retrieval was run (`data/demo_phone_care/eval_queries.json`).

2. **What do scores look like when the answer is not there?** Ten off-topic
   queries, split deliberately into two kinds. Five are answerable from the
   distractor documents, where the correct behaviour is to answer, not to
   refuse. Five have no relevant document at all, where the correct behaviour is
   to decline. A single "off-topic" bucket would mix those two and produce a
   refusal threshold that is wrong for both.

   This is what Phase 12.1's grounded-answer guard will be calibrated from, so
   the separation matters: the threshold has to sit above what an unanswerable
   question scores and below what an answerable one scores, and that gap can
   only be seen if the two are measured apart.

**Also produces the measured profile table for this corpus.** Without one the
controller has nothing measured to choose between, so `ask` on this dataset
falls back to a fixed nprobe and the demonstration has no controller in it. The
table's recall is ANN recall@k against exhaustive search over the same 129
vectors -- the standard definition used everywhere else in this project, and
exactly 1.0 at the top level by construction, because nprobe=11 opens every
cluster this index has.

**The latency in that table is this laptop's, and it is labelled as this
laptop's.** The handset numbers in `results/device_profile_table.json` were
measured on the phone over the SQuAD index and do not transfer to a 129-vector
corpus. Nothing here is presented as a device measurement.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smartann.config import resolve_paths
from smartann.controller.profile import ProfileEntry, ProfileTable
from smartann.generate.corpus import QaCorpus
from smartann.index.ecovector import EcoVectorIndex
from smartann.index.storage import resolve_index_root
from smartann.pipeline.embedder import Embedder
from smartann.types import AnnConfig

DATASET = "demo_phone_care"
# The retrieval evaluation is scored at 5, which is what the demonstration shows
# a reader and what Phase 11.4 asked for.
K = 5
# The profile table is built at 10, which is what the handset's own table in
# results/device_profile_table.json uses and what the Phase 12.4 stress test
# scores. A table whose recall column is measured at one k while the policy's
# consumer is scored at another makes the controller optimise a quantity nobody
# reports: on this corpus recall@5 saturates at nprobe=3 and recall@10 does not,
# so the mismatch silently pinned every decision to the same level.
K_TABLE = 10
EF_CENTROID = 128
EF_LIST = 64
PER_CLUSTER_K = 32
LATENCY_REPEATS = 5  # per query per level; 91 queries x 5 is 455 samples per level


@dataclass(frozen=True)
class Corpus:
    """The chunk-to-document mapping the labels are expressed in."""

    doc_id_of_chunk: dict[int, int]
    slug_of_doc: dict[int, str]
    title_of_doc: dict[int, str]

    def slug(self, chunk_id: int) -> str:
        return self.slug_of_doc[self.doc_id_of_chunk[chunk_id]]

    def title(self, chunk_id: int) -> str:
        return self.title_of_doc[self.doc_id_of_chunk[chunk_id]]


def load_corpus(db_path: Path) -> Corpus:
    """Chunk -> document -> the slug used in the hand labels.

    `file_path` holds `demo_phone_care/<slug>`, written by the build script, so
    the labels and the index agree on what a document is called without a second
    mapping to keep in step.
    """
    corpus = QaCorpus(db_path)
    conn = corpus._conn
    doc_id_of_chunk = {
        int(r["chunk_id"]): int(r["doc_id"])
        for r in conn.execute("SELECT chunk_id, doc_id FROM metadata")
    }
    slug_of_doc: dict[int, str] = {}
    title_of_doc: dict[int, str] = {}
    for r in conn.execute("SELECT doc_id, file_path, title FROM documents"):
        slug_of_doc[int(r["doc_id"])] = str(r["file_path"]).split("/", 1)[-1]
        title_of_doc[int(r["doc_id"])] = str(r["title"])
    corpus.close()
    return Corpus(doc_id_of_chunk, slug_of_doc, title_of_doc)


def exact_top_k(base: np.ndarray, qvec: np.ndarray, k: int) -> list[int]:
    """Brute-force nearest neighbours, for the recall the profile table reports.

    129 vectors makes this exact and instant, so the ANN's recall is measured
    against the real answer rather than against a larger approximation.
    """
    sims = base @ qvec
    return [int(i) for i in np.argsort(-sims)[:k]]


def search(
    index: EcoVectorIndex, qvec: np.ndarray, nprobe: int, k: int
) -> tuple[list[int], list[float], Any]:
    cfg = AnnConfig(
        nprobe=nprobe, ef_centroid=EF_CENTROID, ef_list=EF_LIST, k=k, per_cluster_k=PER_CLUSTER_K
    )
    res = index.search(qvec, cfg)
    return res.ids[:k].tolist(), res.dists[:k].tolist(), res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=Path, default=Path("data/demo_phone_care/eval_queries.json"))
    ap.add_argument("--out-csv", type=Path, default=Path("results/demo_eval.csv"))
    ap.add_argument("--out-json", type=Path, default=Path("results/demo_eval.json"))
    ap.add_argument("--out-table", type=Path, default=Path("results/demo_profile_table.json"))
    ap.add_argument("--nprobe", type=int, default=11, help="Effort level for the eval itself.")
    ap.add_argument(
        "--allow-busy",
        action="store_true",
        help="Measure timings anyway on a loaded machine. The profile table then "
        "records that it was taken on a busy box, and its latencies are not quotable.",
    )
    args = ap.parse_args()

    # The same gate scripts/phase2_acceptance.py applies before any timing run.
    # This script writes a profile table that a controller then chooses effort
    # from, so a corrupted p95 here becomes a wrong decision later.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from phase2_acceptance import assert_machine_quiet

    busy_note = ""
    try:
        assert_machine_quiet()
    except RuntimeError as exc:
        if not args.allow_busy:
            print(exc)
            return 2
        busy_note = " TAKEN ON A BUSY MACHINE (--allow-busy): the latencies here are not quotable."

    spec = json.loads(args.queries.read_text(encoding="utf-8"))
    gold_primary = set(spec["gold_primary"])
    # The wider primary set, when the corpus revision defines one. Both are
    # scored and both are reported: the frozen set keeps this run comparable
    # with the 36-document one, and the extended set avoids counting a
    # correct retrieval as a miss just because the document was added later.
    gold_primary_ext = set(spec.get("gold_primary_extended", spec["gold_primary"]))
    gold_relevant = set(spec["gold_relevant"])

    ip = resolve_index_root(DATASET, "ecovector")
    if ip is None:
        print("no demo index. Run scripts/phase11_build_demo_index.py first.")
        return 2
    paths = resolve_paths()
    proc = Path(paths.data_root) / "processed" / DATASET
    corpus = load_corpus(proc / "chunks.sqlite")
    base = np.load(proc / "base.npy").astype(np.float32)
    index = EcoVectorIndex.open(ip.root)
    embedder = Embedder(cache_path=Path(paths.data_root) / "cache" / "demo.sqlite")
    _ = embedder.model  # load outside anything timed

    n_clusters = int(np.count_nonzero(index.sizes > 0))
    print(f"{DATASET}: {base.shape[0]} chunks in {n_clusters} clusters, k={K}")

    rows: list[dict[str, Any]] = []

    def evaluate(query: str, kind: str, gold: set[str], note: str) -> dict[str, Any]:
        qvec = embedder.encode([query])[0].astype(np.float32)
        ids, dists, res = search(index, qvec, args.nprobe, K)
        sims = [1.0 - float(d) for d in dists]
        slugs = [corpus.slug(i) for i in ids]
        docs_hit = set(slugs)

        in_primary = docs_hit & gold_primary
        n_relevant = sum(1 for s in slugs if s in gold_relevant)
        n_distractor = sum(1 for s in slugs if s.startswith("distract-"))

        if kind == "paraphrase":
            recall = len(in_primary) / len(gold_primary)
            recall_ext = len(docs_hit & gold_primary_ext) / len(gold_primary_ext)
        elif gold:
            recall = len(docs_hit & gold) / len(gold)
            recall_ext = recall
        else:
            # Nothing is relevant, so there is nothing to recall.
            recall = recall_ext = float("nan")

        return {
            "query": query,
            "kind": kind,
            "n_gold": len(gold) if gold else 0,
            "recall_at_5": recall,
            "recall_at_5_extended": recall_ext,
            "relevant_at_5": n_relevant / K,
            "distractors_at_5": n_distractor,
            "best_similarity": max(sims) if sims else float("nan"),
            "mean_similarity": float(np.mean(sims)) if sims else float("nan"),
            "top_doc": slugs[0] if slugs else "",
            "top_title": corpus.title(ids[0]) if ids else "",
            "top_chunk_id": ids[0] if ids else -1,
            "retrieved_docs": "|".join(slugs),
            "retrieved_similarities": "|".join(f"{s:.4f}" for s in sims),
            "n_cluster_loads": res.n_cluster_loads,
            "n_dist_exact": res.n_dist_exact,
            "note": note,
        }

    for q in spec["paraphrases"]:
        rows.append(evaluate(q, "paraphrase", gold_primary, "paraphrase of the heating question"))
    for item in spec["off_topic"]:
        rows.append(evaluate(item["query"], item["kind"], set(item["gold"]), item.get("note", "")))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def group(kind: str) -> list[dict[str, Any]]:
        return [r for r in rows if r["kind"] == kind]

    para, in_corpus, out_corpus = (
        group("paraphrase"),
        group("off_topic_in_corpus"),
        group("out_of_corpus"),
    )

    def best(rs: list[dict[str, Any]]) -> list[float]:
        return [float(r["best_similarity"]) for r in rs]

    print(f"\nparaphrases (n={len(para)})")
    print(
        f"  recall@5 vs the frozen primary set       "
        f"{np.mean([r['recall_at_5'] for r in para]):.3f}"
    )
    print(
        f"  recall@5 vs the extended primary set     "
        f"{np.mean([r['recall_at_5_extended'] for r in para]):.3f}"
    )
    print(
        f"  relevant@5 (share of the context useful) {np.mean([r['relevant_at_5'] for r in para]):.3f}"
    )
    print(f"  distractors in top 5                     {sum(r['distractors_at_5'] for r in para)}")
    print(
        f"  best similarity  min {min(best(para)):.3f}  mean {np.mean(best(para)):.3f}  max {max(best(para)):.3f}"
    )

    print(f"\noff-topic but answerable here (n={len(in_corpus)})")
    print(
        f"  recall@5 against each query's own gold   {np.mean([r['recall_at_5'] for r in in_corpus]):.3f}"
    )
    print(
        f"  best similarity  min {min(best(in_corpus)):.3f}  mean {np.mean(best(in_corpus)):.3f}  max {max(best(in_corpus)):.3f}"
    )

    print(f"\nnot in the corpus at all (n={len(out_corpus)})")
    print(
        f"  best similarity  min {min(best(out_corpus)):.3f}  mean {np.mean(best(out_corpus)):.3f}  max {max(best(out_corpus)):.3f}"
    )
    for r in sorted(out_corpus, key=lambda r: -float(r["best_similarity"])):
        print(f"    {float(r['best_similarity']):.3f}  {r['query']}  -> {r['top_doc']}")

    answerable_floor = min(best(para) + best(in_corpus))
    unanswerable_ceiling = max(best(out_corpus))
    gap = answerable_floor - unanswerable_ceiling
    print(
        f"\nseparation: lowest answerable {answerable_floor:.3f} vs highest unanswerable "
        f"{unanswerable_ceiling:.3f}  ->  gap {gap:+.3f}"
    )
    if gap <= 0:
        print("  NO GAP. A single similarity threshold cannot separate these two sets.")

    # ---------------------------------------------------------------- profile
    levels = list(range(1, n_clusters + 1))
    # The table is built over the WHOLE query set, not the ten paraphrases.
    #
    # The ten paraphrases all ask one question, so their true neighbours sit in
    # one or two clusters and their recall saturates at nprobe=3. A table built
    # from them tells a controller that effort above 3 buys nothing, which is
    # true of that question and false of the corpus: across the broader set,
    # recall@10 at nprobe=3 is nowhere near 1.0. A controller reading the narrow
    # table picks the same level whatever the device is doing, because every
    # level above 3 looks equally good to it.
    #
    # Document titles are included because they are natural questions about this
    # corpus and they spread the neighbourhoods across the whole index.
    docs = json.loads(Path("data/demo_phone_care/documents.json").read_text(encoding="utf-8"))
    table_queries = (
        list(spec["paraphrases"])
        + [item["query"] for item in spec["off_topic"]]
        + [d["title"] for d in docs]
    )
    qvecs = [embedder.encode([q])[0].astype(np.float32) for q in table_queries]
    exact = [exact_top_k(base, v, K_TABLE) for v in qvecs]
    print(f"  profile table over {len(table_queries)} queries, recall at k={K_TABLE}")

    # Warm every level before timing any of it, exactly as scripts/device_trace.py
    # does before it starts recording. Without this the first measurement at each
    # level pays for first-touch page faults on cluster files that no later query
    # pays, and that one sample lands in a 200-sample p95 -- which is how nprobe=4
    # came out slower than nprobe=11 on the first run of this script.
    for npb in levels:
        for v in qvecs:
            search(index, v, npb, K_TABLE)

    # Levels are INTERLEAVED, not measured one after another.
    #
    # Measuring all 20 repeats of nprobe=3 and then all 20 of nprobe=4 gives any
    # transient load -- a sync client waking up, an indexer starting -- entirely
    # to whichever level was running at the time. That is how a first version of
    # this table came out with nprobe=3 at a 12.8 ms p95 and nprobe=4 at 4.6 ms:
    # more work measured as less time. A controller then picked nprobe=4 over
    # nprobe=3 for a reason that was pure noise.
    #
    # Round-robin spreads any such disturbance across every level instead, which
    # is the same within-config interleaving ADR-0030 settled on for the device
    # runs and for the same reason.
    lat: dict[int, list[float]] = {npb: [] for npb in levels}
    recalls: dict[int, list[float]] = {npb: [] for npb in levels}
    for _ in range(LATENCY_REPEATS):
        for npb in levels:
            for v, gold_ids in zip(qvecs, exact, strict=True):
                t0 = time.perf_counter()
                ids, _d, _r = search(index, v, npb, K_TABLE)
                lat[npb].append((time.perf_counter() - t0) * 1000.0)
                recalls[npb].append(len(set(ids) & set(gold_ids)) / K_TABLE)

    entries: list[ProfileEntry] = []
    for npb in levels:
        entries.append(
            ProfileEntry(
                nprobe=npb,
                n_samples=len(lat[npb]),
                p50_ms=float(np.percentile(lat[npb], 50)),
                p95_ms=float(np.percentile(lat[npb], 95)),
                recall=float(np.mean(recalls[npb])),
                approx_tokens=0.0,
            )
        )
        print(
            f"  nprobe={npb:>2}  p50 {entries[-1].p50_ms:6.3f} ms  p95 {entries[-1].p95_ms:6.3f} ms  "
            f"recall@{K_TABLE} vs exact {entries[-1].recall:.4f}"
        )

    # A latency table in which more work takes less time is corrupted, and a
    # controller choosing from it is choosing from noise. Reported rather than
    # smoothed: the fix is a quiet machine, not a monotone filter over the data.
    def _inversions(key: str) -> list[str]:
        values = [getattr(e, key) for e in entries]
        return [
            f"nprobe={entries[i].nprobe} {values[i]:.2f} ms > "
            f"nprobe={entries[i + 1].nprobe} {values[i + 1]:.2f} ms"
            for i in range(len(entries) - 1)
            if values[i] > values[i + 1]
        ]

    p50_inversions, p95_inversions = _inversions("p50_ms"), _inversions("p95_ms")
    if p50_inversions or p95_inversions:
        print()
        print("  WARNING: the latency ladder is not monotonic in nprobe.")
        for label, inv in (("p50", p50_inversions), ("p95", p95_inversions)):
            for line in inv:
                print(f"    {label}: {line}")
        print("    More clusters cannot be faster. Re-run on a quiet machine before")
        print("    quoting these latencies or letting a controller choose from them.")

    table = ProfileTable(
        entries=tuple(entries),
        dataset=DATASET,
        measured_under=(
            # The corpus size is read, not written in. It was hardcoded as 129
            # here and became wrong the moment the corpus grew -- a provenance
            # string describing a run that never happened.
            f"THIS LAPTOP, not the handset: retrieval only, warm cache, {len(qvecs)} "
            f"queries (10 paraphrases, 10 off-topic, every document title) x "
            f"{LATENCY_REPEATS} interleaved repeats per level over the "
            f"{base.shape[0]}-chunk demo corpus in {n_clusters} clusters, recall at "
            f"k={K_TABLE} to match the handset table and the 12.4 stress test. Device "
            f"latencies live in results/device_profile_table.json and are for SQuAD." + busy_note
        ),
        min_samples=LATENCY_REPEATS * len(qvecs),
    )
    table.write(args.out_table)

    summary = {
        "dataset": DATASET,
        "n_chunks": int(base.shape[0]),
        "n_clusters": n_clusters,
        "k": K,
        "nprobe_for_eval": args.nprobe,
        "relevance_model": spec["relevance_model"],
        "gold_primary_frozen": sorted(gold_primary),
        "gold_primary_extended": sorted(gold_primary_ext),
        "n_documents": len(set(corpus.slug_of_doc.values())),
        "paraphrases": {
            "n": len(para),
            "recall_at_5_frozen_primary": float(np.mean([r["recall_at_5"] for r in para])),
            "recall_at_5_extended_primary": float(
                np.mean([r["recall_at_5_extended"] for r in para])
            ),
            "relevant_at_5": float(np.mean([r["relevant_at_5"] for r in para])),
            "distractors_in_top5": int(sum(r["distractors_at_5"] for r in para)),
            "best_similarity": {
                "min": min(best(para)),
                "mean": float(np.mean(best(para))),
                "max": max(best(para)),
            },
        },
        "off_topic_in_corpus": {
            "n": len(in_corpus),
            "recall_at_5": float(np.mean([r["recall_at_5"] for r in in_corpus])),
            "best_similarity": {
                "min": min(best(in_corpus)),
                "mean": float(np.mean(best(in_corpus))),
                "max": max(best(in_corpus)),
            },
        },
        "out_of_corpus": {
            "n": len(out_corpus),
            "best_similarity": {
                "min": min(best(out_corpus)),
                "mean": float(np.mean(best(out_corpus))),
                "max": max(best(out_corpus)),
            },
            "per_query": {r["query"]: float(r["best_similarity"]) for r in out_corpus},
        },
        "profile_table_monotonic": {
            "p50_inversions": p50_inversions,
            "p95_inversions": p95_inversions,
            "note": (
                "a latency ladder where more clusters take less time is corrupted by "
                "machine noise, not a property of the index"
            ),
        },
        "separation": {
            "lowest_answerable": answerable_floor,
            "highest_unanswerable": unanswerable_ceiling,
            "gap": gap,
            "for_phase_12_1": (
                "a grounded-answer threshold must sit strictly between these two numbers. "
                "Any value in that interval refuses every unanswerable query in this set "
                "and answers every answerable one."
            ),
        },
        "provenance": {
            "relevance": "hand-labelled by reading the documents, before any retrieval was run",
            "similarity": "measured: cosine between gte-small embeddings, computed by the index",
            "latency_in_profile_table": "measured on this laptop, NOT on the handset",
            "recall_in_profile_table": (
                f"ANN recall@{K_TABLE} against exhaustive search over the same vectors"
            ),
        },
    }
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out_csv}\nwrote {args.out_json}\nwrote {args.out_table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

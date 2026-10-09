#!/usr/bin/env python
"""12.1 and 12.2 -- fix both thresholds from measurements rather than taste.

Two constants decide how the demonstration behaves, and both would otherwise be
round numbers somebody liked:

* **the grounding threshold** -- below this best-passage score, refuse to answer.
  Taken from the Phase 11.4 separation: answerable queries never scored below a
  floor, unanswerable ones never above a ceiling, and the threshold is the
  midpoint of that gap. Computed by `GroundingGuard.calibrate`, which refuses
  outright if the two groups overlap.

* **the cache similarity threshold** -- above this query-to-query similarity,
  reuse the earlier answer. Measured here, by comparing two distributions over
  the same query set: how similar paraphrases of one question are to each other,
  and how similar unrelated questions are. A threshold between the two reuses an
  answer only for questions that genuinely mean the same thing.

The second one is the dangerous one, which is why it gets the wider margin. A
cache hit does not degrade an answer, it replaces it: two different questions
treated as the same would both receive one answer, confidently and with a
citation, and nothing in the output would look wrong.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smartann.config import resolve_paths
from smartann.pipeline.embedder import Embedder
from smartann.pipeline.guard import GroundingGuard


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=Path, default=Path("data/demo_phone_care/eval_queries.json"))
    ap.add_argument("--eval-json", type=Path, default=Path("results/demo_eval.json"))
    ap.add_argument("--out-guard", type=Path, default=Path("results/grounding_threshold.json"))
    ap.add_argument("--out-cache", type=Path, default=Path("results/cache_threshold.json"))
    args = ap.parse_args()

    for p in (args.queries, args.eval_json):
        if not p.exists():
            print(f"missing {p}. Run scripts/phase11_demo_eval.py first.")
            return 2

    # ------------------------------------------------------------- guard
    try:
        guard = GroundingGuard.calibrate(args.eval_json)
    except ValueError as exc:
        print(f"cannot calibrate the grounding guard: {exc}")
        return 1
    guard.write(args.out_guard)
    print("grounding guard")
    print(f"  answerable queries scored no lower than   {guard.answerable_floor:.4f}")
    print(f"  unanswerable queries no higher than       {guard.unanswerable_ceiling:.4f}")
    print(f"  gap                                       {guard.gap:.4f}")
    print(f"  threshold (midpoint)                      {guard.threshold:.4f}")

    # ------------------------------------------------------------- cache
    spec = json.loads(args.queries.read_text(encoding="utf-8"))
    paraphrases: list[str] = list(spec["paraphrases"])
    others: list[str] = [item["query"] for item in spec["off_topic"]]

    paths = resolve_paths()
    embedder = Embedder(cache_path=Path(paths.data_root) / "cache" / "demo.sqlite")
    vecs_same = np.asarray(embedder.encode(paraphrases), dtype=np.float32)
    vecs_other = np.asarray(embedder.encode(others), dtype=np.float32)

    # Within the paraphrase set: every pair means the same thing, so every one
    # of these SHOULD be a cache hit.
    same = np.array(
        [
            float(vecs_same[i] @ vecs_same[j])
            for i, j in itertools.combinations(range(len(paraphrases)), 2)
        ]
    )
    # Across the two sets, and within the off-topic set: different questions,
    # so none of these may EVER be a cache hit.
    cross = np.array([float(a @ b) for a in vecs_same for b in vecs_other])
    within_other = np.array(
        [
            float(vecs_other[i] @ vecs_other[j])
            for i, j in itertools.combinations(range(len(others)), 2)
        ]
    )
    different = np.concatenate([cross, within_other])

    same_floor = float(same.min())
    different_ceiling = float(different.max())
    gap = same_floor - different_ceiling

    print()
    print("semantic cache")
    print(
        f"  same question, different wording  n={same.size:4d}  "
        f"min {same.min():.4f}  median {np.median(same):.4f}  max {same.max():.4f}"
    )
    print(
        f"  different questions               n={different.size:4d}  "
        f"min {different.min():.4f}  median {np.median(different):.4f}  max {different.max():.4f}"
    )
    print(f"  gap between them                  {gap:+.4f}")

    if gap <= 0:
        # Overlap means no threshold reuses every paraphrase without also
        # reusing an answer for a question that is not the same one. The safe
        # resolution is to favour missing over colliding.
        threshold = float(min(1.0, different_ceiling + 0.01))
        method = (
            f"the two distributions OVERLAP ({same_floor:.4f} vs {different_ceiling:.4f}). "
            f"No threshold catches every paraphrase without also colliding, so the "
            f"threshold is set just above the highest unrelated pair: the cache will "
            f"miss some paraphrases rather than ever answer one question with another's "
            f"answer."
        )
    else:
        # Not the midpoint. A miss costs a few hundred milliseconds; a collision
        # answers the wrong question. The threshold therefore sits most of the
        # way towards the paraphrase floor.
        threshold = float(different_ceiling + 0.8 * gap)
        method = (
            f"placed at 80% of the way from the highest unrelated pair "
            f"({different_ceiling:.4f}) to the lowest paraphrase pair ({same_floor:.4f}). "
            f"Not the midpoint: a cache MISS costs time, a cache COLLISION answers a "
            f"different question, so the asymmetry is deliberate."
        )

    print(f"  threshold                         {threshold:.4f}")
    print(f"  -> {method}")

    payload = {
        "threshold": threshold,
        "method": method,
        "same_question": {
            "n_pairs": int(same.size),
            "min": float(same.min()),
            "median": float(np.median(same)),
            "max": float(same.max()),
            "source": "all pairs among the 10 Phase 11.4 paraphrases of one question",
        },
        "different_questions": {
            "n_pairs": int(different.size),
            "min": float(different.min()),
            "median": float(np.median(different)),
            "max": float(different.max()),
            "source": "every paraphrase against every off-topic query, plus all pairs among the off-topic queries",
        },
        "gap": gap,
        "embedder": embedder.model_name,
        "provenance": "measured: cosine between query embeddings, on this corpus's query set",
    }
    args.out_cache.parent.mkdir(parents=True, exist_ok=True)
    args.out_cache.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out_guard}\nwrote {args.out_cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

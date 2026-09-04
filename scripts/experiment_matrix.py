#!/usr/bin/env python
"""Phase 8: the experiment matrix. Retrieval x SCR, one arm per cell.

This is the first experiment that exercises BOTH control dimensions, so it is
the first that can say anything about C3 (joint versus split control). It runs
on SQuAD because SQuAD is the only prepared dataset with chunk TEXT -- SIFT and
NYTimes are vector-only, and SCR has nothing to reduce without text.

**This is a COST-ONLY study and is labelled as one everywhere it is written.**
SCR's actual claim is answer quality per token, and answer quality needs a
generator; Phase 7 is blocked on the MSVC toolchain (confirmed: no
`llama-cpp-python` wheel exists for any version on this platform). So the matrix
measures latency, recall and context size. It does NOT measure whether the text
SCR discarded was needed, and no arm here should be read as evidence that it
was not.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_acceptance import index_root, load_all

from smartann.bench.metrics import recall_at_k_batch
from smartann.bench.writer import RunWriter
from smartann.config import resolve_paths
from smartann.controller import AimdConfig, AimdPolicy, StaticPolicy
from smartann.index.ecovector import EcoVectorIndex, ProbeResult
from smartann.scr import reduce_context
from smartann.telemetry.windows import WindowsMonitor
from smartann.types import (
    AnnConfig,
    Constraints,
    ControllerState,
    PerfHistory,
    QueryRecord,
    ScrConfig,
)

DATASET = "squad"
SCR_OFF = ScrConfig(3, 1, 1, 2, 0.0, enabled=False)
# window_size=2, not 3. Measured first: 99.3% of SQuAD chunks hold <= 3
# sentences (p50 = 3, max = 4, p50 length 385 chars), so a 3-sentence window
# yields exactly ONE window per chunk and SCR becomes a no-op BY CONSTRUCTION --
# a smoke run at window_size=3 returned 0.0% reduction while still paying the
# full embedding cost. SCR can only select among windows if the window is
# smaller than the chunk, so its applicability is a property of the chunking, not
# of the reducer (ADR-0035).
SCR_ON = ScrConfig(
    sliding_window_size=2,
    overlap_size=0,
    context_extension_size=0,
    top_windows=1,
    similarity_threshold=0.0,
    enabled=True,
)


@dataclass(frozen=True)
class Arm:
    name: str
    policy_kind: str  # "static" | "aimd"
    scr: ScrConfig

    @property
    def label(self) -> str:
        return f"{self.policy_kind}+scr" if self.scr.enabled else f"{self.policy_kind}+noscr"


ARMS = [
    Arm("A1", "static", SCR_OFF),  # neither dimension adapts: the baseline
    Arm("A2", "static", SCR_ON),  # SCR on, retrieval fixed  -> SCR's cost alone
    Arm("A3", "aimd", SCR_OFF),  # retrieval adapts, SCR off -> C1 alone
    Arm("A4", "aimd", SCR_ON),  # both -> C3's joint cell
]


def load_chunk_text(limit: int | None = None) -> dict[int, str]:
    """chunk_id -> text. Row i of base.npy is chunk_id i (Phase 1 invariant)."""
    con = sqlite3.connect(f"{resolve_paths().processed / DATASET / 'chunks.sqlite'}")
    q = "SELECT chunk_id, text FROM metadata ORDER BY chunk_id"
    if limit:
        q += f" LIMIT {limit}"
    out = dict(con.execute(q).fetchall())
    con.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds-per-arm", type=float, default=300.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    paths = resolve_paths()
    ip = index_root(DATASET, "ecovector")
    if ip is None:
        print(f"no ecovector index for {DATASET}; build it first", file=sys.stderr)
        return 1
    from smartann.index.storage import IndexMeta

    meta = IndexMeta.from_json(ip.meta)
    index = EcoVectorIndex.open(ip.root)
    _, queries, gt_ids, gt_dists = load_all(DATASET)
    queries = np.ascontiguousarray(queries, dtype=np.float32)

    print("loading chunk text and the embedder (SCR scores windows on the query path)", flush=True)
    chunk_text = load_chunk_text()
    from smartann.pipeline.embedder import Embedder

    embedder = Embedder()
    embedder.encode(["warm up the model so the first arm is not charged for it"])

    out_root = Path(args.out) if args.out else paths.results_root / "E_MATRIX"
    out_root.mkdir(parents=True, exist_ok=True)

    # Calibrate the target once, from arm A1's own quiet behaviour, so every arm
    # is judged against the same budget rather than one derived from itself.
    base_ann = AnnConfig(nprobe=8, ef_centroid=128, ef_list=64, k=args.k, per_cluster_k=32)
    warm = [index.search(queries[i], base_ann).t_total for i in range(200)]
    target_ms = 1.5 * float(np.percentile(warm, 95))
    print(
        f"latency target = 1.5 x quiet p95 at nprobe=8 = {target_ms:.2f} ms "
        f"(fixed once, before any arm runs)\n",
        flush=True,
    )
    constraints = Constraints(target_ms, 3 * target_ms, float("inf"), None)

    summary = []
    for arm in ARMS:
        policy = (
            StaticPolicy(base_ann, arm.scr, tuning_note=f"nprobe={base_ann.nprobe} fixed")
            if arm.policy_kind == "static"
            else AimdPolicy(
                AimdConfig(nprobe_min=1, nprobe_max=32, warmup_queries=20, control_period_s=4.05),
                base_ann,
                arm.scr,
            )
        )
        policy.reset()
        monitor = WindowsMonitor(hz=10.0).start()
        writer = RunWriter(
            out_root / arm.name,
            run_id=arm.name,
            experiment_id="E_MATRIX",
            capabilities=monitor.capabilities(),
        )
        print(f"=== {arm.name}: {arm.label} ===", flush=True)
        writer.event("arm_start", arm=arm.name, label=arm.label)

        lat, rec, toks, red = [], [], [], []
        t_end = time.time() + args.seconds_per_arm
        qi = 0
        while time.time() < t_end:
            row = qi % queries.shape[0]
            q = queries[row]
            device = monitor.latest()

            probe = index.probe(q, base_ann)
            state = ControllerState(device, probe.features, _history(lat, target_ms))
            decision = policy.decide_timed(state, constraints)

            want = min(decision.config.ann.nprobe, probe.cluster_ids.size)
            narrowed = ProbeResult(
                probe.cluster_ids[:want],
                probe.centroid_dists[:want],
                probe.features,
                probe.t_centroid_ms,
                probe.n_dist_modeled,
            )
            r = index.finish(q, narrowed, decision.config.ann)

            # --- SCR on the retrieved chunks -------------------------------
            t_scr0 = time.perf_counter()
            ctx_chars, red_ratio = 0, 0.0
            for cid in r.ids.tolist():
                text = chunk_text.get(int(cid), "")
                if not text:
                    continue
                red_res = reduce_context(text, q, embedder.encode, decision.config.scr)
                ctx_chars += red_res.chars_out
                red_ratio += red_res.reduction_ratio
            t_scr = (time.perf_counter() - t_scr0) * 1000.0
            n_ctx = max(1, r.ids.size)

            se, ov = _recall(r, gt_ids, gt_dists, row, args.k)
            total = r.t_total + probe.t_centroid_ms + t_scr
            lat.append(total)
            rec.append(se)
            toks.append(round(ctx_chars / 4.0))
            red.append(red_ratio / n_ctx)

            writer.append(
                _record(
                    arm,
                    writer.run_id,
                    decision,
                    r,
                    se,
                    ov,
                    t_scr,
                    total,
                    ctx_chars,
                    device,
                    meta,
                    row,
                )
            )
            policy.update(None)
            qi += 1

        writer.event("arm_end")
        monitor.stop()
        writer.write(
            extra={
                "arm": arm.name,
                "label": arm.label,
                "cost_only": True,
                "latency_target_ms": target_ms,
                "scr": arm.scr.__dict__,
                "policy": arm.policy_kind,
                "caveat": "COST-ONLY. No generator exists (Phase 7 blocked on MSVC), so "
                "this measures latency, recall and context size. It does NOT "
                "measure whether the text SCR discarded was needed.",
            }
        )
        s = {
            "arm": arm.name,
            "label": arm.label,
            "n": len(lat),
            "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95)),
            "violation_pct": float(np.mean(np.asarray(lat) > target_ms) * 100),
            "recall": float(np.mean(rec)),
            "approx_tokens": float(np.mean(toks)),
            "reduction": float(np.mean(red)),
        }
        summary.append(s)
        print(
            f"  n={s['n']:,}  p95 {s['p95_ms']:7.2f} ms  viol {s['violation_pct']:5.1f}%  "
            f"recall {s['recall']:.4f}  ~tokens {s['approx_tokens']:7.0f}  "
            f"reduction {s['reduction'] * 100:5.1f}%\n",
            flush=True,
        )

    (out_root / "summary.json").write_text(
        json.dumps({"target_ms": target_ms, "cost_only": True, "arms": summary}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


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


def _recall(r, gt_ids, gt_dists, row, k):  # type: ignore[no-untyped-def]
    ids = np.full((1, k), -1, dtype=np.int64)
    d = np.full((1, k), np.inf, dtype=np.float64)
    m = min(r.ids.size, k)
    ids[0, :m], d[0, :m] = r.ids[:m], r.dists[:m]
    se, ov = recall_at_k_batch(ids, d, gt_ids[row : row + 1], gt_dists[row : row + 1], k=k)
    return float(se[0]), float(ov[0])


def _record(arm, run_id, decision, r, se, ov, t_scr, total, ctx_chars, device, meta, row):  # type: ignore[no-untyped-def]
    return QueryRecord(
        query_id=f"q{row}",
        run_id=run_id,
        t_wall=time.time(),
        dataset=DATASET,
        index_variant=meta.index_variant,
        index_config_hash=meta.config_hash,
        cache_state="warm",
        config=decision.config,
        policy_name=arm.label,
        controller_overhead_us=decision.overhead_us,
        retrieved_ids=r.ids.tolist(),
        retrieved_dists=r.dists.tolist(),
        recall_at_k=se,
        recall_at_k_index=ov,
        recall_proxy=0.0,
        t_embed=0.0,
        t_centroid=r.t_centroid,
        t_cluster_load=r.t_cluster_load,
        t_cluster_search=r.t_cluster_search,
        t_merge=r.t_merge,
        t_scr=t_scr,
        t_generate=0.0,
        t_total=total,
        n_distance_computations=r.n_distance_computations,
        n_dist_exact=r.n_dist_exact,
        n_dist_modeled=r.n_dist_modeled,
        n_flat_clusters_scanned=r.n_flat_clusters_scanned,
        n_cluster_loads=r.n_cluster_loads,
        bytes_read=r.bytes_read,
        energy_mj_model=0.0,
        energy_mj_measured=None,
        peak_rss_mb=r.peak_rss_mb,
        anon_rss_mb=r.anon_rss_mb,
        device=device,
        context_tokens=round(ctx_chars / 4.0),
        answer=None,
        em=None,
        f1=None,
    )


if __name__ == "__main__":
    sys.exit(main())

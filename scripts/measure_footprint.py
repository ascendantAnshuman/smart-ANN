#!/usr/bin/env python
"""Measure one index's resident footprint in a FRESH process. Prints JSON.

Run as a subprocess, one dataset per invocation, because a process-wide RSS
delta cannot isolate one index inside a long-lived process: allocator residue
from whatever ran earlier lands in the same counter. Measuring NYTimes after
SIFT in one process gave 38.24 MB against 5.18 MB for the same index in a
different ordering -- a 7x swing on a number the report quotes against the paper.

A fresh process has one index and one query loop in it, so the delta is the
index.
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_acceptance import index_root, load_all

from smartann.index.ecovector import EcoVectorIndex, _anon_mb
from smartann.types import AnnConfig


def main() -> int:
    dataset = sys.argv[1]
    n_queries = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    proc = psutil.Process(os.getpid())
    gc.collect()
    before = _anon_mb(proc.memory_info())

    ip = index_root(dataset, "ecovector")
    if ip is None:
        print(json.dumps({"error": f"no ecovector index for {dataset}"}))
        return 1
    idx = EcoVectorIndex.open(ip.root)
    _, queries, _, _ = load_all(dataset)
    cfg = AnnConfig(nprobe=16, ef_centroid=64, ef_list=128, k=10, per_cluster_k=32)

    peak = before
    for qi in range(min(n_queries, queries.shape[0])):
        r = idx.search(queries[qi], cfg)
        peak = max(peak, r.anon_rss_mb)

    base_bytes = (
        (Path(ip.root).parent.parent.parent / "processed" / dataset / "base.npy").stat().st_size
    )
    print(
        json.dumps(
            {
                "dataset": dataset,
                "before_mb": before,
                "peak_mb": peak,
                "resident_mb": peak - before,
                "base_mb": base_bytes / 2**20,
                "n_queries": min(n_queries, int(queries.shape[0])),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

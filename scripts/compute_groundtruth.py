#!/usr/bin/env python
"""Exact top-k neighbours by brute force, resumable.

Only needed for QA datasets: both ANN datasets ship exact top-100 in their
ANN-Benchmarks HDF5 files (ADR-0011).

    python scripts/compute_groundtruth.py squad
    python scripts/compute_groundtruth.py squad --k 100 --validate
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from smartann.config import resolve_paths
from smartann.data.download import sha256_file
from smartann.data.groundtruth import compute_topk
from smartann.data.manifest import DataManifest
from smartann.data.readers import load_processed
from smartann.data.registry import load_registry, spec_for


def validate_against_scipy(
    base: np.ndarray, queries: np.ndarray, space: str, seed: int = 0
) -> bool:
    """Check a 200-query x 5,000-vector subset against scipy, exactly.

    Small enough to run every time; large enough to catch a metric or
    tie-breaking error.
    """
    from scipy.spatial.distance import cdist

    rng = np.random.default_rng(seed)
    n_q = min(200, queries.shape[0])
    n_b = min(5_000, base.shape[0])
    q = np.ascontiguousarray(queries[rng.choice(queries.shape[0], n_q, replace=False)])
    b = np.ascontiguousarray(base[:n_b])

    ours = compute_topk(b, q, k=10, space=space, block_mb=64, quiet=True).neighbors
    metric = "cosine" if space in ("ip", "cosine") else "euclidean"
    theirs = np.argsort(cdist(q, b, metric=metric), axis=1)[:, :10].astype(np.int32)

    agree = bool(np.array_equal(ours, theirs))
    n_disagree = int(np.count_nonzero(ours != theirs))
    print(
        f"  scipy cross-check ({n_q} queries x {n_b} vectors, metric={metric}): "
        f"{'EXACT AGREEMENT' if agree else f'{n_disagree} position(s) differ'}"
    )
    return agree


def main(argv: list[str] | None = None) -> int:
    known = sorted(load_registry())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help=f"dataset name; known: {known}")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate", action="store_true", help="cross-check against scipy")
    args = parser.parse_args(argv)

    spec = spec_for(args.name)
    paths = resolve_paths()
    out_dir = paths.processed / args.name
    gt_path = out_dir / "groundtruth_top100.npy"

    if spec.ships_groundtruth:
        print(f"{args.name} ships exact top-100 in its HDF5 file; nothing to compute.")
        return 0
    if gt_path.exists() and not args.force:
        print(f"{gt_path} exists; pass --force to recompute.")
        return 0

    base = load_processed(out_dir, "base")
    queries = load_processed(out_dir, "queries")
    print(
        f"{args.name}: {queries.shape[0]:,} queries x {base.shape[0]:,} base vectors, "
        f"dim={base.shape[1]}, space={spec.search_space}"
    )

    if args.validate and not validate_against_scipy(base, queries, spec.search_space):
        print("  cross-check FAILED -- refusing to write ground truth", file=sys.stderr)
        return 1

    result = compute_topk(
        base,
        queries,
        k=args.k,
        space=spec.search_space,
        checkpoint=out_dir / "groundtruth.ckpt.npz",
    )
    np.save(gt_path, result.neighbors)

    manifest = DataManifest(paths.data_root)
    manifest.record(
        gt_path,
        dataset=args.name,
        stage="groundtruth",
        sha256=sha256_file(gt_path),
        extra={"k": args.k, "space": spec.search_space, "method": "exact brute force"},
    )
    manifest.write()

    ckpt = out_dir / "groundtruth.ckpt.npz"
    if ckpt.exists():
        ckpt.unlink()
    print(f"wrote {gt_path}  shape={result.neighbors.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

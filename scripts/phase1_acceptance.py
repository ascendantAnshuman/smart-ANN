#!/usr/bin/env python
"""Phase 1 acceptance checks 2-7, run against the real prepared data.

Check 1 (`smartann data status`) and check 6 (pytest/ruff/mypy) are run
separately as their own commands. Everything here reports PASS or FAIL with the
measured numbers; nothing is asserted silently.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import psutil

from smartann.config import resolve_paths
from smartann.data.manifest import DataManifest
from smartann.data.readers import load_processed
from smartann.data.registry import spec_for

RESULTS: list[tuple[str, bool, str]] = []


def report(name: str, passed: bool, detail: str) -> None:
    RESULTS.append((name, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}\n       {detail}\n")


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 2**20


# ---------------------------------------------------------------------------
def check2_sift_groundtruth_selfcheck(n_queries: int = 200, k: int = 10) -> None:
    """Recompute SIFT top-10 by brute force and compare to the shipped neighbors."""
    paths = resolve_paths()
    d = paths.processed / "sift1m"
    base = load_processed(d, "base")
    queries = load_processed(d, "queries")
    truth = load_processed(d, "groundtruth")

    rng = np.random.default_rng(0)
    idx = rng.choice(queries.shape[0], n_queries, replace=False)
    q = np.ascontiguousarray(queries[np.sort(idx)], dtype=np.float32)

    # Exact L2 in blocks; ||q||^2 is constant per query and cannot reorder.
    best = np.full((n_queries, k), np.inf, dtype=np.float32)
    best_idx = np.full((n_queries, k), -1, dtype=np.int32)
    block = 100_000
    for lo in range(0, base.shape[0], block):
        hi = min(lo + block, base.shape[0])
        b = np.ascontiguousarray(base[lo:hi], dtype=np.float32)
        d2 = (b * b).sum(1)[None, :] - 2.0 * (q @ b.T)
        cand_idx = np.broadcast_to(np.arange(lo, hi, dtype=np.int32)[None, :], d2.shape)
        merged = np.concatenate([best, d2.astype(np.float32)], axis=1)
        merged_i = np.concatenate([best_idx, cand_idx], axis=1)
        order = np.lexsort((merged_i, merged), axis=1)[:, :k]
        rows = np.arange(n_queries)[:, None]
        best, best_idx = merged[rows, order], merged_i[rows, order]

    shipped = np.asarray(truth[np.sort(idx), :k], dtype=np.int32)
    exact = bool(np.array_equal(best_idx, shipped))
    overlap = np.mean([len(set(a) & set(b_)) / k for a, b_ in zip(best_idx, shipped, strict=True)])
    report(
        "2. SIFT1M ground-truth self-check",
        exact,
        f"{n_queries} random queries, top-{k}: "
        f"{'exact positional agreement' if exact else 'MISMATCH'} with the shipped "
        f"neighbors array; mean set overlap {overlap:.4f}",
    )


def check3_nytimes_metric() -> None:
    """The HDF5 distance attribute and the registry space must agree."""
    import h5py

    paths = resolve_paths()
    spec = spec_for("nytimes")
    hdf5 = paths.raw / "nytimes" / spec.artifact("hdf5").filename
    with h5py.File(hdf5, "r") as f:
        attr = f.attrs["distance"]
    attr = attr.decode() if isinstance(attr, bytes) else str(attr)

    ok = attr == "angular" and spec.space == "cosine" and spec.search_space == "ip"
    report(
        "3. NYTimes metric check",
        ok,
        f"HDF5 distance attribute = {attr!r};  registry space = {spec.space!r};  "
        f"derived search_space = {spec.search_space!r} (normalized={spec.normalized})",
    )


def check4_qa_round_trip(n: int = 5) -> None:
    """Re-embed 5 random chunks and compare to their stored vectors."""
    import sqlite3

    from smartann.pipeline.embedder import Embedder

    paths = resolve_paths()
    spec = spec_for("squad")
    db = paths.processed / "squad" / "chunks.sqlite"
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    n_chunks = int(conn.execute("SELECT COUNT(*) FROM metadata").fetchone()[0])
    rng = np.random.default_rng(0)
    picked = sorted(rng.choice(n_chunks, n, replace=False).tolist())

    embedder = Embedder(
        model_name=spec.embedding_model or "thenlper/gte-small",
        cache_path=None,  # deliberately uncached: re-embed for real
        dim=spec.dim,
    )
    base = load_processed(paths.processed / "squad", "base")

    sims = []
    lines = []
    for cid in picked:
        row = conn.execute("SELECT * FROM metadata WHERE chunk_id = ?", (cid,)).fetchone()
        doc = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (row["doc_id"],)).fetchone()
        stored_blob = conn.execute(
            "SELECT vector FROM embeddings WHERE emb_id = ?", (cid,)
        ).fetchone()[0]
        stored = np.frombuffer(stored_blob, dtype=np.float32)

        fresh = embedder.encode([row["text"]])[0]
        sim = float(fresh @ stored)
        sims.append(sim)

        # base.npy row i must be chunk_id i -- Phase 2 depends on this contract.
        npy_sim = float(np.asarray(base[cid]) @ stored)
        lines.append(
            f"chunk {cid} (doc {row['doc_id']} '{doc['title'][:24]}' "
            f"offset {row['char_offset']} len {row['char_len']}): "
            f"cos(re-embed, sqlite)={sim:.6f}  cos(base.npy, sqlite)={npy_sim:.6f}"
        )
    embedder.close()
    conn.close()

    ok = all(s >= 0.999 for s in sims)
    report(
        "4. QA round-trip (sqlite text -> re-embed -> stored vector)",
        ok,
        f"min cosine {min(sims):.6f} over {n} chunks (threshold 0.999)\n       "
        + "\n       ".join(lines),
    )


def check5_mmap_does_not_double_rss() -> None:
    """Loading SIFT1M's 512 MB base twice must not cost 1 GB."""
    paths = resolve_paths()
    d = paths.processed / "sift1m"

    before = rss_mb()
    a = load_processed(d, "base")
    after_first = rss_mb()
    b = load_processed(d, "base")
    after_second = rss_mb()

    nbytes_mb = a.shape[0] * a.shape[1] * 4 / 2**20
    delta_first = after_first - before
    delta_second = after_second - after_first
    ok = delta_second < nbytes_mb * 0.10 and delta_first < nbytes_mb * 0.50
    report(
        "5. Memory-mapped base does not double RSS",
        ok,
        f"array is {nbytes_mb:.0f} MB on disk\n       "
        f"RSS before        {before:8.1f} MB\n       "
        f"RSS after load #1 {after_first:8.1f} MB  (delta {delta_first:+.1f} MB)\n       "
        f"RSS after load #2 {after_second:8.1f} MB  (delta {delta_second:+.1f} MB)\n       "
        f"both arrays are memmaps: {isinstance(a, np.memmap) and isinstance(b, np.memmap)}",
    )


def check7_manifest_completeness() -> None:
    """Every artifact under the data root must have a MANIFEST.json entry."""
    paths = resolve_paths()
    m = DataManifest(paths.data_root)
    missing = m.missing_files()
    unrecorded = m.unrecorded_files()
    n = len(m.for_dataset("sift1m")) + len(m.for_dataset("nytimes")) + len(m.for_dataset("squad"))

    ok = not missing and not unrecorded
    detail = (
        f"{n} artifacts recorded, {m.total_bytes() / 2**30:.2f} GiB total\n       "
        f"recorded-but-absent: {missing or 'none'}\n       "
        f"present-but-unrecorded: {unrecorded or 'none'}"
    )
    report("7. MANIFEST.json completeness", ok, detail)


def summary() -> None:
    paths = resolve_paths()
    m = DataManifest(paths.data_root)
    print("=" * 74)
    for name, passed, _ in RESULTS:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print("=" * 74)

    print("\nResolved values:")
    for name in ("sift1m", "nytimes", "squad"):
        spec = spec_for(name)
        base_p = paths.processed / name / "base.npy"
        if base_p.exists():
            n_base = int(np.load(base_p, mmap_mode="r").shape[0])
            print(
                f"  {name:8s} n_base={n_base:>9,}  dim={spec.dim:>3}  "
                f"space={spec.space:<7} search_space={spec.search_space:<3} "
                f"Nc={spec.resolve_nc(n_base)}"
            )
    print(f"\nTotal recorded under {paths.data_root}: {m.total_bytes() / 2**30:.2f} GiB")


def main() -> int:
    check2_sift_groundtruth_selfcheck()
    check3_nytimes_metric()
    check4_qa_round_trip()
    check5_mmap_does_not_double_rss()
    check7_manifest_completeness()
    summary()
    return 0 if all(p for _, p, _ in RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Phase 1 acceptance checks 2-7, run against the real prepared data.

Check 1 (`smartann data status`) and check 6 (pytest/ruff/mypy) are run
separately as their own commands. Everything here reports PASS or FAIL with the
measured numbers; nothing is asserted silently.
"""

from __future__ import annotations

import os
import sys
from functools import partial

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

    detail = (
        f"{n_queries} random queries, top-{k}: "
        f"{'exact positional agreement' if exact else 'positional MISMATCH'} with the "
        f"shipped neighbors array; mean set overlap {overlap:.4f}"
    )
    ok = exact
    if not exact:
        # A mismatch is only a bug if it means we found *worse* neighbours. If
        # every disagreement is an exact distance tie, the two answers are
        # equally correct and differ only in tie-breaking convention. Diagnose
        # rather than assert, so the failure mode is identifiable at a glance.
        ties, deficit = _diagnose_disagreement(base, q, best_idx, shipped, larger_is_better=False)
        ok = ties and deficit <= 0.0
        detail += (
            f"\n       diagnosis: every disagreement is an exact distance tie: {ties}"
            f"\n       max quality deficit of our result vs shipped: {deficit:.3e}"
            f"\n       => {'equally correct, tie-break convention only' if ok else 'GENUINELY WORSE -- metric or normalisation is wrong'}"
        )
    report("2. SIFT1M ground-truth self-check", ok, detail)


def _score(q64: np.ndarray, base: np.ndarray, larger_is_better: bool, j: int) -> float:
    """Exact float64 score of base row *j* against query *q64*.

    Module level, with the candidate index last so callers can bind the query
    eagerly via functools.partial rather than closing over a loop variable.
    float64 so the tie verdict is not itself a rounding artifact.
    """
    v = base[j].astype(np.float64)
    return float(q64 @ v) if larger_is_better else float(np.sum((q64 - v) ** 2))


def _diagnose_disagreement(
    base: np.ndarray,
    queries: np.ndarray,
    ours: np.ndarray,
    theirs: np.ndarray,
    larger_is_better: bool,
) -> tuple[bool, float]:
    """Are two top-k answers equally good? Returns (all_ties, max_deficit).

    Computed in float64 so the verdict is not itself a rounding artifact.
    """
    all_ties = True
    max_deficit = 0.0
    for i in range(queries.shape[0]):
        extra = set(ours[i].tolist()) - set(theirs[i].tolist())
        missing = set(theirs[i].tolist()) - set(ours[i].tolist())
        if not extra:
            continue
        # partial binds this iteration's query eagerly
        score = partial(_score, queries[i].astype(np.float64), base, larger_is_better)

        for a in extra:
            for b in missing:
                if abs(score(a) - score(b)) > 0.0:
                    all_ties = False
        ours_total = sum(score(j) for j in ours[i].tolist())
        theirs_total = sum(score(j) for j in theirs[i].tolist())
        deficit = (theirs_total - ours_total) if larger_is_better else (ours_total - theirs_total)
        max_deficit = max(max_deficit, deficit)
    return all_ties, max_deficit


def check3b_nytimes_groundtruth_selfcheck(n_queries: int = 200, k: int = 10) -> None:
    """Same self-check for NYTimes, which additionally exercises the remapping."""
    paths = resolve_paths()
    d = paths.processed / "nytimes"
    base = load_processed(d, "base")
    queries = load_processed(d, "queries")
    truth = load_processed(d, "groundtruth")

    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(queries.shape[0], n_queries, replace=False))
    q = np.ascontiguousarray(queries[idx], dtype=np.float32)

    best = np.full((n_queries, k), -np.inf, dtype=np.float32)
    best_idx = np.full((n_queries, k), -1, dtype=np.int32)
    for lo in range(0, base.shape[0], 50_000):
        hi = min(lo + 50_000, base.shape[0])
        b = np.ascontiguousarray(base[lo:hi], dtype=np.float32)
        s = q @ b.T
        ci = np.broadcast_to(np.arange(lo, hi, dtype=np.int32)[None, :], s.shape)
        ms = np.concatenate([best, s], 1)
        mi = np.concatenate([best_idx, ci], 1)
        order = np.lexsort((mi, -ms), axis=1)[:, :k]
        rows = np.arange(n_queries)[:, None]
        best, best_idx = ms[rows, order], mi[rows, order]

    shipped = np.asarray(truth[idx, :k], dtype=np.int32)
    exact = bool(np.array_equal(best_idx, shipped))
    overlap = np.mean([len(set(a) & set(b_)) / k for a, b_ in zip(best_idx, shipped, strict=True)])

    detail = (
        f"{n_queries} random queries, top-{k} on the COMPACTED base "
        f"({base.shape[0]:,} vectors after dropping 239 zero-norm rows): "
        f"{'exact positional agreement' if exact else 'positional MISMATCH'}; "
        f"mean set overlap {overlap:.4f}"
    )
    ok = exact
    if not exact:
        ties, deficit = _diagnose_disagreement(base, q, best_idx, shipped, larger_is_better=True)
        ok = ties and deficit <= 0.0
        detail += (
            f"\n       diagnosis: every disagreement is an exact cosine tie: {ties}"
            f"\n       max quality deficit of our result vs shipped: {deficit:.3e}"
            f"\n       => {'equally correct; NYTimes contains duplicate/collinear vectors' if ok else 'GENUINELY WORSE -- remapping is wrong'}"
        )
    report("3b. NYTimes ground-truth self-check (validates the remap)", ok, detail)


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


def check8_groundtruth_distances() -> None:
    """The Phase 1 addendum: recomputed distances, ordering, and units.

    Three separate claims, reported together because they fail together:
      (a) every ground-truth row is non-decreasing in distance -- the sharp test
          of ADR-0014, since the ids were shipped by the dataset and remapped,
          while these distances were computed from the compacted vectors;
      (b) our values reproduce the PUBLISHED distances in the HDF5, which is
          independent of anything in this repo;
      (c) the stored units are hnswlib's, so no transform sits between a
          retrieved distance and the threshold it is compared against.
    """
    import h5py

    from smartann.data.groundtruth import DIST_CONVENTION, check_row_monotonic

    paths = resolve_paths()
    # ANN-Benchmarks stores euclidean as plain L2 and angular as 1 - cos.
    # Determined empirically against the files, not assumed: our first guess
    # (sqrt(2-2cos)) was wrong by a median of 0.48.
    published = {
        "sift1m": ("sift-128-euclidean.hdf5", lambda p: p**2, "published^2"),
        "nytimes": ("nytimes-256-angular.hdf5", lambda p: p, "published (identical)"),
    }

    lines: list[str] = []
    ok = True
    for name in ("sift1m", "nytimes", "squad"):
        d = paths.processed / name
        path = d / "groundtruth_dists_top100.npy"
        if not path.exists():
            report("8. ground-truth distances", False, f"{path} is missing")
            return
        dists = np.load(path).astype(np.float64)
        rep = check_row_monotonic(dists)
        ok &= rep.ok
        units = DIST_CONVENTION[spec_for(name).search_space]
        line = (
            f"{name:8s} {dists.shape[0]:>6,}x{dists.shape[1]:<3} {units:<13} "
            f"monotonic {rep.n_queries - rep.n_violating_rows:,}/{rep.n_queries:,} "
            f"worst {rep.worst_violation:+.2e}"
        )

        if name in published:
            fname, transform, label = published[name]
            with h5py.File(paths.raw / name / fname, "r") as f:
                pub = np.asarray(f["distances"], dtype=np.float64)
                keep = np.linalg.norm(np.asarray(f["test"], dtype=np.float32), axis=1) > 0
            pub = transform(pub[keep][:, : dists.shape[1]])
            err = np.abs(dists - pub)
            # Combined band, not pure relative. NYTimes' 1-cos goes to ~0 for
            # near-duplicate vectors, where a 3.6e-07 absolute error reads as a
            # 3.6e+05 relative one -- the same trap as signed similarities in
            # score-equality recall, and it caught this check first.
            band = 1e-5 + 1e-4 * np.abs(pub)
            matched = float(np.mean(err <= band))
            ok &= matched == 1.0
            line += f"\n           vs {label}: max abs {err.max():.2e}, {matched:.6f} inside atol+rtol|x|"
        else:
            line += "\n           no published reference (we computed this ground truth)"
        lines.append(line)

    report("8. ground-truth distances", ok, "\n       ".join(lines))


def check9_tie_structure() -> None:
    """How much recall the two metrics can disagree about, per dataset.

    Not pass/fail -- a measurement. If ties at the k-boundary were rare, the
    score-equality metric would be an expensive way to compute the index-overlap
    one, and Phase 2 would be carrying complexity for nothing.
    """
    paths = resolve_paths()
    k = 10
    lines = []
    for name in ("sift1m", "nytimes", "squad"):
        dists = np.load(paths.processed / name / "groundtruth_dists_top100.npy").astype(np.float64)
        at_k = dists[:, k - 1]
        band = 1e-6 + 1e-6 * np.abs(at_k)
        # neighbours beyond depth k that are indistinguishable from the k-th
        extra = np.sum(np.abs(dists[:, k:] - at_k[:, None]) <= band[:, None], axis=1)
        lines.append(
            f"{name:8s} k={k}: {float(np.mean(extra > 0)) * 100:5.2f}% of queries have a tie "
            f"at the boundary, mean {extra.mean():.3f} extra, max {int(extra.max())}"
        )
    report("9. tie structure at the k-boundary (measurement)", True, "\n       ".join(lines))


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
    check3b_nytimes_groundtruth_selfcheck()
    check4_qa_round_trip()
    check5_mmap_does_not_double_rss()
    check7_manifest_completeness()
    check8_groundtruth_distances()
    check9_tie_structure()
    summary()
    return 0 if all(p for _, p, _ in RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Phase 2 acceptance checks, run against real built indexes.

Every check ends in a verdict against a stated threshold (ADR-0020). The three
that are genuinely measurements rather than pass/fail -- probe cost, resident
footprint against the paper, build cost -- still print their numbers and still
carry an explicit verdict, so none of them is a paragraph to read carefully.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil

from smartann.bench.metrics import recall_at_k_batch, tie_rate
from smartann.config import config_hash, load_yaml, resolve_paths
from smartann.data.readers import load_ground_truth, load_processed
from smartann.data.registry import spec_for
from smartann.index.baselines import FlatHnswIndex, compare_note
from smartann.index.ecovector import EcoVectorIndex, _anon_mb
from smartann.index.storage import IndexBuildConfig, IndexMeta, IndexPaths, memory_model_mb
from smartann.types import AnnConfig

RESULTS: list[tuple[str, bool, str]] = []
NL = chr(10) + "       "

# MobileRAG Figure 6: EcoVector's reported resident footprint. If ours lands in
# this neighbourhood that is independent evidence the index is built correctly,
# and it is the cheapest validation of the whole implementation available.
MOBILERAG_FIG6_MB = {"sift1m": 9.67, "nytimes": 3.53}

# Check 8's verdict table, as ruled.
PROBE_HEALTHY, PROBE_WORKABLE = 0.15, 0.35


def report(name: str, passed: bool, detail: str) -> None:
    RESULTS.append((name, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}{chr(10)}       {detail}{chr(10)}", flush=True)


CONFIG_DIR = Path("configs/index")


def index_root(dataset: str, variant: str) -> IndexPaths | None:
    """The index built from configs/index/<dataset>_<variant>.yaml, or None.

    Derived from the config rather than discovered on disk. Picking the
    lexicographically largest hash, or even the newest mtime, means the
    acceptance numbers silently describe whichever artifact happened to sort or
    land last -- which is fine until the first --force rebuild and wrong
    thereafter, with no symptom. The config file is what the build was told to
    do, so it is what the check should ask about.
    """
    cfg_path = CONFIG_DIR / f"{dataset}_{variant}.yaml"
    if not cfg_path.exists():
        return None
    cfg = load_yaml(cfg_path, IndexBuildConfig)
    root = resolve_paths().data_root / "index" / dataset / config_hash(cfg)
    return IndexPaths(root) if (root / "meta.json").exists() else None


def load_all(dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths = resolve_paths()
    d = paths.processed / dataset
    gt = load_ground_truth(d, paths.data_root)
    return (
        load_processed(d, "base"),
        np.asarray(load_processed(d, "queries")),
        np.asarray(gt.ids),
        np.asarray(gt.dists),
    )


# ---------------------------------------------------------------------------
def check1a_exhaustive_equivalence(dataset: str = "sift1m", n_sub: int = 10_000) -> None:
    """With every cluster probed, EcoVector must be exact.

    Deliberately on a 10k subset with nc=200 (mean cluster 50) so that BOTH
    storage paths are exercised: the k-means size distribution has a long enough
    left tail that a meaningful minority land under min_graph_size=32. At least
    5 clusters on each path is required -- lopsided coverage would leave one
    path effectively untested by the strongest check in the project.
    """
    from smartann.index.builder import build_ecovector
    from smartann.index.validation import check_equivalence

    paths = resolve_paths()
    base = np.asarray(load_processed(paths.processed / dataset, "base"))[:n_sub]
    queries = np.asarray(load_processed(paths.processed / dataset, "queries"))[:50]
    # From the registry, like every other call site. A literal here would be a
    # second source of truth for exactly the thing IndexBuildConfig refuses to
    # carry -- and it would be wrong the moment a dataset changed metric.
    space = spec_for(dataset).search_space

    out = IndexPaths(paths.data_root / "index" / f"_check1a_{dataset}")
    res = build_ecovector(
        base,
        dataset=dataset,
        space=space,
        nc=200,
        m=16,
        ef_construction=100,
        min_graph_size=32,
        seed=42,
        config_hash="check1a",
        out=out,
        quiet=True,
    )
    m = res.meta
    coverage_ok = m.n_flat_clusters >= 5 and m.n_graph_clusters >= 5

    idx = EcoVectorIndex.open(out.root)
    rep = check_equivalence(idx, base, queries, k=10)

    report(
        f"1a. exhaustive equivalence ({dataset} {n_sub:,}-vector subset, nc=200)",
        rep.ok and coverage_ok,
        f"{rep.n_score_equal}/{rep.n_queries} queries score-equal to brute force, "
        f"{rep.n_exact_id_matches} id-identical, worst distance error "
        f"{rep.max_distance_error:.3e}{NL}"
        f"cluster coverage: {m.n_graph_clusters} graph / {m.n_flat_clusters} flat / "
        f"{m.n_empty_clusters} empty (need >=5 on each path: "
        f"{'OK' if coverage_ok else 'LOPSIDED'}){NL}"
        f"sizes: largest {m.largest_cluster}, smallest non-empty "
        f"{m.smallest_nonempty_cluster}; {m.n_duplicate_centroids} duplicate centroid(s)",
    )


def check1b_high_effort_recall(dataset: str, threshold: float = 0.95) -> None:
    """A recall datapoint at high effort. NOT an equivalence proof."""
    ip = index_root(dataset, "ecovector")
    if ip is None:
        report(f"1b. high-effort recall ({dataset})", False, "no ecovector index built")
        return
    idx = EcoVectorIndex.open(ip.root)
    _, queries, gi, gd = load_all(dataset)
    nq = min(1000, queries.shape[0])
    cfg = AnnConfig(nprobe=64, ef_centroid=128, ef_list=512, k=10, per_cluster_k=64)

    ids, dists = _run(idx, queries[:nq], cfg)
    se, ov = recall_at_k_batch(ids, dists, gi[:nq], gd[:nq], k=10)
    report(
        f"1b. high-effort recall ({dataset}, nprobe=64 ef_list=512)",
        float(se.mean()) >= threshold,
        f"score-equality {se.mean():.4f}  index-overlap {ov.mean():.4f}  "
        f"(threshold {threshold:.2f}, tie rate {tie_rate(gd[:nq], 10) * 100:.2f}%){NL}"
        f"{nq} queries; this is a datapoint, not a correctness proof -- check 1a is that",
    )


def check2_ef_monotonicity(dataset: str = "sift1m") -> None:
    """Per-query effort must trade recall for work. The controller's premise."""
    ip = index_root(dataset, "ecovector")
    if ip is None:
        report(f"2. per-query effort changes recall ({dataset})", False, "no index built")
        return
    idx = EcoVectorIndex.open(ip.root)
    _, queries, gi, gd = load_all(dataset)
    nq = min(300, queries.shape[0])

    rows = []
    recalls = []
    for nprobe, ef in ((1, 16), (2, 32), (4, 64), (8, 128), (16, 256), (32, 512)):
        cfg = AnnConfig(
            nprobe=nprobe, ef_centroid=max(64, nprobe), ef_list=ef, k=10, per_cluster_k=32
        )
        t0 = time.perf_counter()
        ids, dists = _run(idx, queries[:nq], cfg)
        ms = (time.perf_counter() - t0) * 1000 / nq
        se, _ = recall_at_k_batch(ids, dists, gi[:nq], gd[:nq], k=10)
        recalls.append(float(se.mean()))
        rows.append(
            f"nprobe={nprobe:<3} ef_list={ef:<4} recall {se.mean():.4f}  {ms:6.3f} ms/query"
        )

    monotone = all(b >= a - 1e-9 for a, b in itertools.pairwise(recalls))
    report(
        f"2. per-query effort changes recall monotonically ({dataset})",
        monotone and recalls[-1] > recalls[0],
        NL.join(rows) + f"{NL}monotone: {monotone}; span {recalls[0]:.4f} -> {recalls[-1]:.4f}",
    )


def check4_ef_efficiency_and_wallclock(dataset: str) -> None:
    """Two claims, reported separately because they point opposite ways here."""
    eco_p, flat_p = index_root(dataset, "ecovector"), index_root(dataset, "flat_hnsw")
    if eco_p is None or flat_p is None:
        report(f"4. ef-efficiency vs flat HNSW ({dataset})", False, "both indexes must be built")
        return
    eco, flat = EcoVectorIndex.open(eco_p.root), FlatHnswIndex.open(flat_p.root)
    _, queries, gi, gd = load_all(dataset)
    nq = min(300, queries.shape[0])
    target = 0.90

    def sweep(index: object, is_eco: bool) -> list[tuple[int, float, float]]:
        out = []
        for ef in (16, 32, 64, 128, 256, 512):
            cfg = AnnConfig(
                nprobe=16 if is_eco else 1, ef_centroid=64, ef_list=ef, k=10, per_cluster_k=32
            )
            t0 = time.perf_counter()
            ids, dists = _run(index, queries[:nq], cfg)
            ms = (time.perf_counter() - t0) * 1000 / nq
            se, _ = recall_at_k_batch(ids, dists, gi[:nq], gd[:nq], k=10)
            out.append((ef, float(se.mean()), ms))
        return out

    eco_s, flat_s = sweep(eco, True), sweep(flat, False)

    def ef_for(sweep_rows: list[tuple[int, float, float]]) -> tuple[int | None, float]:
        for ef, rec, ms in sweep_rows:
            if rec >= target:
                return ef, ms
        return None, float("nan")

    eco_ef, eco_ms = ef_for(eco_s)
    flat_ef, flat_ms = ef_for(flat_s)
    ok = eco_ef is not None and flat_ef is not None and eco_ef <= flat_ef

    lines = [f"{'ef':>5} {'ecovector':>20} {'flat_hnsw':>20}"]
    for (ef, er, ems), (_, fr, fms) in zip(eco_s, flat_s, strict=True):
        lines.append(f"{ef:>5} {er:>10.4f} {ems:>7.3f}ms {fr:>10.4f} {fms:>7.3f}ms")
    lines.append("")
    lines.append(
        f"EF-EFFICIENCY (the claim measurable here): recall>={target:.2f} reached at "
        f"ef_list={eco_ef} (ecovector) vs {flat_ef} (flat_hnsw)"
    )
    lines.append(f"WALL-CLOCK (a separate claim): {eco_ms:.3f} ms vs {flat_ms:.3f} ms per query")
    lines.append(compare_note())

    report(f"4. ef-efficiency vs flat HNSW ({dataset})", ok, NL.join(lines))


def check5_resident_footprint(dataset: str) -> None:
    """Anon RSS bound, and the same number against MobileRAG Figure 6."""
    ip = index_root(dataset, "ecovector")
    if ip is None:
        report(f"5. resident footprint ({dataset})", False, "no ecovector index built")
        return
    base_bytes = (resolve_paths().processed / dataset / "base.npy").stat().st_size
    proc = psutil.Process(os.getpid())

    gc.collect()
    before = _anon_mb(proc.memory_info())
    idx = EcoVectorIndex.open(ip.root)
    _, queries, _, _ = load_all(dataset)
    cfg = AnnConfig(nprobe=16, ef_centroid=64, ef_list=128, k=10, per_cluster_k=32)
    peak = before
    for qi in range(min(200, queries.shape[0])):
        r = idx.search(queries[qi], cfg)
        peak = max(peak, r.anon_rss_mb)
    resident = peak - before

    budget = 0.20 * base_bytes / 2**20
    paper = MOBILERAG_FIG6_MB.get(dataset)
    lines = [
        f"base array {base_bytes / 2**20:.0f} MB; 20% budget {budget:.1f} MB",
        f"measured anonymous RSS during search: {resident:.2f} MB "
        f"({resident / (base_bytes / 2**20) * 100:.2f}% of the base footprint)",
    ]
    if paper is not None:
        lines.append(
            f"MobileRAG Figure 6 reports {paper:.2f} MB for EcoVector on this dataset; "
            f"ours is {resident:.2f} MB ({resident / paper:.2f}x)"
        )
    report(f"5. resident footprint ({dataset})", resident <= budget, NL.join(lines))


def check6_cluster_unload_releases_memory(dataset: str = "sift1m") -> None:
    """Memory must not grow with the number of queries served."""
    ip = index_root(dataset, "ecovector")
    if ip is None:
        report("6. cluster unload releases memory", False, "no ecovector index built")
        return
    idx = EcoVectorIndex.open(ip.root)
    _, queries, _, _ = load_all(dataset)
    cfg = AnnConfig(nprobe=16, ef_centroid=64, ef_list=128, k=10, per_cluster_k=32)
    proc = psutil.Process(os.getpid())

    for qi in range(20):  # warm up allocator
        idx.search(queries[qi], cfg)
    gc.collect()
    start = proc.memory_info().rss / 2**20
    for qi in range(20, 220):
        idx.search(queries[qi % queries.shape[0]], cfg)
    gc.collect()
    end = proc.memory_info().rss / 2**20

    growth = end - start
    report(
        "6. cluster load/unload releases memory",
        growth < 50.0,
        f"RSS {start:.1f} MB -> {end:.1f} MB over 200 queries ({growth:+.1f} MB){NL}"
        f"a leak would grow with query count; threshold is 50 MB of allocator noise",
    )


def check7_interleaved_configs(dataset: str = "sift1m") -> None:
    """No state leaks between queries, and no id is ever returned twice.

    The duplicate-id assertion is a free canary on exactly the code path most
    likely to break: a vector reachable from two cluster files means the id map
    or the assignment is broken, and every number from that index is worthless.
    """
    ip = index_root(dataset, "ecovector")
    if ip is None:
        report("7. interleaved configs are independent", False, "no index built")
        return
    idx = EcoVectorIndex.open(ip.root)
    _, queries, gi, gd = load_all(dataset)
    nq = 200
    cheap = AnnConfig(nprobe=2, ef_centroid=64, ef_list=16, k=10, per_cluster_k=16)
    rich = AnnConfig(nprobe=32, ef_centroid=128, ef_list=512, k=10, per_cluster_k=64)

    solo = {}
    for label, cfg in (("cheap", cheap), ("rich", rich)):
        ids, dists = _run(idx, queries[:nq], cfg)
        se, _ = recall_at_k_batch(ids, dists, gi[:nq], gd[:nq], k=10)
        solo[label] = float(se.mean())

    mixed: dict[str, list[float]] = {"cheap": [], "rich": []}
    dup_violations = 0
    for qi in range(nq):
        label, cfg = ("cheap", cheap) if qi % 2 else ("rich", rich)
        r = idx.search(queries[qi], cfg)
        if np.unique(r.ids).size != r.ids.size:
            dup_violations += 1
        se, _ = recall_at_k_batch(
            r.ids[None, :], r.dists[None, :], gi[qi : qi + 1], gd[qi : qi + 1], k=10
        )
        mixed[label].append(float(se[0]))

    deltas = {k: abs(float(np.mean(v)) - solo[k]) for k, v in mixed.items()}
    ok = dup_violations == 0 and all(d < 0.02 for d in deltas.values())
    report(
        "7. interleaved configs are independent",
        ok,
        f"cheap: alone {solo['cheap']:.4f} vs interleaved {np.mean(mixed['cheap']):.4f} "
        f"(delta {deltas['cheap']:.4f}){NL}"
        f"rich:  alone {solo['rich']:.4f} vs interleaved {np.mean(mixed['rich']):.4f} "
        f"(delta {deltas['rich']:.4f}){NL}"
        f"duplicate-id violations: {dup_violations}/{nq} (any at all invalidates the index)",
    )


def check8_probe_cost(dataset: str) -> None:
    """t_centroid / t_total, against the ruled verdict table."""
    ip = index_root(dataset, "ecovector")
    if ip is None:
        report(f"8. probe cost ({dataset})", False, "no ecovector index built")
        return
    idx = EcoVectorIndex.open(ip.root)
    _, queries, _, _ = load_all(dataset)
    cfg = AnnConfig(nprobe=8, ef_centroid=64, ef_list=128, k=10, per_cluster_k=32)

    fractions = []
    for qi in range(min(300, queries.shape[0])):
        r = idx.search(queries[qi], cfg)
        if r.t_total > 0:
            fractions.append(r.t_centroid / r.t_total)
    f = np.asarray(fractions)
    p50, p95 = float(np.percentile(f, 50)), float(np.percentile(f, 95))

    if p50 < PROBE_HEALTHY:
        verdict = "HEALTHY (<0.15): the two-phase split is cheap, ef_list stays the primary lever"
    elif p50 <= PROBE_WORKABLE:
        verdict = "WORKABLE (0.15-0.35): usable, but probe cost is not negligible"
    else:
        verdict = (
            "PREMISE WEAKENED (>0.35): ef_centroid becomes the primary lever and the "
            "Phase 6 profile table must be rebuilt around it"
        )
    report(
        f"8. probe cost ({dataset}, BALANCED profile)",
        p50 <= PROBE_WORKABLE,
        f"t_centroid/t_total: p50 {p50:.4f}, p95 {p95:.4f}{NL}{verdict}{NL}"
        f"WARM CACHE ONLY -- this platform cannot evict the page cache, so a cluster "
        f"'load' here is a memcpy, not disk I/O. Phase 3.5 gives the cold-cache number "
        f"and it may move this verdict.",
    )


def check9_memory_model(dataset: str) -> None:
    """Predicted resident footprint against the measured one, term by term."""
    ip = index_root(dataset, "ecovector")
    if ip is None:
        report(f"9. memory model ({dataset})", False, "no ecovector index built")
        return
    meta = IndexMeta.from_json(ip.meta)
    model = memory_model_mb(meta.nc, meta.dim, meta.n_base, meta.largest_cluster, meta.m)

    proc = psutil.Process(os.getpid())
    gc.collect()
    before = proc.memory_info().rss / 2**20
    idx = EcoVectorIndex.open(ip.root)
    _, queries, _, _ = load_all(dataset)
    cfg = AnnConfig(nprobe=8, ef_centroid=64, ef_list=128, k=10, per_cluster_k=32)
    for qi in range(50):
        idx.search(queries[qi], cfg)
    measured = proc.memory_info().rss / 2**20 - before

    lines = [f"{term:>16} {mb:9.3f} MB" for term, mb in model.items()]
    lines.append(f"{'MEASURED RSS delta':>16} {measured:9.3f} MB")
    ratio = measured / model["total"] if model["total"] else float("inf")
    lines.append(f"measured / predicted = {ratio:.2f}x")
    report(f"9. memory model ({dataset})", 0.2 <= ratio <= 5.0, NL.join(lines))


def check10_build_cost() -> None:
    """Build cost for every index built, for the feasibility claim."""
    rows = []
    ok = True
    for dataset in ("sift1m", "nytimes", "squad"):
        for variant in ("ecovector", "flat_hnsw", "ivf_disk"):
            ip = index_root(dataset, variant)
            if ip is None:
                continue
            m = IndexMeta.from_json(ip.meta)
            rows.append(
                f"{dataset:8s} {variant:10s} {m.build_seconds:8.1f}s  "
                f"peak build RSS {m.build_peak_rss_mb:7.0f} MB  {m.build_workers:2d} workers  "
                f"{m.bytes_on_disk / 2**20:8.1f} MB on disk  ({m.kmeans_algorithm})"
            )
    if not rows:
        ok = False
        rows = ["no indexes built"]
    report("10. build cost (measurement)", ok, NL.join(rows))


# ---------------------------------------------------------------------------
def _run(index: object, queries: np.ndarray, cfg: AnnConfig) -> tuple[np.ndarray, np.ndarray]:
    """Search every query, returning padded id and distance matrices."""
    ids = np.full((queries.shape[0], cfg.k), -1, dtype=np.int64)
    dists = np.full((queries.shape[0], cfg.k), np.inf, dtype=np.float64)
    for qi in range(queries.shape[0]):
        r = index.search(queries[qi], cfg)  # type: ignore[attr-defined]
        n = min(r.ids.size, cfg.k)
        ids[qi, :n], dists[qi, :n] = r.ids[:n], r.dists[:n]
    return ids, dists


def summary() -> None:
    print("=" * 78)
    for name, passed, _ in RESULTS:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["sift1m", "nytimes"])
    ap.add_argument("--skip", nargs="*", default=[])
    args = ap.parse_args()

    checks = {
        "1a": lambda: check1a_exhaustive_equivalence(),
        "1b": lambda: [check1b_high_effort_recall(d) for d in args.datasets],
        "2": lambda: check2_ef_monotonicity(),
        "4": lambda: [check4_ef_efficiency_and_wallclock(d) for d in args.datasets],
        "5": lambda: [check5_resident_footprint(d) for d in args.datasets],
        "6": lambda: check6_cluster_unload_releases_memory(),
        "7": lambda: check7_interleaved_configs(),
        "8": lambda: [check8_probe_cost(d) for d in args.datasets],
        "9": lambda: [check9_memory_model(d) for d in args.datasets],
        "10": lambda: check10_build_cost(),
    }
    for key, fn in checks.items():
        if key in args.skip:
            print(f"[SKIP] check {key}", flush=True)
            continue
        fn()
    summary()
    return 0 if all(p for _, p, _ in RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())

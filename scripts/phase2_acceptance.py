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
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from smartann.bench.metrics import recall_at_k_batch
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


# Anchored to this file, not to the working directory. A relative path turns
# "run from the wrong directory" into every check reporting FAIL with "no index
# built" -- a silent misreport, which is exactly the failure mode the
# no-fabrication rules exist to prevent.
CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "index"


def index_root(dataset: str, variant: str) -> IndexPaths | None:
    """The index built from configs/index/<dataset>_<variant>.yaml, or None.

    Derived from the config rather than discovered on disk. Picking the
    lexicographically largest hash, or even the newest mtime, means the
    acceptance numbers silently describe whichever artifact happened to sort or
    land last -- which is fine until the first --force rebuild and wrong
    thereafter, with no symptom. The config file is what the build was told to
    do, so it is what the check should ask about.

    Returns None only when the index is genuinely not built. A missing config
    *directory* is a broken checkout rather than a missing index, and raises.
    """
    if not CONFIG_DIR.is_dir():
        raise FileNotFoundError(
            f"{CONFIG_DIR} does not exist. Without it every check would report "
            "'no index built' and FAIL, which is a misreport rather than a result."
        )
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

    # A scratch index, not a deliverable: built for this check and removed
    # afterwards, so it never sits in the index root looking like a real
    # artifact that something might later resolve or report on.
    tmp = Path(tempfile.mkdtemp(prefix=f"smartann_check1a_{dataset}_"))
    try:
        out = IndexPaths(tmp / "idx")
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
    finally:
        # Windows will not delete a file while it is memory-mapped, and flat
        # clusters are mapped (ADR-0029). Dropping the index and collecting
        # before the rmtree releases the maps; ignore_errors would have hidden
        # the failure and leaked one temp tree per acceptance run.
        idx = None
        gc.collect()
        shutil.rmtree(tmp)

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


def check1b_recall_against_ceiling(dataset: str, ratio_threshold: float = 0.98) -> None:
    """Does the in-cluster search recover what the partition makes available?

    Re-posed. The original asked for absolute recall >= 0.95, which fused two
    different questions: *is the index built correctly* and *does this dataset
    partition well*. Only the first is about our work, check 1a already answers
    it directly, and an absolute recall number cannot separate a bad in-cluster
    search from a corpus that does not localise.

    The ratio to the cluster ceiling -- the best achievable at this nprobe with
    a perfect in-cluster search -- isolates the part we control. It is also a
    STRICTER test: a regression in the in-cluster search drops the ratio and
    fails on either dataset, where the absolute threshold could not have
    detected one on NYTimes because it was already failing for an unrelated
    reason.

    The absolute recall and the coverage-to-0.95 figure are still printed, as a
    dataset property carrying no verdict.
    """
    ip = index_root(dataset, "ecovector")
    if ip is None:
        report(f"1b. recall against the cluster ceiling ({dataset})", False, "no index built")
        return
    idx = EcoVectorIndex.open(ip.root)
    _, queries, gi, gd = load_all(dataset)
    nq = min(1000, queries.shape[0])
    nprobe = 64
    cfg = AnnConfig(nprobe=nprobe, ef_centroid=128, ef_list=512, k=10, per_cluster_k=64)

    ids, dists = _run(idx, queries[:nq], cfg)
    se, ov = recall_at_k_batch(ids, dists, gi[:nq], gd[:nq], k=10)

    # Ceiling: fraction of each query's true top-10 lying in the probed clusters.
    assign = np.asarray(idx.assignments)
    gt_clusters = assign[np.asarray(gi[:nq, :10])]
    sizes = idx.sizes.astype(np.float64)
    hits, covered = [], []
    for qi in range(nq):
        probed = idx.probe(queries[qi], cfg).cluster_ids
        hits.append(float(np.isin(gt_clusters[qi], probed).mean()))
        covered.append(float(sizes[probed].sum()))
    ceiling = float(np.mean(hits))
    coverage = float(np.mean(covered)) / idx.meta.n_base
    ratio = float(ov.mean()) / ceiling if ceiling > 0 else 0.0

    report(
        f"1b. in-cluster search recovers the ceiling ({dataset}, nprobe={nprobe})",
        ratio >= ratio_threshold,
        f"index-overlap {ov.mean():.4f} / ceiling {ceiling:.4f} = {ratio:.4f} "
        f"(threshold {ratio_threshold:.2f}){NL}"
        f"in-cluster loss {ceiling - float(ov.mean()):.4f}, cluster-miss loss "
        f"{1.0 - ceiling:.4f}{NL}"
        f"DATASET PROPERTY, no verdict: absolute score-equality recall "
        f"{se.mean():.4f}; this nprobe examines {coverage * 100:.1f}% of the corpus. "
        f"NYTimes needs ~40% coverage for a 0.95 ceiling against SIFT's under 2% "
        f"(ADR-0025 amendment) -- a 20x difference in how the data partitions.{NL}"
        f"ORIGINAL POSING (absolute recall >= 0.95): "
        f"{'PASS' if float(se.mean()) >= 0.95 else 'FAIL'} at {se.mean():.4f}",
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


def _sweep_cost_recall(
    indexes: dict[str, Any],
    queries: np.ndarray,
    gi: np.ndarray,
    gd: np.ndarray,
    grid: list[AnnConfig],
    warmup: int = 1,
    repeats: int = 5,
) -> dict[str, list[tuple[str, float, float, float, float]]]:
    """Per variant: (config, mean distance computations, recall, median ms, IQR ms).

    Timed per query with warmup and repeats, reporting median and inter-quartile
    range. The single-shot version reported 27.076 ms and 8.589 ms for two
    configs doing provably identical work (10,724 modelled comparisons each) --
    a 3x spread at zero work difference, which is first-touch page-cache effects
    being read as a performance result. A wall-clock number without a dispersion
    estimate beside it should not be quoted, so this returns one.
    """
    n_timed = min(50, queries.shape[0])
    labels = list(indexes)
    out: dict[str, list[tuple[str, float, float, float, float]]] = {k: [] for k in labels}

    for cfg in grid:
        # Recall pass: every query, once. Recall needs breadth, not repetition,
        # and is deterministic so it is unaffected by ordering.
        rec, dist = {}, {}
        for label in labels:
            total_dist = 0
            ids = np.full((queries.shape[0], cfg.k), -1, dtype=np.int64)
            dists = np.full((queries.shape[0], cfg.k), np.inf, dtype=np.float64)
            for qi in range(queries.shape[0]):
                r = indexes[label].search(queries[qi], cfg)
                n = min(r.ids.size, cfg.k)
                ids[qi, :n], dists[qi, :n] = r.ids[:n], r.dists[:n]
                total_dist += r.n_distance_computations
            se, _ = recall_at_k_batch(ids, dists, gi, gd, k=cfg.k)
            rec[label] = float(se.mean())
            dist[label] = total_dist / queries.shape[0]

        # Timing pass, INTERLEAVED across variants and alternating on every
        # repeat. A blocked sweep -- all of variant A, then all of variant B --
        # measures the machine's drift as much as the code: an earlier blocked
        # run put a 4x discontinuity between two configs that differed only in
        # position. Interleaving makes drift hit both sides equally.
        for label in labels:
            for _ in range(warmup):
                for qi in range(min(5, n_timed)):
                    indexes[label].search(queries[qi], cfg)
        per: dict[str, list[float]] = {k: [] for k in labels}
        for qi in range(n_timed):
            best = dict.fromkeys(labels, float("inf"))
            for _ in range(repeats):
                for label in labels:
                    t0 = time.perf_counter()
                    indexes[label].search(queries[qi], cfg)
                    best[label] = min(best[label], (time.perf_counter() - t0) * 1000.0)
            for label in labels:
                per[label].append(best[label])

        for label in labels:
            t = np.asarray(per[label])
            out[label].append(
                (
                    f"nprobe={cfg.nprobe},ef={cfg.ef_list}",
                    dist[label],
                    rec[label],
                    float(np.median(t)),
                    float(np.percentile(t, 75) - np.percentile(t, 25)),
                )
            )
    return {k: sorted(v, key=lambda row: row[1]) for k, v in out.items()}


def check4_ef_efficiency_and_wallclock(dataset: str) -> None:
    """Recall against WORK, plus wall-clock as a separate claim.

    Not against ``ef_list``. Two reasons that axis is wrong here:

    1. ``finish()`` calls ``set_ef(max(ef_list, per_cluster_k))``, so with
       ``per_cluster_k=32`` every ``ef_list <= 32`` is the same configuration.
       A sweep from 16 upward silently duplicates its first points for
       EcoVector while flat's ``ef=16`` really is 16, and any "reaches the
       target at a lower ef" verdict is then partly an artifact of the flooring.
    2. Even without that, ``ef_list`` is not work-equivalent across the
       variants: EcoVector at ``nprobe=16`` examines up to 16 x per_cluster_k
       candidates under the same label that gives flat roughly ``ef``.

    So the x-axis is ``n_distance_computations``. **That number is itself partly
    a model**: flat clusters count their comparisons exactly, graph traversal is
    estimated as ``ef * M``. This therefore compares two cost models on equal
    terms rather than measuring real distance evaluations, and the conclusion is
    only as good as that shared model. Wall-clock, reported separately below, is
    the measured quantity.
    """
    eco_p = index_root(dataset, "ecovector")
    flat_p = index_root(dataset, "flat_hnsw")
    ivf_p = index_root(dataset, "ivf_disk")
    if eco_p is None or flat_p is None or ivf_p is None:
        report(f"4. work-efficiency vs ivf_disk ({dataset})", False, "all three indexes required")
        return
    eco = EcoVectorIndex.open(eco_p.root)
    flat = FlatHnswIndex.open(flat_p.root)
    ivf = EcoVectorIndex.open(ivf_p.root)
    _, queries, gi, gd = load_all(dataset)
    nq = min(300, queries.shape[0])
    q, gi, gd = queries[:nq], gi[:nq], gd[:nq]
    target = 0.90
    pck = 32

    # Every ef_list >= per_cluster_k, so no two EcoVector points collapse onto
    # the same effective configuration.
    eco_grid = [
        AnnConfig(nprobe=npb, ef_centroid=64, ef_list=ef, k=10, per_cluster_k=pck)
        for npb, ef in ((2, 32), (4, 32), (8, 64), (16, 64), (16, 128), (32, 256), (64, 512))
    ]
    flat_grid = [
        AnnConfig(nprobe=1, ef_centroid=64, ef_list=ef, k=10, per_cluster_k=pck)
        for ef in (16, 32, 64, 128, 256, 512, 1024)
    ]
    swept = _sweep_cost_recall({"ecovector": eco, "ivf_disk": ivf}, q, gi, gd, eco_grid)
    eco_s, ivf_s = swept["ecovector"], swept["ivf_disk"]
    flat_s = _sweep_cost_recall({"flat_hnsw": flat}, q, gi, gd, flat_grid)["flat_hnsw"]

    def cost_for(rows: list[tuple[str, float, float, float, float]]) -> tuple[float, float, str]:
        for label, ndist, rec, ms, _iqr in rows:
            if rec >= target:
                return ndist, ms, label
        return float("nan"), float("nan"), "not reached"

    eco_cost, eco_ms, eco_lbl = cost_for(eco_s)
    ivf_cost, ivf_ms, ivf_lbl = cost_for(ivf_s)
    flat_cost, flat_ms, flat_lbl = cost_for(flat_s)

    # The verdict is EcoVector against ivf_disk, which is the same partition,
    # the same files and the same load path with brute force instead of graphs.
    # That isolates exactly what the per-cluster graphs buy.
    ok = np.isfinite(eco_cost) and np.isfinite(ivf_cost) and eco_cost <= ivf_cost

    def table(name: str, rows: list[tuple[str, float, float, float, float]]) -> list[str]:
        out = [
            f"{name}:",
            f"{'config':>22} {'dist_comps':>11} {'recall':>8} {'ms median':>10} {'IQR':>8}",
        ]
        out += [
            f"{lbl:>22} {nd:>11,.0f} {rec:>8.4f} {ms:>10.3f} {iqr:>8.3f}"
            for lbl, nd, rec, ms, iqr in rows
        ]
        return [*out, ""]

    lines = table("ecovector", eco_s) + table("ivf_disk (same partition, no graphs)", ivf_s)
    lines += [
        f"VERDICT -- ecovector vs ivf_disk: recall>={target:.2f} at {eco_cost:,.0f} modelled "
        f"distance computations ({eco_lbl}) vs {ivf_cost:,.0f} ({ivf_lbl})",
        f"  wall-clock at that point (median of {5} repeats after warmup): "
        f"{eco_ms:.3f} ms vs {ivf_ms:.3f} ms per query",
        "  Same partition, same files, same load path -- the only difference is per-cluster",
        "  graphs versus brute force, so this is a claim the index wins or loses on its merits.",
        "  n_distance_computations is exact for flat clusters and a CAPPED MODEL for graph",
        "  traversal (min(ef*M, cluster size)), so the two sides share one cost model.",
        "",
    ]
    lines += table("flat_hnsw (REPORTED, NO VERDICT)", flat_s)
    lines += [
        f"flat_hnsw reaches recall>={target:.2f} at {flat_cost:,.0f} ({flat_lbl}), "
        f"{flat_ms:.3f} ms per query.",
        "  Deliberately carries no verdict. Asking an IVF index to reach a recall target in",
        "  fewer distance computations than one fully resident graph asks it to win the exact",
        "  thing it trades away. What it buys instead is the resident footprint measured in",
        "  check 5 -- read the two together, never this table alone.",
        compare_note(),
    ]
    report(f"4. work-efficiency vs ivf_disk ({dataset})", ok, NL.join(lines))


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

    fractions, ex_load, load_share = [], [], []
    for qi in range(min(300, queries.shape[0])):
        r = idx.search(queries[qi], cfg)
        if r.t_total > 0:
            fractions.append(r.t_centroid / r.t_total)
            load_share.append(r.t_cluster_load / r.t_total)
            # The same fraction with cluster LOAD removed from the denominator.
            # 83% of t_total is hnswlib.load_index on a warm page cache, i.e.
            # graph reconstruction rather than I/O, so leaving it in makes the
            # probe look cheap partly as an artifact of something else being
            # expensive. Both numbers are reported; the verdict uses the honest
            # one, which is this.
            denom = r.t_total - r.t_cluster_load
            if denom > 0:
                ex_load.append(r.t_centroid / denom)
    f = np.asarray(fractions)
    g = np.asarray(ex_load)
    x50, x95 = float(np.percentile(g, 50)), float(np.percentile(g, 95))
    mean_load = float(np.mean(load_share)) * 100

    # Verdict on the load-excluded figure, which is the one that is not
    # flattered by an unrelated cost sitting in the denominator.
    p50 = x50
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
        f"t_centroid / t_total                    : p50 {float(np.percentile(f, 50)):.4f}, "
        f"p95 {float(np.percentile(f, 95)):.4f}{NL}"
        f"t_centroid / (t_total - t_cluster_load) : p50 {x50:.4f}, p95 {x95:.4f}  "
        f"<-- verdict uses this{NL}"
        f"cluster load is {mean_load:.1f}% of t_total, which inflates the first figure's "
        f"denominator{NL}{verdict}{NL}"
        f"WARM CACHE ONLY -- this platform cannot evict the page cache, so a cluster 'load' "
        f"here is graph RECONSTRUCTION (hnswlib.load_index), not disk I/O. Phase 3.5 gives "
        f"the cold-cache number and it may move this verdict.",
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
        "1b": lambda: [check1b_recall_against_ceiling(d) for d in args.datasets],
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

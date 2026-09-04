# Phase 2 report — EcoVector index

Run: `python scripts/phase2_acceptance.py --datasets sift1m nytimes`
Result: **13 PASS, 2 FAIL**. Both failures are the same root cause.

## Verdicts

| check | sift1m | nytimes |
|---|---|---|
| 1a exhaustive equivalence | PASS (worst error 0.000e+00) | n/a |
| 1b high-effort recall | PASS 0.9881 | **FAIL 0.8751** (threshold 0.95) |
| 2 effort changes recall | PASS 0.3033 → 0.9537, monotone | n/a |
| 4 work-efficiency vs ivf_disk | PASS (tie) | **FAIL** (0.90 not reached) |
| 5 resident footprint | PASS 13.27 MB | PASS 5.30 MB |
| 6 load/unload releases memory | PASS +0.1 MB / 200 queries | — |
| 7 interleaved configs | PASS, 0 duplicate-id violations | — |
| 8 probe cost | PASS p50 0.0237 | PASS p50 0.0235 |
| 9 memory model | PASS 1.50x | PASS 1.33x |
| 10 build cost | PASS (measurement) | PASS (measurement) |

## The two failures share one cause

Both are cluster-miss on NYTimes. ADR-0025 decomposes it: at `nprobe=64` the
ceiling — the best achievable with a *perfect* in-cluster search — is 0.8709,
and measured index-overlap recall is 0.8603. In-cluster search loses 1.1 points
of a 13.9-point shortfall. `ef_list` and `M` cannot recover it.

Measured against corpus coverage rather than `nprobe` (ADR-0025 amendment),
**NYTimes needs ~40% of the corpus examined to reach a 0.95 ceiling** — against
under 2% for SIFT. That 20x difference is a property of 256-dimensional angular
data under IVF partitioning, and no setting of `nc`, `M` or `ef_list` repairs
it. A finer partition is strictly better here on both recall-per-work and
memory; the original "lower nc helps" reading was an artifact of comparing at
equal `nprobe`, which is 4x different work across `nc`.

## Thresholds that did not discriminate, stated as such

Two bands were pre-registered too loose to carry information. Recording this
rather than tightening them now, because tightening a threshold after seeing the
data is fitting the threshold to the result.

- **Check 5's 20% budget.** 97.7 MB on SIFT against a measured 13.27 MB — 7x
  headroom. It would pass almost any correct implementation. The informative
  numbers are the MobileRAG Figure 6 ratios, **1.37x (SIFT)** and **1.50x
  (NYTimes)**, which carry no verdict and are the real evidence the index is
  built correctly.
- **Check 9's 0.2–5.0x band.** Spans a factor of 25; a 6.87 MB prediction would
  "pass" against anything from 1.4 MB to 34 MB. The informative numbers are the
  model ratios, **1.50x** and **1.33x** — the Section 3.4 model predicts within
  50% — and they likewise carry no verdict.

## Findings that outrank the failures

1. **The per-cluster graphs buy nothing at these cluster sizes** (ADR-0024
   + amendment). EcoVector and ivf_disk reach identical recall at identical
   modelled work, and ivf_disk is **4.1x faster** at the matched operating
   point (15.470 ms vs 62.901 ms, medians of 5 repeats after warmup), stable at
   4.2-4.5x across the sweep. `ef * M >= 1024` exceeds the mean cluster
   (500 / 539), so every per-cluster graph search is exhaustive anyway.
2. **83% of per-query time is `hnswlib.load_index`** on a warm cache, i.e.
   graph reconstruction rather than I/O (ADR-0026). Search is 4%.
3. **`ef_centroid` is free and therefore not a control lever** (ADR-0025). The
   centroid graph matches brute force exactly at every nprobe on both datasets.
4. **EcoVector's build is 5.5x lighter than flat HNSW's** in peak RSS
   (177 MB vs 977 MB on SIFT), which is the on-device feasibility claim.
5. **Zero empty clusters and zero duplicate centroids** on both datasets,
   including NYTimes at 14.078% duplicate rows — the k-means risk that ADR-0014
   flagged did not materialise.

## Open decisions

- NYTimes 1b: raise `nprobe`, lower `nc` (measured table in ADR-0025), or revise
  the threshold with a recorded reason.
- Whether to keep per-cluster graphs at all, or lower `nc` until they earn their
  place. Same axis; should be decided together.
- Whether to cache reconstructed cluster graphs, which trades against the
  bounded footprint checks 5 and 6 defend.

  **This one must be decided before Phase 3 fits its cost model, not after.**
  A cache turns the dominant cost term from `n_cluster_loads` into
  `n_cache_misses`. Those are different random variables with different
  distributions -- misses depend on query locality and cache size, loads do not
  -- so a model fitted on one is not merely imprecise for the other, it is
  invalid. Fitting first and caching later would silently invalidate every
  latency and energy prediction Phase 3 produces, and the failure would surface
  as unexplained controller underperformance in Phase 6.


## Consequence for Phase 6: the action space is smaller than assumed

Phase 6 should be told this now rather than discovering it while tuning a
controller.

SMART-ANN's premise is a controller that adapts retrieval effort per query. The
`AnnConfig` knobs are `nprobe`, `ef_centroid`, `ef_list` and `per_cluster_k`.
Phase 2 measured two of them to be inert:

- **`ef_centroid` is free** (ADR-0025). The centroid graph matches brute force
  exactly at every `nprobe` on both datasets, gap 0.0000. It has no error to
  trade against, so raising or lowering it buys and costs nothing measurable.
- **`ef_list` is inert once the in-cluster search is exhaustive** (ADR-0024).
  With `ef * M >= 1024` against a mean cluster of ~500, every per-cluster search
  already examines everything. The check 4 sweep shows it directly: `nprobe=16,
  ef=64` and `nprobe=16, ef=128` give 0.8773 and 0.8780 -- a 2x change in the
  knob moving recall by 0.0007.

That leaves **`nprobe` as the effort knob the controller will actually turn**,
and check 2 confirms it spans a wide range on its own (recall 0.3033 to 0.9537,
2.1 to 64.3 ms). If the per-cluster graphs are dropped, the action space is
`nprobe` alone -- a one-dimensional controller, not the multi-dimensional one
Phase 6's design assumes.

This is not necessarily bad. A one-dimensional action space makes the bandit
easier to learn, the safety shield easier to reason about, and the Pareto front
easier to characterise. But it changes what C1 ("runtime-adaptive ANN
parameters", plural) can honestly claim, and it should be settled before Phase 6
builds a profile table over dimensions that do not move the outcome.

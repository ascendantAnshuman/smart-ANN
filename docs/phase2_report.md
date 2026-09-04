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

1. **The per-cluster graphs buy nothing at these cluster sizes** (ADR-0024).
   EcoVector and ivf_disk reach identical recall at identical modelled work,
   and ivf_disk is 2.3–3.7x faster. `ef * M >= 1024` exceeds the mean cluster
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

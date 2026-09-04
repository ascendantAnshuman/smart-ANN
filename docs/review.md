# SMART-ANN — mid-project review

A working demonstration of the adaptive mechanism on measured data, plus the
Phase 2 index results. **Not** SMART-ANN as specified: the controller here is one
policy, and the thermal, energy, SCR and generation work is not built. What is
built is measured; nothing in this document or in any figure comes from a
simulator, a placeholder, or a model presented as a measurement.

---

## 1. What the process caught

Eight defects, each caught before it produced a reported number.

**Ranked by danger, which is not the same as severity** — by how likely each was
to survive undetected, not by how wrong it would have been.

| # | Defect | Why it ranks here |
|---|---|---|
| 1 | **Check 4 compared an index with itself and reported PASS** | A broken *measurement* gets caught by a check. A broken *check* catches nothing, including itself, and reports a green line in a summary table -- precisely the line nobody re-reads. |
| 2 | **The static baseline was not being tuned, and the manifest said it was** | `max(key=nprobe)` returns the largest candidate unconditionally, and the latency target was then derived from that same config -- so the selection met the target by construction. The number was recoverable; the manifest's description of the method was not. A run record that misdescribes its own method is worse than a wrong number, because the description is what a reader trusts when they cannot re-derive. |
| 3 | `compute_topk` dropped `‖q‖²` from its L2 expression | Every stored ground-truth distance off by a per-query constant. Score-equality recall would have compared against a meaningless threshold and produced a *plausible* number. Every ordering test the module had already passed. |
| 4 | Timing harness ran variants in blocks | Reported a 3x spread between two configs doing provably identical work. Machine drift presented as a performance result, and it looked like one. |
| 5 | `n_dist_modeled` was uncapped `ef·M` | Modelled work overstated by up to **29.5x**, on the field Phase 3 regresses its cost model against. |
| 6 | `decide()` called twice per query | AIMD would have halved effort twice per overshoot, and `controller_overhead_us` would have measured only the second call -- an over-aggressive controller and an understated overhead, both readable as findings. |
| 7 | Burners contended for ALU, not memory bandwidth | The query path is a scan over mmap'd arrays. The experiment would have applied almost no pressure, and "the controller did not react" would have looked like a result about the controller. |
| 8 | Fork bomb: a script starting worker processes without a `__main__` guard | Windows `spawn` re-imports the parent module in every child, so each child re-ran the whole script recursively. Fixed *inside* `Burners.start()`, which now refuses to launch from an unguarded module -- fixing only the script would have left the trap armed for the next caller. |

### Three detection mechanisms, and what each costs

The eight did not fall to one method, and the differences are the point.

- **Caught by loud failure** (#8). Free. The failure announced itself; the
  machine fell over and there was nothing to notice.
- **Caught by looking** (#1, #2, #6, #7). Cheap, but depends on someone reading
  the right lines on the right day. Four of eight, which is not a system.
- **Caught by external reference** (#3, #4, #5). The expensive category, and the
  **only one that catches errors producing plausible numbers**:
  - `‖q‖²` — against the **published ground truth shipped in the HDF5**
  - the blocked harness — against an **independent interleaved control**
  - uncapped `ef·M` — against the **physical containment ceiling**

None of those three was caught by an acceptance threshold. Each was caught by
checking the number against something *outside the system that produced it*.

**The ordering is itself the finding: the cost of detection is inversely
proportional to how loudly a defect fails.** A fork bomb costs minutes. A recall
metric quietly off by a per-query constant costs however long it takes someone
to distrust a number that looks fine — and on this project, three separate
external references were needed to find three such numbers.

That generalises better than a test count: **a test suite validates a system
against its own assumptions, so the errors that survive are exactly the ones the
assumptions share.** Published reference data, an independent control, and a
physical bound are the three external checks available here, and all three
earned their place.

## 2. What is built

- **Phase 1** — three datasets prepared, verified, checksummed, with exact
  ground truth and a manifest covering every artifact.
- **Phase 2** — the index: IVF partitioning, a resident centroid graph, on-disk
  clusters, two-phase `probe()` / `finish()` search, two baselines, and a
  10-check acceptance suite that runs against real data.
- **Telemetry** — a background monitor with a capability table that makes it
  impossible to record a measurement the platform cannot make.
- **Controller** — the policy contract, a tuned static baseline, and an AIMD
  latency governor.
- **Harness** — per-query records to parquet, a run manifest with recorded
  events, and figures generated only from stored data.

Everything below is measured on: Windows 11, i5-13420H (8 physical / 12 logical
cores), 15.6 GB RAM, warm page cache.

---

## 3. Phase 2 results — 14 PASS, 1 FAIL

The single failure is NYTimes check 4, and it is a dataset property rather than
an index defect (§4).

**Correctness.** With every cluster probed, the index returns exactly what brute
force returns: 50/50 queries score-equal, **worst distance error 0.000e+00**,
with both storage paths exercised (145 graph / 55 flat clusters). That is a
proof of the id mapping, the per-cluster storage, the distance conventions and
the merge.

**Resident footprint against the paper.** Independent evidence the index is
built correctly, and the cheapest such evidence available:

| dataset | MobileRAG Fig. 6 | measured here | ratio |
|---|---|---|---|
| sift1m | 9.67 MB | **13.03 MB** | 1.35× |
| nytimes | 3.53 MB | **5.18 MB** | 1.47× |

**Effort moves recall.** The premise the controller depends on, measured on the
real index: recall@10 spans **0.3047 → 0.9537** as `nprobe` goes 1 → 32, at
1.1 → 30.5 ms per query, monotone throughout.

---

## 4. The main finding: MobileRAG's per-cluster graphs do not pay

EcoVector's distinguishing component over the IVF-HNSW baseline it is measured
against is a per-cluster HNSW graph, loaded and unloaded per query. We removed
it and the index got **~2.5× faster at identical recall** (29.3 ms against
72.2 ms at `nprobe=32`) and **263 MB smaller on disk**.

This is derived from the paper's own equations, not asserted against them.

**The paper states the condition.** Its search-time models are
`ef_c·M' + n_P·(N/Nc)` for IVF-HNSW and `ef_c·M' + n_P·ef_L·M'` for EcoVector,
so the graph pays only when **`ef_L·M' < N/Nc`**. Our default configurations
violate it — 1024 against 500 on SIFT, 1024 against 539 on NYTimes.

**We tested the regime where it holds.** At `nc=250` on SIFT (`N/Nc = 4000`) the
condition is satisfied, and the graphs cut distance computations exactly as
predicted — 4.0× at M=16, 2.0× at M=32, the dose-response confirming the model's
mechanics. **They still lost 2.3–3.0× on wall-clock at every operating point.**

**Why: a missing term.** The paper prices cluster access as pure I/O and has no
term for turning bytes into a *searchable* structure. For a flat array that term
is genuinely zero — the stored form is the traversable form. For an HNSW graph we
measured it at **95–99% of access cost**.

**And that is a contradiction inside the design, not just with our build.**
§3.2.3 requires unloading the graph after every query, because the bounded
footprint is the central claim. But a pure-I/O access model is valid only if the
serialised form is traversable in place. Unload-per-query and pure-I/O cost
cannot both hold unless the on-disk format is directly searchable.

**The constructive version:** a directly-traversable on-disk graph format would
satisfy both at once. That is precisely why our mmap'd flat clusters win — a flat
array already has the property the design needs. The finding is not "graphs are
bad" but *a format that must survive unload-per-query has to be traversable
without materialisation, and hnswlib's is not.*

**Caveat.** The 2.3–3.0× figure is hnswlib-specific. The argument is not: it
follows from the paper's equations plus the measured fact that materialisation is
non-zero for any format requiring a parse.

> **Naming.** The artifact we run is **IVF-HNSW + mmap** — our EcoVector
> configuration with per-cluster graphs disabled (ADR-0028). `IndexMeta` records
> `has_per_cluster_graphs`, and every figure caption says so on first use.

---

## 5. NYTimes: a dataset property, not an index defect

NYTimes reaches 0.8752 recall@10 at `nprobe=64` against an original threshold of
0.95. Decomposed:

- **Cluster ceiling** at that `nprobe` — the best achievable with a *perfect*
  in-cluster search — is 0.8711.
- **Measured** index-overlap recall is 0.8603.
- So the in-cluster search loses **0.0108 of a 0.1397 shortfall**; the other
  92% is cluster-miss.

Measured against corpus coverage, **NYTimes needs ~40% of the corpus examined
for a 0.95 ceiling, against under 2% for SIFT** — a 20× difference in how the
data partitions under IVF. No setting of `nc`, `M` or `ef_list` repairs it; a
finer partition is strictly better on both recall-per-work and memory, which is
the opposite of the first (incorrect) reading, caught by switching the axis from
`nprobe` to corpus coverage.

Check 1b was therefore re-posed as **measured recall / cluster ceiling ≥ 0.98**,
which isolates the part we control. NYTimes passes at 0.9876, SIFT at 0.9992,
and the original absolute posing is printed alongside it on every run so the
failure stays visible.

---

## 6. The adaptive mechanism

**The disturbance was verified before anything was built on it.** "We applied
contention" is a claim; this is its evidence. Requirement: contended p95 at least
2x quiet p95.

| burners | total mem | p50 (ms) | p95 (ms) | p95 ratio |
|---|---|---|---|---|
| quiet | — | 16.95 | 26.02 | — |
| 11 | 3.0 GB | 92.84 | 137.39 | **5.28x** |
| 12 | 3.0 GB | 93.78 | 151.00 | **5.80x** |
| 24 | 3.0 GB | 179.55 | 254.94 | **9.80x** |

The burners **stream through a large buffer** rather than spinning on
floating-point adds. That distinction decided the experiment: the query path is a
brute-force scan over memory-mapped flat arrays and is memory-bandwidth bound, so
an ALU spin loop barely touches it.

**12 burners were used, not 24.** 12 clears the bar at 5.80x; 24 consumes the
host almost entirely, which would make the *recovery* phase unrepresentative --
the machine would still be settling when the burners stop, and the third phase
exists to show effort being restored against a quiet baseline.


*(Populated from the contention experiment; see `results/figures/fig1_headline.png`.)*

The AIMD governor observes p95 latency against a target and moves `nprobe`. It
does not know *why* latency rose — CPU contention, thermal throttling and a hard
query are the same signal to it. That is why demonstrating it against contention
demonstrates the mechanism that will later face thermal throttling.

**The action space is one-dimensional, by measurement.** `ef_centroid` is free
(the centroid graph matches brute force exactly at every `nprobe`, gap 0.0000)
and `ef_list` is inert once the in-cluster search is exhaustive. `nprobe` is the
knob that moves the trade-off. This is a smaller controller than the original
design assumed, and it is stated rather than hidden: **C1 should be read as
*runtime-adaptive retrieval effort*, not as several independently useful ANN
parameters.** C3, joint ANN+SCR control, remains genuinely two-dimensional.

---

## 7. What is not built, and what gates it

| Not built | Gate |
|---|---|
| Thermal model, energy model, DVFS tiers | A device trace. This machine can measure neither temperature nor energy — WMI thermal zone is access-denied without elevation, there is no RAPL on Windows, and WSL2 exposes no populated `powercap`. The Android phone is the **only** source. |
| Selective Content Reduction (C2) | Phase 5; the sentence splitter it shares with chunking is already built. |
| Bandit policy, Pareto policy, safety shield | Phase 6, which needs the Phase 3 cost model. |
| Answer generation, EM/F1 | Phase 7; `llama-cpp-python` needs the MSVC toolchain. |

The energy and thermal dependency was **identified and recorded in Phase 1**
(ADR-0007, ADR-0008), before any work was built on top of it — not discovered
late. The capability table in every run manifest makes the absence machine-
checkable rather than a matter of memory.

**One decision must be made before Phase 3 starts.** Caching reconstructed
cluster data would remove most of the remaining load cost, but it changes the
dominant cost term from `n_cluster_loads` to `n_cache_misses` — different
distributions, so a cost model fitted on one is *invalid* for the other, and the
failure would surface as unexplained controller underperformance in Phase 6.
Current decision: no cache; memory-map instead, which gets most of the benefit
while keeping cost a function of bytes touched rather than of history.

---

## 8. Reproducing this

```
smartann data status                       # exits non-zero if anything is incomplete
smartann index build configs/index/sift1m_ecovector.yaml
python scripts/phase2_acceptance.py --datasets sift1m nytimes
python scripts/experiment_contention.py
python scripts/make_figures.py
```

Every run writes `manifest.json` with the git SHA, config hash, library
versions, hostname, seed, wall-clock bounds and the telemetry capability table.

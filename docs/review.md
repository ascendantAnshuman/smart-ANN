# SMART-ANN — mid-project review

A working demonstration of the adaptive mechanism on measured data, plus the
Phase 2 index results. **Not** SMART-ANN as specified: the controller here is one
policy, and the thermal, energy, SCR and generation work is not built. What is
built is measured; nothing in this document or in any figure comes from a
simulator, a placeholder, or a model presented as a measurement.

---

## 1. What the process caught

Sixteen defects, each caught before it produced a reported number.

**Ranked by danger, which is not the same as severity** — by how likely each was
to survive undetected, not by how wrong it would have been.

| # | Defect | Why it ranks here |
|---|---|---|
| 1 | **Check 4 compared an index with itself and reported PASS** | A broken *measurement* gets caught by a check. A broken *check* catches nothing, including itself, and reports a green line in a summary table — the line nobody re-reads. |
| 2 | **A generated conclusion was hardcoded while its numbers were computed** | `headline_sentence()` ended with "the mechanism works; the control law does not". True of the run it was written for; false of the next, where it printed beside a *negative* recall cost. The prose would have been quoted into this report as a measured finding. |
| 3 | **The static baseline was not tuned, and the manifest said it was** | `max(key=nprobe)` returns the largest candidate unconditionally, and the target was derived from that same row, so selection met the target by construction. The number was recoverable; the manifest's description of its own method was not. |
| 4 | `compute_topk` dropped `‖q‖²` | Every ground-truth distance off by a per-query constant. Score-equality recall would have compared against a meaningless threshold and produced a *plausible* number. |
| 5 | Timing harness ran variants in blocks | A 3× spread between two configs doing provably identical work. Machine drift presented as a performance result. |
| 6 | Uncapped `ef·M` distance model | Overstated work by up to **29.5×**, on the field Phase 3 regresses against. |
| 7 | Footprint measured in a long-lived process | Same index read 5.18 MB or 38.24 MB depending on what ran before it. Changed a headline claim from 1.47× to 11.36× of the paper. |
| 8 | `decide()` called twice per query | AIMD halving effort twice per overshoot; overhead measured on the wrong call. |
| 9 | Overhead as a ratio of percentiles | Reported p95 *below* p50 — impossible for one distribution. Understated the worst case 5×. |
| 10 | Violations scored against the run's original target | Target-step's middle phase measured compliance with a budget no longer in force: static read 0.1% when it was 28.2%. |
| 11 | Burners contended for ALU, not memory bandwidth | The experiment would have applied almost no pressure, and "the controller did not react" would have looked like a result. |
| 12 | Shared time origin across sequential runs | Second run's data fell outside the zoom window: one figure panel empty, another showing a single policy. |
| 13 | `_controller_p95` referenced but never defined | A patch aborted before writing it; the figure still rendered, using the wrong estimator while labelled as the right one. |
| 14 | Timing runs corrupted by background load, **three times** | Once by a concurrent experiment, once by this project's own O(n²) figure generation perturbing the latencies it was plotting, once by leftover load. |
| 15 | Fork bomb from a missing `__main__` guard | Windows `spawn` re-imported the parent module in every child, recursively. |
| 16 | `nc="auto"` overridden for SIFT with the reason in a registry string | A real decision recorded where the decision log could not see it. |

### Five detection mechanisms, and what each costs

- **Caught by loud failure** (#15). Free. The machine fell over.
- **Caught by reading the code** (#1, #3, #8, #11, #14, #16). Cheap, but depends
  on someone reading the right lines on the right day. Six of sixteen, which is
  not a system.
- **Caught by rendering the artifact and looking at it** (#12, #13). Neither a
  threshold nor a test would fire: the script exited 0 and wrote a valid PNG
  both times. Only opening the file revealed an empty panel.
- **Caught by rereading a generated conclusion against the numbers printed
  beside it** (#2). Its own category, because narrative-generation bugs are
  uniquely durable: the numbers were right, the sentence was wrong, and nothing
  reruns a sentence. A hardcoded verdict in generated prose will eventually
  state the opposite of its own data, and no test asserts on English.
- **Caught by external reference** (#4, #5, #6, #7, #9, #10). The expensive
  category, and the **only one that catches errors producing plausible
  numbers**: published ground truth in the HDF5, an independent interleaved
  control, a physical containment bound, a fresh-process measurement, an
  impossible percentile ordering, a constraint that had changed.

**The ordering is the finding: cost of detection is inversely proportional to
how loudly a defect fails.** A fork bomb costs minutes. A recall metric quietly
off by a per-query constant, or a sentence asserting a conclusion its own
numbers contradict, costs however long it takes someone to distrust something
that looks fine.

That generalises better than a test count: **a test suite validates a system
against its own assumptions, so the errors that survive are exactly the ones the
assumptions share** — and generated prose is outside the test suite entirely.

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
| sift1m | 9.67 MB | **10.41 MB** | **1.08×** |
| nytimes | 3.53 MB | **40.12 MB** | **11.36×** |

SIFT agrees with the paper almost exactly. **NYTimes does not, and that is an
open question rather than a result.** It is 11.4× the paper's figure and 8.7×
our own Section 3.4 model (4.62 MB predicted). The likely cause is that the
per-cluster brute-force scan materialises the memory-mapped cluster during the
matmul — NYTimes clusters are 256-dimensional, so a probed set of 16 is roughly
9 MB of vectors against SIFT's 4 MB — but that is a hypothesis and has not been
measured. It does not affect the index's correctness or the controller results,
and check 5 still passes its 20% bound (40.12 MB against 56.6 MB), which is
precisely the case where a loose threshold tells a reader nothing.

Both figures were measured in a **fresh subprocess per dataset**. Measured inside
a long-lived process the NYTimes number came out as 5.18 MB in one ordering and
38.24 MB in another — a 7× swing driven by allocator residue from earlier checks.
On Windows psutil's private counter is commit charge, so these are upper bounds
on resident memory.

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

## 6. The adaptive mechanism: built and measured, first control law wrong

Three sentences, because the honest version is neither a success nor a setback:

1. **The mechanism is built and measured.** Per-query effort selection, full
   instrumentation, sub-1% controller overhead, every query recorded.
2. **The first control law is wrong, and the reason is structural** — it acts
   about fifty times faster than its own feedback arrives — which is a known
   failure mode with a known family of fixes.
3. **The experiment that found it is the experiment that was built to find it.**

### The disturbance was verified before anything was built on it

"We applied contention" is a claim; this is its evidence. Requirement: contended
p95 at least 2x quiet p95.

| burners | total mem | p50 (ms) | p95 (ms) | p95 ratio |
|---|---|---|---|---|
| quiet | — | 16.95 | 26.02 | — |
| 11 | 3.0 GB | 92.84 | 137.39 | **5.28x** |
| 12 | 3.0 GB | 93.78 | 151.00 | **5.80x** |
| 24 | 3.0 GB | 179.55 | 254.94 | **9.80x** |

The burners **stream through a large buffer** rather than spinning on
floating-point adds; the query path is a scan over memory-mapped flat arrays and
is memory-bandwidth bound, so an ALU spin loop barely touches it. **12 burners
were used, not 24**: 12 clears the bar, while 24 consumes the host almost
entirely and would leave the machine still settling when the burners stop,
making the recovery phase unrepresentative.

### Results — two independent disturbances, same conclusion

| | contention | target-step (deterministic) |
|---|---|---|
| static, disturbed | 99.9% over target, recall 0.8790 | 28.2% over target, recall 0.8805 |
| adaptive, disturbed | **9.1%** over target, recall 0.3072 | **6.3%** over target, recall 0.4079 |
| **adaptive, QUIET** | **5.9%** vs static's 1.3% | **6.1%** vs static's 0.7% |
| adaptive median `nprobe` | **1 in every phase** | **1 in every phase** |

> Under contention the static policy exceeded its latency target on **99.9%** of
> queries and the adaptive policy on **9.1%**, at a cost of **0.572 recall@10**
> (0.8790 → 0.3072).
>
> **But in the quiet phase, with no disturbance at all, the adaptive policy
> missed the target on 5.9% of queries against static's 1.3%, while holding
> median `nprobe` at 1 against 16 and giving up 0.430 recall.**

The second sentence is the finding. With nothing wrong, the controller is worse
on **both** axes at once. That cannot be explained by the workload, so the fault
is in the law and not the mechanism.

The `target-step` run is the cleaner evidence: no external pressure exists at
all, and the controller still sits at `nprobe=1` in every phase.

### Diagnosis: the control period is shorter than the observation delay

The policy acts **once per query**. The signal it acts on — a 50-sample windowed
p95 — does not reflect a change in effort for roughly **50 queries**. The
controller is therefore acting on the consequences of decisions it has already
superseded, up to fifty times over. That is a stability condition, not a tuning
preference, and it is violated by about 50×.

Under that condition multiplicative decrease ratchets *by construction*: seven
consecutive cuts take effort from 1.0 to below 0.01, and seven queries pass long
before fifty do.

**TCP AIMD adjusts `cwnd` once per RTT, not once per ACK, precisely because the
RTT is the feedback delay.** This implemented AIMD's arithmetic without its
clock.

The figure shows it rather than asserting it: `nprobe` sawtooths 1 → 33 → 1 at
roughly 0.4 s period through the entire quiet phase, then pins at the floor
under contention — a limit cycle, not an adaptation.

**A second-order problem, worth one line.** The window is fixed in *samples*, so
its delay in *time* is a function of the controller's own output:

| nprobe | ms/query | 50-sample window delay |
|---|---|---|
| 64 | 81.40 | **4.07 s** |
| 16 | 18.63 | 0.93 s |
| 4 | 5.60 | 0.28 s |
| 1 | ~1.10 | **0.055 s** |

A 74× swing across the action space. A controller whose feedback delay depends
on its own output is a harder object than one with a fixed delay, and this argues
for a time-based window independently of the ratchet.

**Controller overhead** (percentiles of the per-query ratio):

| policy | p50 | p95 |
|---|---|---|
| static | 0.002% | 0.004% |
| adaptive | **0.603%** | **0.861%** |

Under 1% even at p95, so the control layer is cheap even when its decisions are
poor. Hard rule 5 satisfied.

### The action space is one-dimensional, by measurement

`ef_centroid` is free — the centroid graph matches brute force exactly at every
`nprobe`, gap 0.0000 — and `ef_list` is inert once the in-cluster search is
exhaustive. `nprobe` is the knob that moves the trade-off. **C1 should therefore
be read as *runtime-adaptive retrieval effort*, not as several independently
useful ANN parameters.** C3, joint ANN+SCR control, remains genuinely
two-dimensional.

### Confirmatory run: the diagnosis is correct

The diagnosis makes a falsifiable prediction — pace actions no faster than the
feedback delay and the ratchet stops. The setting was **derived from
already-measured data and committed before the controller change existed**
(`docs/predeclared/control_period_test.json`, commit `62ee5bc`):
`control_period_s = 4.05`, from 50 samples × the measured quiet latency at
`nprobe=64` (81.40 ms). One run, one setting, all three possible outcomes
described in advance.

| | period = 0 (once per query) | period = 4.05 s |
|---|---|---|
| **quiet effort changes** | ~150 (sawtooth, ~0.4 s period) | **4 in 60 s** |
| quiet `nprobe` p50 | **1** (floor) | **20** (static: 16) |
| quiet recall | 0.4510 (static 0.8809) | **0.9115** (static 0.8806) |
| contended violations | 9.1% (static 99.9%) | 12.0% (static 99.8%) |
| recovery `nprobe` p50 | 1 | 7, violations 0.1% |

**The change count is the result.** Four effort changes across the quiet phase
against a continuous limit cycle is the ratchet stopping; the recall and
`nprobe` averages are downstream of it. The controller now settles slightly
above the tuned static baseline and earns *more* recall than it (0.9115 against
0.8806) while both sit near target.

Dropping to `nprobe=1` under contention is now correct rather than collapse:
static demonstrates that 99.8% of queries miss the target at `nprobe=16` when
the machine is 5.8× contended, so no effort level both serves and complies. The
controller conceding recall there is the trade working.

**Reported between "confirmed" and "partial", as pre-declared.** The ratchet
stopped — that was the prediction and it held. But recovery is slow: additive
increase now fires once per 4.05 s, so 60 s of recovery returned `nprobe` to 7
rather than 20. **No second setting was run.** That is a separate question with
its own predicted mechanism and its own declared value, recorded in
`docs/predeclared/recovery_slope_test.json` and not scheduled.

### Phase 6 direction

**Standing rule, generalising beyond AIMD:** a control loop must act no faster
than the feedback delay of the signal it reads, and this must be verified
explicitly for any future policy — including a bandit, whose reward is a delayed
observation in exactly the same sense. The failure is not a bad constant; it is
a violated stability condition, and it is checkable before a line of policy code
is written.

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

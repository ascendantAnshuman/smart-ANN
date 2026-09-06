# SMART-ANN — mid-project review

The adaptive mechanism demonstrated on measured data, the Phase 2 index
results, answer quality for the SCR ladder, the joint quality/cost surface, and
a device feasibility probe that closes the hardware gate.

**Not** SMART-ANN as specified. The thermal and energy models are not built, no
policy consumes device state, and the controller's own control law was shown to
be wrong and diagnosed rather than fixed. What *is* built is measured: nothing
in this document or in any figure comes from a simulator, a placeholder, or a
model presented as a measurement.

---

## 1. What the process caught

Twenty-two defects, each caught before it produced a reported number.

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
| 17 | **The CLI entry point never ran, and the verification certified it** | `cli.py` had no `__main__` guard, so `python -m smartann.cli <anything>` imported the module, ran nothing and exited **0**. Every runbook command used that form. `verify_all.py` checked `data status` *through the broken entry point* and scored the silent success as a pass, so the one check aimed at "do the documented entry points work" could only ever answer yes. Reported 7/7 was 6/7. (ADR-0042) |
| 18 | **Three of four ablation rungs were the same rung** | The SCR ladder swept `top_windows` 4/3/2/1 at `window=2`, but SQuAD chunks yield at most two windows, so three arms were byte-identical to the baseline. They read as "reduction is free" rather than "reduction did not happen". (ADR-0044) |
| 19 | **The generation baseline was silently truncated** | Two thirds of the unreduced arm exceeded the model's 512-token limit and was cut, making the baseline a *position-based* reduction. Comparing relevance-based reduction against that would have shown little difference and been read as "reduction is free". (ADR-0045) |
| 20 | **A "C3 supported" verdict that was pure measurement drift** | Phase 8 ran its cells in BLOCKED order and one generation took 20.4 minutes when the host slept. The mean cost of that cell read 5,778 ms against a median of 817 ms, and blocked ordering inflated ms/token from 1.01 in the cells measured first to 1.96 in those measured last. Both single-dimension baselines sit at the highest nprobe, measured last — so the drift inflated exactly the cells the joint cells were compared against, and manufactured the conclusion. Re-run interleaved with median cost: verdict survived, cell set changed. (ADR-0030, ADR-0046) |
| 21 | **A build that succeeds and then crashes on import** | hnswlib compiles on-device in 19 s, pip reports success, exit 0 — and `import hnswlib` dies with SIGILL. `-march=native` emitted an instruction the core lacks. Both builds exit 0; only the functional check the pre-declaration demanded distinguishes them. (ADR-0046) |
| 22 | **A figure registered in the dispatch table but not in the default set** | `fig_scr_quality` drew nothing on a default run — no figure, no error, and the summary reported success. Found by counting the files written against the figures that should exist. |

**Numbering is chronological, not ranked, from #17 on.** By this section's own
criterion #17 belongs at the top — it is the only defect here that corrupted the
project's *verification* rather than one of its measurements. The numbers are
left stable because ADRs cite them.

### Seven detection mechanisms, and what each costs

- **Caught by loud failure** (#15). Free. The machine fell over.
- **Caught by a user running the documented command** (#17). The cheapest
  mechanism in the list and the only one this project did not own: a reader
  pasted a runbook command into their own terminal and got nothing back.
  Every internal check had passed, including the one built to catch exactly
  this. Sixteen defects were found by the project looking at itself; this one
  needed someone outside it.
- **Caught by reading the code** (#1, #3, #8, #11, #14, #16). Cheap, but depends
  on someone reading the right lines on the right day. Six of twenty-two, which is
  not a system.
- **Caught by rendering the artifact and looking at it** (#12, #13). Neither a
  threshold nor a test would fire: the script exited 0 and wrote a valid PNG
  both times. Only opening the file revealed an empty panel.
- **Caught by rereading a generated conclusion against the numbers printed
  beside it** (#2). Its own category, because narrative-generation bugs are
  uniquely durable: the numbers were right, the sentence was wrong, and nothing
  reruns a sentence. A hardcoded verdict in generated prose will eventually
  state the opposite of its own data, and no test asserts on English.
- **Caught by an invariant the data itself must satisfy** (#20, #21). Generation cost is near-linear in prompt length, so cells disagreeing on
  ms/token were measured on different machine states rather than doing
  different work — a diagnostic that needs no external reference, only the
  ratio the physics requires. #21 is the same idea: a library that builds
  must also run, and the check that a build exits 0 does not test that.
- **Caught by printing the treatment beside the outcome** (#18, #19). Both
  would have produced quality numbers that were *correct* for what actually
  ran, and wrong for what was claimed to have run. Neither is visible in the
  quality column; both are obvious the moment the reduction percentage and
  the truncation rate sit in the same table. An ablation that does not report
  how much treatment it applied cannot be audited.
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

*(All figures below come from a single canonical acceptance run against the
current code, after the check-4, footprint and per-phase-target fixes. Earlier
drafts quoted a run that predated them.)*

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
| sift1m | 9.67 MB | **10.42 MB** | **1.08×** |
| nytimes | 3.53 MB | **39.85 MB** | **11.29×** |

SIFT agrees with the paper almost exactly. **NYTimes does not, and that is an
open question rather than a result.** It is 11.4× the paper's figure and 8.7×
our own Section 3.4 model (4.62 MB predicted). The likely cause is that the
per-cluster brute-force scan materialises the memory-mapped cluster during the
matmul — NYTimes clusters are 256-dimensional, so a probed set of 16 is roughly
9 MB of vectors against SIFT's 4 MB — but that is a hypothesis and has not been
measured. It does not affect the index's correctness or the controller results,
and check 5 still passes its 20% bound (39.85 MB against 56.6 MB), which is
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

## 6b. Phase 3 — the cost model

**Latency**, fitted per stage on 21,388 quiet queries and validated on 44,067
held-out contended ones:

| | MAPE | verdict |
|---|---|---|
| quiet (fitted) | **14.75%** | PASS (< 15%) |
| contended (held out) | **35.87%** | FAIL (< 15%) |

Reported as two numbers, never one. Averaging them would read as roughly 25% and
hide that the model is excellent in one regime and unusable in the other.

**The failure is the result, and its cause is measured.** A work-only model
scored 36.44%, so `dev_cpu_util` was added to every CPU-competing stage — named
from the mechanism before the refit and validated on held-out data. It moved the
number by 0.6 points, because the feature is **censored**: 12,469 contended rows
sit at exactly `cpu_util = 1.000` while `t_cluster_load` spans 0.67–27.59 ms
across them. A 41× spread in the thing it must predict, while the feature does
not move. A second mechanism compounds it — the contended median `cpu_util` is
0.446, *lower* than a saturated machine should read, because the 10 Hz telemetry
thread is itself starved under contention.

**Consequence for Phase 6:** the model is sound for choosing effort on a quiet
machine and unsound for anticipating a disturbance. That is a direct argument for
keeping the controller **reactive** rather than model-based — which is what the
AIMD governor already is. The fix is a non-saturating load metric, not another
coefficient on a censored feature (ADR-0033).

Two quantities are reported rather than silently fitted: `ef_centroid` is
**unidentifiable** (constant at 128 everywhere; the first fit returned R² =
0.0000), and `n_cluster_loads` / `bytes_read` are **weakly identified**
(r = 0.966), so their individual coefficients must not be quoted alone.

**Energy is specified, implemented and unit-tested — deliberately not fitted.**
`assert_energy_measurable()` refuses on a platform whose capability table says
energy is unavailable, and the fit script reports BLOCKED rather than a number.
**Phase 3 closes with no energy MAPE at all.** A criterion that cannot be
evaluated here is not marked PASS; Phase 3.5 fits and validates it on the phone.
`t_cluster_load` is billed to the CPU current rather than the disk current, which
is the measured departure from the paper (ADR-0026 amendment, ADR-0031).

## 6c. Phase 5 and 6 — SCR, the profile table, and a third policy

**SCR is built and its cost is measured. Its benefit is not measurable here.**
Windows over a retrieved chunk, scored against the query, top-`k` kept,
neighbours re-attached, overlaps merged. What the numbers say:

| | latency p95 | violations | recall | context tokens | reduction |
|---|---|---|---|---|---|
| static, no SCR | 18.17 ms | 36.4% | 0.9017 | 538 | — |
| static + SCR | **1223 ms** | 100% | 0.9134 | 369 | 30.4% |
| adaptive, no SCR | 14.39 ms | 1.8% | 0.8625 | 538 | — |
| adaptive + SCR | 1220 ms | 100% | 0.6916 | 370 | 30.2% |

**SCR costs 67x the latency to remove 30% of the tokens**, because it embeds
every window of every retrieved chunk on the query path — ten gte-small forward
passes per query at k=5. Every SCR arm violates its budget outright, and the
joint cell is *worse* than either single dimension: adaptation cannot recover a
cost that large.

This is **cost-only**, and it is no longer the last word. Phase 7 was never
actually blocked (ADR-0043) and section 6e now carries the quality numbers this
table could not produce. Read together they answer the question: **SCR as
implemented cannot pay for itself.** It costs ~1220 ms p95 on the query path to
save ~300 ms of generation, and every reduction level costs answer quality.

The escape is not adaptation but precomputation -- the corpus is static, so
window embeddings could be built at index time, which is the difference between
the ~1220 ms measured here and the ~8 ms measured warm in 6e.

One configuration finding worth carrying (ADR-0035): a first run reported **0.0%
reduction** while paying full cost, because 99.3% of SQuAD chunks hold ≤3
sentences and a 3-sentence window yields exactly one window per chunk. SCR can
only select among windows, so **its applicability is a property of the chunking,
not of the reducer** — and a zero-reduction result would have read as "SCR does
not help" rather than as a mismatch between two independently chosen defaults.

### The profile table and the budget policy

The controller now has three policies. The third reads a table of **measurements**
rather than the Phase 3 cost model, because that model fails at 35.87% MAPE
outside its fitted regime (ADR-0033) — a policy predicting from it would be most
confident exactly where it is most wrong.

```
budget = 14.01 ms, fixed from the table before any policy ran

  static   p95 14.20 ms   viol 6.0%   recall 0.8480   nprobe 12
  aimd     p95 13.44 ms   viol 4.3%   recall 0.8103   nprobe 10
  budget   p95  9.51 ms   viol 0.2%   recall 0.7797   nprobe  8
```

**The budget policy honours its declared latency budget at 0.2% violation — 30x
better than static — for 0.068 recall.** Controller overhead stays under 0.2% of
query latency for all three.

**C4 is half-delivered, and the half is named.** Latency budgets work and are
measured. Energy budgets have the machinery and no measurement:
`BudgetPolicy.honours_energy_budget()` returns **False**, and every run manifest
records it, because acting on a modelled energy as if measured is the precise
failure the capability gate exists to prevent.

**A methods note that cost a rebuild** (ADR-0036). The first profile sweep was
*blocked*, so latency came out falling as effort rose and the derived Pareto
front declared `nprobe` 1, 2, 4 and 8 "dominated and never worth choosing" —
exactly the cheap levels a controller must reach for under pressure. ADR-0030
had already required interleaving, having been written after the same mistake in
a different harness. **A rule recorded once in a document does not propagate to
code written later**, so the reason now lives in the sweep's own docstring.

## 6e. Phase 7 — answer quality, and what reduction actually costs

Until this phase every SCR setting could be priced only in characters, tokens
and milliseconds. On those axes a reduction that deletes the answer is
indistinguishable from a free one — both look like a win. This is the missing
axis.

SQuAD, n=1000, `google/flan-t5-base`, greedy decoding, one
retrieval configuration held fixed so every difference is attributable to
reduction alone. Paired bootstrap against `off`, 10,000 resamples; `*` marks a
95% CI excluding zero.

| arm | EM | F1 | tokens | reduction | gen ms | ΔF1 vs `off` |
|---|---|---|---|---|---|---|
| `off` | 0.6060 | 0.6769 | 571 | 0.0% | 650 | — (baseline) |
| `keep1_ext` | 0.5950 | 0.6661 | 475 | 19.4% | 536 | -0.0108 [-0.0212, -0.0009] \* |
| `keep2` | 0.5850 | 0.6569 | 448 | 25.2% | 514 | -0.0200 [-0.0329, -0.0083] \* |
| `keep1` | 0.5110 | 0.5824 | 269 | 61.4% | 346 | -0.0945 [-0.1154, -0.0740] \* |

**Reduction is not free at any level.** Every rung's confidence interval
excludes zero, including the mildest, which loses 0.011 F1 for 19.4% fewer
characters. The reading that the broken ladder would have produced — three arms
holding quality flat while cutting context — was an artifact, not a result
(ADR-0044).

**But the trade-off is smooth and monotone, which is what C2 needs.** `keep1`
buys a 47% cut in generation latency for a 14% relative F1 loss. A controller
under a latency budget has a real dial to turn, and the cost of turning it is
now a measured quantity rather than an assumption.

### The finding that decides whether SCR is worth having

SCR's own cost was measured here at ~8 ms per query against a **warm** embedding
cache. Phase 5 measured it on the query path, cold, at ~1220 ms p95. Those two
numbers bracket the entire question:

- **cold, on-path** — SCR costs roughly 4× what it saves. It cannot pay for itself.
- **warm, precomputed** — 8 ms against ~300 ms saved. A clear win.

The difference is *when* window embeddings are computed, and the corpus is
**static**, so they could be computed once at index time. That converts SCR from
a loss into a win, and it is an implementation choice rather than a property of
the method. It is recorded rather than acted on, because it would move the cost
model's dominant term — the same trap already flagged for cluster caching, where
a model fitted on `n_cluster_loads` is invalid for `n_cache_misses`.

## 6f. Phase 8 — C3, joint versus split control

The question is about a **surface**, not a controller: if the best joint setting
is no better than the best single-dimension setting at the same cost, no control
law can recover the difference. So the grid is swept directly.

Cost is the **median** of retrieval plus generation. `ms/token` is the drift
diagnostic — generation is near-linear in prompt length, so cells that disagree
on it were measured on different machine states, not doing different work.

| nprobe | SCR | family | F1 | cost ms | ms/token |
|---|---|---|---|---|---|
| 1 | `off` | retrieval only | 0.5287 | 1022 | 1.780 |
| 1 | `keep2` | joint | 0.5256 | 781 | 1.758 |
| 1 | `keep1` | joint | 0.4710 | 516 | 1.935 |
| 4 | `off` | retrieval only | 0.6459 | 1043 | 1.818 |
| 4 | `keep2` | joint | 0.6374 | 804 | 1.792 |
| 4 | `keep1` | joint | 0.5779 | 526 | 1.930 |
| 16 | `off` | retrieval only | 0.7134 | 1063 | 1.812 |
| 16 | `keep2` | reduction only | 0.7017 | 837 | 1.817 |
| 16 | `keep1` | reduction only | 0.6351 | 550 | 1.953 |

ms/token spread across cells: **1.11×**, within the 1.25× gate.

**Declared test:** a joint cell supports C3 if **no single-dimension cell
dominates it** — at least as accurate *and* at least as cheap. Result: nprobe=1/`keep1`, nprobe=4/`keep2`, nprobe=4/`keep1`
sit on the Pareto front.

**C3 is supported, and the shape of the support matters.** No joint cell
dominates the best single-dimension cell outright; the best F1 on the surface is
still retrieval-only at maximum effort. What joint control does is **extend the
frontier downward in cost** — it reaches operating points at 516–804 ms that
neither dimension reaches alone. That is precisely the region an adaptive
controller occupies under pressure, so the gain is real but it is a trade-off
along the front, not a free win.

**This verdict is the second one this experiment produced.** The first run also
said "C3 supported", and it was an artifact — see the defect table, #20.

## 6g. Phases 3.5 and 4 — the device

Everything above was measured on a laptop standing in for a phone. This section
is the phone.

**vivo I2304, Android 16, arm64-v8a, 8 cores in three clock domains.** Getting
the project onto it produced two findings before a single measurement was taken
(ADR-0046, ADR-0050): hnswlib builds and then **crashes on import** unless
`HNSWLIB_NO_NATIVE=1` is set, and HNSW index binaries **do not transfer**
between chroma-hnswlib and upstream hnswlib — proven with byte-identical files
that one library refuses to read. The centroid graph is therefore rebuilt
on-device in 0.016 s, and equivalence was checked rather than assumed: **host
and device both score recall@10 = 0.9543** on the same 300 queries.

### The thermal model — fitted, never chosen

19,144 queries over 25.0 minutes, `nprobe` swept 1→32, 478 thermal samples.
Skin temperature rose **32.90 → 37.51 °C**.

```
dT/dt = a·work_rate + b·(T − T_ref) + c
```

| window | n | R² | a | τ |
|---|---|---|---|---|
| 3 s | 446 | 0.109 | 6.08e-04 | 107 s |
| 15 s | 89 | 0.229 | 9.43e-04 | 67 s |
| 30 s | 44 | 0.304 | 9.14e-04 | 68 s |
| 60 s | 22 | 0.187 | 5.93e-04 | 97 s |
| 120 s | 11 | 0.691 | 8.69e-04 | 76 s |

**The whole sweep is reported because the differencing window moves R² sixfold**,
and reporting one row would hide that the answer depends on a choice no reader
would think to question. The 120 s row scores highest and is **excluded** —
eleven points against three free parameters is a number that flatters itself.

What is convincing is not the R² but the **stability**: τ stays 67–107 s and `a`
stays 5.9–9.4e-04 across every window. Point estimates that barely move across
analysis choices are better evidence than any single goodness-of-fit.

**τ ≈ 68 s is the result that matters.** Phase 6's control law collapsed because
its control period was shorter than its observation delay. This is the *measured*
delay for the thermal loop — so a thermally-aware controller sampling faster
than ~68 s would reproduce that collapse, and now there is a number saying so
rather than an argument.

**Latency on the device**, measured for the first time:

| nprobe | p50 | p95 |
|---|---|---|
| 1 | 5.59 ms | 13.82 ms |
| 4 | 20.50 ms | 52.26 ms |
| 8 | 40.46 ms | 101.46 ms |
| 16 | 82.61 ms | 189.82 ms |
| 32 | 169.21 ms | 354.10 ms |

Clean linear scaling, and roughly **13× slower than the laptop** — which is the
gap that makes adaptive effort worth having on a phone at all.

### Energy — measured, and one number refused

15,425 queries over 20.0 minutes, **unplugged**, 209 samples, **zero read
failures**. Every sample `DISCHARGING`; every current negative.

| | |
|---|---|
| total energy | **1531.4 J** |
| mean power | **1.277 W** |
| p5 / p95 | 0.736 / 2.218 W |
| all-in per query | **99.28 mJ** |

**Effort drives power, and this needs no model:**

| work (clusters/s) | median power |
|---|---|
| 95–126 | 0.864 W |
| 126–141 | 0.925 W |
| 141–153 | 0.989 W |
| 153–189 | 1.330 W |
| 189–251 | 1.940 W |

A 2.6× increase in retrieval work costs a 2.2× increase in power, correlation
+0.645. **That is C4's premise demonstrated on real
hardware:** retrieval effort has a measurable energy price, so an energy budget
is something a controller can genuinely trade against.

**What is refused.** The obvious next step — marginal energy per cluster — fits
at R² 0.415 and returns an idle power of
**-0.209 W**. A phone idling at negative power does not
exist. Work rate spans 95–251 clusters/s and never approaches zero, so the
intercept extrapolates 95 units past the nearest observation; independently, at
the trace's mean effort the marginal term predicts 115 mJ/query against a
measured all-in average of 99.3 mJ/query.

A number implying negative idle power is not a slightly-wrong version of the
right answer. It is computed, recorded as `reportable: false`, and printed with
its reasoning (ADR-0052).

**The fix is an experiment change, not an analysis change** (ADR-0053). The
trace now alternates 60 s of work with 30 s of idle, so the baseline is
measured rather than extrapolated, and the fit refuses to report attribution
unless at least 5% of intervals sit at zero work. Seven tests plant a known
baseline and slope and verify both are recovered. **What remains is 20 minutes
of handset time.**

### Two guards, from two silent failures

An earlier 20-minute run completed with **all 209 samples CHARGING** — median
current **0 mA**, a 103/102 split between negative and positive, the charger
modulating exactly as ADR-0047 predicted. Integrating over a sign that is a coin
flip yields approximately zero, so that run was not a weak measurement but no
measurement.

Asking more carefully was not the fix. `device_trace.py` now refuses to start
unless it reads `DISCHARGING`, and `fit_energy_model.py` refuses any trace
containing a charging sample or a positive current. Same lesson as
`MachineWatch` (ADR-0048): **a precondition asserted once and never verified is
not a precondition.**

## 6h. Contribution scoreboard

The phase table says what was built. This says what can be *claimed*, which is
the number that matters and is consistently lower.

| | Contribution | State | What is missing |
|---|---|---|---|
| **C1** | Runtime-adaptive ANN effort | **Demonstrated** — three independent experiments | Narrowed by measurement: `ef_centroid` is free and `ef_list` inert, so it is one-dimensional. Stated as *adaptive retrieval effort*, not "several ANN parameters". |
| **C2** | Adaptive SCR aggressiveness | **Demonstrated** — quality measured, n=1000 | Reduction costs quality at every level (all CIs exclude zero), and the trade-off is smooth enough to control. Whether SCR *pays* depends on precomputing window embeddings; on-path it costs 4× what it saves. |
| **C3** | Joint vs split control | **Supported on the surface** — 3 joint cells on the Pareto front | Joint control extends the frontier *downward in cost*; it does not beat single-dimension control at peak quality. A controller that exploits this is not yet built. |
| **C4** | Budget-constrained retrieval | **Half, and the half is the important one** | Latency budgets honoured at 0.2% violation. Energy is now MEASURED on the device and effort demonstrably drives power (0.86 → 1.94 W). `EnergyBudgetPolicy.honours_energy_budget()` returns True against a real source. Missing: per-query attribution, which needs one 20-minute trace with idle periods (ADR-0053). |
| **C5** | Battery / device-state policy | **Built, not demonstrated** | `DeviceStatePolicy` derates on temperature, thermal slope, battery and throttling; 25 tests pass against synthetic state. It has never run against the handset in an experiment, and until it does the claim is about code rather than behaviour. |

**Two demonstrated, one supported, two partial.** Up from one-and-a-half at
the last review, and the movement came from two places that were never really
blockers: a dependency that had been installed since Phase 1 (ADR-0043), and a
USB cable (ADR-0046).

The two partials are partial for different reasons, and the difference matters.
C4 has the measurement and lacks one 20-minute run. C5 has the code and lacks an
experiment. Neither is blocked by anything unknown.

## 7. What remains

Nothing is blocked. Everything below is a known quantity of work.

| | What | Cost |
|---|---|---|
| **C4** | Per-query energy attribution | One 20-minute device trace with idle periods. Instrumentation built and tested (ADR-0053). |
| **C5** | Demonstration | An experiment running `DeviceStatePolicy` against live handset telemetry. |
| **Phase 9** | On-device validation | Needs C4 and C5 first. |
| **Phase 6** | Bandit / Pareto policies, safety shield | Optional; the profile table and three policies already cover the claim. |

### The one decision that is not mine

**`DeviceState.freq_ratio` is a single scalar, and the target SoC has three
clock domains.** Eight cores at 2.016 / 2.803 / 3.187 GHz; one instantaneous
sample held ratios 1.00, 0.61, 0.22 and 0.27 simultaneously. Which ratio a query
experiences depends on where Android's scheduler placed the thread — something
this project neither controls nor observes from inside the process.

`types.py` is unchanged, because the options differ in what the controller would
be permitted to claim:

- a **per-cluster vector** — honest and complete, but changes every consumer of `DeviceState`
- the **ratio of the core the thread actually ran on** — physically meaningful, needs a per-query scheduler read
- a **documented worst case** — cheap, and weakens every claim built on it

`AndroidMonitor` currently reports `max(ratio)` across cores with a comment
saying it answers *"is any cluster at speed"*, which is what a throttling
detector asks. That is a placeholder, not an answer.

### Two rules that carry into all device work

**`HNSWLIB_NO_NATIVE=1` is mandatory.** Without it the on-device build succeeds
— exit 0, wheel produced — and the library dies with SIGILL on import. Both
builds report success; only running the code distinguishes them (ADR-0046).

**Measure unplugged.** The current's sign is unstable on AC. `device_trace.py`
refuses to start otherwise, and the fit refuses to integrate such a trace
(ADR-0047, ADR-0052).

### A standing design trap, now applying in two places

Caching reconstructed cluster data would remove most of the remaining load cost,
but it moves the dominant cost term from `n_cluster_loads` to `n_cache_misses` —
different distributions, so a model fitted on one is *invalid* for the other,
and the failure would surface as unexplained controller underperformance.
Current decision: no cache; memory-map instead.

**The same trap now applies to SCR.** Precomputing window embeddings at index
time would move its cost from ~1220 ms on the query path to ~8 ms — a change
worth making, and worth re-fitting for rather than slipping in.

## 8. Reproducing this

```
smartann data status                       # exits non-zero if anything is incomplete
smartann index build configs/index/sift1m_ecovector.yaml
python scripts/phase2_acceptance.py --datasets sift1m nytimes
python scripts/experiment_contention.py
python scripts/phase7_qa.py --n-questions 1000
python scripts/phase8_joint.py --n-questions 250
python scripts/make_figures.py
python scripts/verify_all.py                # all seven, end to end
```

On the handset, in Termux, **unplugged** (it refuses to start otherwise):

```
PYTHONPATH=/sdcard/smartann python /sdcard/smartann/device_trace.py   --minutes 21 --work-s 60 --idle-s 30 --out /sdcard/smartann/trace_c4
```

then back on the host:

```
python scripts/fit_thermal_model.py
python scripts/fit_energy_model.py
```

`python -m smartann.cli ...` is equivalent to the `smartann` console script.
It was **not** equivalent until 2026-09-05: `cli.py` had no `__main__` guard, so
the module form imported the CLI, ran nothing and exited 0. Every runbook
command used it, and `verify_all.py` routed its data-status check through it and
scored the silent success as a pass (ADR-0042). Steps now fail if they produce
no output.

Every run writes `manifest.json` with the git SHA, config hash, library
versions, hostname, seed, wall-clock bounds and the telemetry capability table.

# SMART-ANN

A runtime resource-aware adaptive control layer for on-device RAG. SMART-ANN
dynamically adjusts ANN retrieval effort and selective context reduction based
on thermal state, CPU/memory pressure, battery state, workload, and explicit
energy/latency constraints.

**Research question.** Can runtime adaptation maintain acceptable retrieval
quality while reducing energy per query and bounding p99 latency under changing
thermal and workload conditions, compared to a *well-tuned* static configuration?

The baseline is **MobileRAG** (Park, Lee, Kim — [arXiv:2507.01079](https://arxiv.org/abs/2507.01079)):
an EcoVector index (IVF clustering + HNSW over centroids in RAM + per-cluster
HNSW graphs on disk) plus Selective Content Reduction (SCR). MobileRAG's search
parameters are fixed at build time. SMART-ANN keeps that architecture and adds a
feedback controller on top.

## Architecture

```
                          ┌──────────────────────────┐
                          │   Device Monitor         │
                          │  temp, freq, CPU, RAM,   │
                          │  battery, charging       │
                          └────────────┬─────────────┘
                                       │ s_device
                                       ▼
Query ─► Embed ─► Cheap centroid probe ─► FEATURE VECTOR ─► CONTROLLER
                        │ (difficulty signal)                   │
                        │                          ┌────────────┴────────────┐
                        │                          ▼                         ▼
                        │                   ANN config                 SCR config
                        │              (nprobe, ef_c, ef_L,       (window, overlap,
                        │               k, per_cluster_k)         ctx_ext, top_w, τ)
                        ▼                          │                         │
                  EcoVector search  ◄───────────────┘                         │
                        │                                                     │
                        ▼                                                     │
                 Retrieved chunks ──────────────────────► SCR ◄───────────────┘
                                                            │
                                                            ▼
                                                    Prompt augmentation
                                                            │
                                                            ▼
                                                        sLM (Qwen2.5-0.5B)
                                                            │
                        ┌───────────────────────────────────┘
                        ▼
              Feedback: latency, energy, temp delta,
              retrieval confidence, SLO violation
                        │
                        └──────────► CONTROLLER (updates policy state)
```

## Run the demo

One command, from a fresh checkout, on Windows:

```powershell
.\scripts\run_demo.ps1
```

That script creates a virtual environment, installs the project with its
demonstration extras, builds the 71-document phone-care index, calibrates the two
thresholds from measurements, runs the offline proof, and opens the dashboard. It
is the whole Phase 11-15 path in order, and every step prints what it produced.

If you would rather drive it yourself:

```powershell
$PY = "C:\Users\HP\.venvs\smartann312\Scripts\python.exe"

& $PY scripts\phase11_build_demo_index.py    # 71 documents -> 255 chunks, 16 clusters
& $PY scripts\phase11_demo_eval.py           # retrieval eval + the measured profile table
& $PY scripts\phase12_calibrate.py           # refusal and cache thresholds, from the eval
& $PY scripts\phase12_offline_proof.py       # answers with every outbound socket blocked
& $PY -m streamlit run app\dashboard.py      # the demonstration page
```

And to ask one question in the terminal, with the whole pipeline explained:

```powershell
& $PY -m smartann.cli ask "Why does my phone get hot even when it is not charging?" `
    --dataset demo_phone_care --no-kb --budget-ms 12 `
    --table results\demo_profile_table.json --replay c5 --explain
```

`--replay c5` drives the controller from a **recorded handset trace**, so the
device state it reacts to is a measurement rather than an invention. `--stress
hot` pins it to the hottest moment ever recorded on the phone instead;
`--stress low-battery` uses a typed state and says so everywhere it appears.

**Python 3.12, not 3.13.** `chroma-hnswlib` publishes no cp313 wheel, so a 3.13
environment cannot install this project at all (ADR-0017).

**For a genuinely offline run**, set `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`. Without them the ML libraries contact the model hub at
load time even for a model already in the local cache -- which the Phase 12.5
proof discovered and records in `results/offline_proof.json`. The query path
itself sends nothing either way, and every trace carries the measured byte count.

## Ask it something

Retrieval returns passages. `smartann ask` returns an **answer**, and shows the
controller's decision, the evidence and the per-stage cost that produced it.

```bash
smartann ask "Who developed the theory of relativity?" --budget-ms 120
```

```
------------------------------------------------------------------------------
  1. CONTROLLER -- what effort to spend, decided before searching
------------------------------------------------------------------------------

    policy              budget
    latency budget      120 ms
    chose nprobe        8   (clusters to open)
    decision cost       28.6 us

    picked from the measured profile table, level nprobe=8:
      measured p95      101.5 ms   over 3,829 recorded queries
      measured recall   0.9065
      measured under    vivo I2304, on-device, 25-min trace, nprobe swept round-robin

------------------------------------------------------------------------------
  2. RETRIEVAL -- top 5 from squad
------------------------------------------------------------------------------

    8 clusters opened, 1,949 exact distances computed

    [1]  similarity 0.923   chunk #27287
         With his mathematical insight, Sir Isaac Newton formulated laws of
         motion that were not improved-on for nearly three hundred years. By
         the early 20th century, Einstein developed a theory of relativity ...

==============================================================================
  ANSWER
==============================================================================

      Einstein

      generated by google/flan-t5-base, greedy decoding

------------------------------------------------------------------------------
  COST -- measured on this machine, this query
------------------------------------------------------------------------------

    embed         40.0 ms  #---------------------------   3.2%
    probe          0.3 ms  #---------------------------   0.0%
    search       425.6 ms  ##########------------------  34.4%
    reduce         0.0 ms  #---------------------------   0.0%
    generate     772.5 ms  #################-----------  62.4%
    TOTAL       1238.3 ms
```

**The budget changes the work, not just the label.** The controller reads the
device-measured profile table and picks the highest-recall level whose measured
p95 fits, then the search opens exactly those clusters:

| budget | chosen nprobe | clusters opened | level's measured recall |
|---|---|---|---|
| 60 ms | 4 | 4 | 0.8417 |
| 120 ms | 8 | 8 | 0.9065 |

**It declines rather than inventing.** The prompt offers `unanswerable`, so a
question the corpus cannot support gets a refusal instead of a fabrication:

```
$ smartann ask "What is my bank PIN number?" --budget-ms 120

      unanswerable

      The model declined. The prompt offers 'unanswerable' so a
      generator handed context without the answer can say so instead
      of inventing one.
```

Useful flags: `--scr` turns context reduction on, `--nprobe N` fixes the effort
level when no budget is given, `--no-show-context` prints the answer without the
passages, and `--json-out PATH` writes the whole result as JSON.

**Two things it deliberately does not do.** It does not report recall for a
typed question, because recall needs a ground-truth neighbour list and one does
not exist for a question a user invented. The recall shown belongs to the effort
*level*, measured over the benchmark set, and is labelled that way. And it
refuses a profile table measured on a different dataset: `sift1m` latencies
choosing effort for a `squad` query would be plausible numbers from the wrong
corpus, which is the kind of mistake that leaves no symptom.

The answer path drives the same objects `scripts/phase7_qa.py` scores, so an
answer here and an answer counted in the C2 results come from one
implementation rather than two.

### Type your own questions

Run `ask` with no question and it prompts in a loop. The index, embedder and
generator load once, so the first question costs about thirty seconds and every
one after it is fast.

```
$ smartann ask --budget-ms 120

  Type a question and press Enter. 'quit' ends the session.
  The models stay loaded, so every question after the first is fast.

ask> Who developed the theory of relativity?
      ... -> Einstein                    752 ms
ask> What is the Amazon rainforest?
      ... -> moist broadleaf forest      684 ms
ask> quit
```

`--json-out results/answers.json` records the session, and the dashboard reads
that file.

## Grow the corpus, and the web demo

SQuAD is a 2016 snapshot, so it cannot answer "who is the CEO of Google". You can
add current material without touching the benchmark index:

```bash
smartann kb add-wiki --starter                    # a curated Wikipedia set
smartann kb add-wiki "Cricket" "Sundar Pichai"    # anything by title
smartann kb add-file notes.md docs/               # your own .txt or .md
smartann kb list
```

Added documents are chunked into three-sentence windows and embedded with the same
model as the benchmark, so their passages compete with SQuAD's on one cosine scale
and are merged by score. **The SQuAD index is never modified**, so every recorded
measurement stays reproducible. The knowledge base is searched exactly rather than
approximately: at this size a brute-force scan costs under a millisecond and has
recall 1.0 by construction, so there is no second approximation to account for.

With the starter set loaded, the two questions that used to fail now answer from
cited passages:

| question | before | after |
|---|---|---|
| who is the ceo of google | Brian McClendon, wrong | **Sundar Pichai**, cited to the Google article |
| how many players in a cricket team | `[1]`, a formatting bug | **eleven**, cited to the Cricket article |

Every answer names the passage it came from, with a link. An answer found in no
passage says so, because the model rephrased rather than quoted.

### The web demo

```bash
smartann serve
```

Opens `http://127.0.0.1:8000` with an ask box that shows the answer, its citation,
the controller's decision, the evidence with similarity scores, and a per-stage
latency bar. The dashboard is one click away. Loopback only, one question at a
time, and no framework: the standard library serves it, so no dependency was
added.

## Answering under real device conditions

The project's claim is that retrieval effort should follow the device's state.
That cannot be shown on the development laptop, which reports no temperature and
no battery current: `DeviceStatePolicy` correctly declines to act, and the demo
shows a controller doing nothing. Inventing a temperature to make it move would
break the rule this project is built on.

So the recorded handset traces drive it instead. `--replay` feeds the controller
the telemetry the phone actually reported, sample by sample:

```bash
smartann ask "who developed the theory of relativity" --budget-ms 120 \
  --replay c5 --sensor hottest
```

In an interactive session each question advances along the trace, so the phone
warms up as you ask and the effort level falls with it.

**Which thermometer decides everything.** The handset exposes a skin sensor and
the hottest of its internal zones, and on trace `c5` they disagree by 35.1 °C at
peak. The same policy, on the same trace, at the same budget:

| thermometer | peak | derated | levels used | mean recall |
|---|---|---|---|---|
| skin, the case | 35.3 °C | 7 of 237 samples | nprobe 4–8 | 0.9046 |
| hottest zone | 70.4 °C | 237 of 237 | nprobe 1 | 0.6405 |

`DeviceTiers` derates at 42 °C. Reading the skin, that threshold is never crossed
on a run that recorded a 70 °C component, and the shield never trips. Reading the
hottest zone, it trips throughout and gives up 0.2660 measured recall against the
0.9065 the latency budget alone allowed.

This is why C5 is recorded as **real but not actionable**: the signal the shield
needs is not the signal the platform puts in front of it. The dashboard replays
both and plots the decisions, and `smartann serve` lets you drag a slider along
the trace and watch the level change.

Every temperature in that table was recorded on the handset; every recall and
latency is the measured profile table's. The only thing computed is which row the
policy picks. A state you type by hand instead is carried through the whole
report labelled `HYPOTHETICAL`, and never presented as a measurement.

## Effort by query difficulty (C6)

`probe()` computes the centroid distance gap — how much the nearest cluster beats
the second — before any cluster is opened. A high gap means one cluster clearly
dominates and the query is easy; a gap near 1 means it sits between clusters and
the neighbours are scattered.

Until Phase 9 **no policy read it**. Every query at a given budget got identical
effort, which is the uniform allocation this project exists to argue against,
applied to the one axis where the information was already free.

```bash
python scripts/phase9_query_adaptive.py          # measure and fit
smartann ask --budget-ms 120 --query-adaptive    # spend the fitted allocation
```

The experiment buckets queries by measured gap, finds the allocation that
maximises recall under a measured device cost, and scores it on **500 queries
never used to fit it**. Recall is measured locally because it is a property of
the index and the query, not of the machine; every cost is the handset's measured
p95.

| budget | uniform recall | adaptive recall | delta | device ms |
|---|---|---|---|---|
| nprobe 1 | 0.6022 | 0.6022 | +0.0000 | 13.8 → 13.8 |
| nprobe 4 | 0.8212 | **0.8304** | +0.0092 | 52.3 → 49.9 |
| nprobe 8 | 0.8952 | **0.9114** | +0.0162 | 101.5 → 104.3 |
| nprobe 16 | 0.9472 | **0.9562** | +0.0090 | 189.8 → 196.3 |
| nprobe 32 | 0.9764 | 0.9758 | −0.0006 | 354.1 → 322.9 |

Three of five budgets improve. One held-out query gaining one neighbour moves
mean recall by 0.0002, so +0.0162 is about 81 query-neighbours — well clear of
noise. Nothing improves at nprobe=1 because every bucket is already at the floor;
there is nothing to reallocate.

The comparison is at **near**-equal cost, not exactly equal: the allocation is
constrained to the budget on the fit half, and the held-out half's difficulty mix
differs, so realised cost drifts by a few percent. The recorded verdict says so.

## The dashboard

```bash
smartann dashboard --open
```

One self-contained HTML file, ~1.2 MB, no network and no image directory — it
works emailed. Fifteen sections covering every result the project has produced:

| section | what it shows |
|---|---|
| Contributions | C1–C6, each with the file that settles it. C5 reads as a negative result. |
| The index | recall 0.30 → 0.95 across the effort sweep; 41 MB resident against 283 MB of raw vectors |
| Effort levels | the measured menu the controller chooses from |
| Policies compared | static 5.98% over budget vs budget policy 0.17% — a 35× reduction for 0.068 recall |
| Device conditions | both thermometers replayed over a measured run, plus all six runs |
| Query difficulty | recall by difficulty bucket, and the held-out C6 comparison |
| Energy | 131 mJ per query, 3.21 mJ per cluster opened, power up 98% from quietest to busiest fifth |
| Heat | where the device's power actually goes, and the controllable share |
| Cost model | the latency fit **and where it misses its target** |
| Answer quality | what context reduction costs, by arm |
| Joint control | the Pareto surface |
| Knowledge base | what you added, and when |
| Verification | 6 of 7 steps |
| Figures | every plot, embedded |

No number on it is typed by hand — each is read from the same JSON the figures
and the paper read, so the page cannot drift from the results. A section whose
input is missing renders as *not run* rather than disappearing, because "we
measured this and it was small" must not look like "we never measured this".

Two presentation rules the page follows, both of which change what a reader can
conclude. Latency and recall never share a pair of axes, because two scales on
one plot let a reader read a crossing point out of an arbitrary choice of units;
each gets its own bar column against its own maximum, with every value printed.
And a refusal is styled as a refusal, not as an answer, so `unanswerable` cannot
be mistaken for something the system found.

## Status

| Phase | Scope | State |
|-------|-------|-------|
| 0 | Scaffold, contracts, config, manifest | **done** |
| 1 | Datasets, embeddings, ground truth | **done** |
| 2 | EcoVector index | **done** — 14 PASS / 1 FAIL |
| 3 | Telemetry and cost model | **done** — latency fitted; energy specified, not fitted |
| 3.5 | On-device trace capture (ADR-0008) | **blocked** — needs the phone |
| 4 | Device simulator | blocked on 3.5's trace |
| 5 | Selective Content Reduction | **done** — cost measured, quality not measurable |
| 6 | Controller framework and policies | **done** — static, AIMD, budget + profile table |
| 7 | Full RAG pipeline | **blocked** — no `llama-cpp-python` wheel; needs MSVC |
| 8 | Benchmark harness and experiments | **done** — matrix, figures, verification |
| 9 | Android end-to-end validation | blocked, same gate as 3.5 |
| 10 | Paper artifacts | in progress — `docs/review.md` |

Every entry point is implemented; there are no `NotImplementedError` sites left,
and a test asserts that. Nothing in this repository returns a plausible dummy
value.

### Contributions — what can actually be claimed

The phase table says what was built. This says what the measurements support,
and it is consistently lower.

| | Contribution | State |
|---|---|---|
| C1 | Runtime-adaptive ANN effort | **demonstrated** — budget policy cuts SLO violations 5.98% → 0.17% |
| C2 | Adaptive SCR aggressiveness | **measured** — every reduction level tested cost quality; best arm is `off` at 0.6769 F1 |
| C3 | Joint vs split control | **supported** — 3 joint cells on the Pareto front, reachable by neither family alone |
| C4 | Budget-constrained retrieval | **demonstrated** — 131 mJ/query measured on the handset; power up 98% across effort |
| C5 | Battery / device-state policy | **NOT demonstrated** — the signal is real but the policy reads the wrong thermometer |
| C6 | Effort by query difficulty | **supported** — +0.0162 recall at near-equal cost, 500 held-out queries |

The dashboard derives this same table from the files on disk rather than from
this README, so if the two ever disagree, the page is right.

### Findings that changed the design

- **MobileRAG's per-cluster HNSW graphs do not pay** at the cluster sizes the
  design produces — identical recall to brute force, ~2x the wall-clock, and
  they still lose *inside* the regime the paper's own condition
  `ef_L·M' < N/Nc` says they should win. The paper's access model has no term
  for turning bytes into a searchable structure, which we measured at 95-99% of
  load time (ADR-0028, ADR-0031).
- **The first control law was structurally wrong**, not badly tuned: it acted
  once per query on a 50-sample windowed p95, so it acted on up to fifty
  superseded decisions. Pacing actions to the feedback delay fixed it, confirmed
  by a pre-declared prediction (ADR-0032).
- **The latency cost model cannot extrapolate** past its fitted regime, because
  `cpu_util` saturates: 12,469 samples pinned at 1.000 across a 41x latency
  spread. That is why the budget policy reads measured tables rather than
  predicting (ADR-0033).

## Why is the device hot?

The cable makes the question sharp. The handset is warm, nothing is charging it,
and something still put 7.8 C into it over 22 minutes. `smartann thermal explain`
answers that from the captured traces and scores how much of the heat retrieval
effort actually controls.

```bash
smartann thermal explain --prefix c5_          # the C5 policy trace
smartann thermal explain --prefix c4_ --json-out results/thermal_attribution.json
```

It splits the **measured** battery draw, and every term comes from a
discharging trace rather than a model:

| | C4 trace | C5 trace |
|---|---|---|
| idle floor, measured over real idle intervals | 0.657 W | 0.657 W |
| retrieval work, at the measured per-cluster slope | 0.304 W | 0.573 W |
| charging | 0.000 W — cable was out | 0.000 W — cable was out |
| unattributed | 0.055 W | 0.177 W |
| **controllable heat score** | **30 / 100** | **41 / 100** |

**The score is a ceiling on the controller, not a result of it.** A score of 41
means a policy that drove retrieval effort to zero would remove about two fifths
of the heat, and the idle floor would keep producing the rest. Power the split
cannot explain is counted against the score rather than handed to retrieval,
because heat nobody attributed is not heat a controller can remove.

**The term-by-term thermal decomposition is refused, and the refusal is the
interesting part.** Phase 4's fit was captured on AC, so its constant term
carries charger heat that cannot be separated from idle power. Dropping that
constant does not leave a conservative estimate, it leaves an unbalanced
equation: integrated over the C5 trace the retrieval term alone reaches +216 C
next to a thermometer that moved 5.6 C. One 21-minute unplugged trace, re-fit
with `scripts/fit_thermal_model.py`, unlocks that section. Nothing else is
missing.

### What it found: the policy is reading the wrong thermometer

Both device traces show the same thing, and it explains the C5 result rather
than softening it.

| | skin, the sensor `DeviceStatePolicy` reads | hottest zone |
|---|---|---|
| C4 peak | 34.0 C | 54.1 C |
| C5 peak | 35.3 C | 70.4 C |

`DeviceTiers` derates at 42 C and shields at 46 C, anchored on the one thermal
measurement available when it was written (ADR-0047). On both traces the skin
never approaches either line while the silicon sits above both for the entire
run. A policy watching the skin sees a cool device and does nothing, which is
exactly what C5 recorded: 542 of 19,719 adaptive decisions carried any device
reason at all.

That is a sensor-selection finding, not a tuning problem. Lowering the
thresholds until they fire on skin temperature would be choosing constants to
fit one trace, which is the rule this project keeps ("fitted, never chosen").
The open question is which thermometer the controller should read, and it sits
alongside the `freq_ratio` decision in `docs/review.md` section 7.

`scripts/make_figures.py --only thermal_attribution` draws both panels.

## What is real and what is simulated

This distinction matters for reading any number this project produces, so it is
stated up front rather than in a footnote.

**Real, always:** retrieval quality (recall@k against exact brute-force ground
truth), work counts (distance computations, cluster loads, bytes read), and CPU
time. Queries genuinely execute against a real index.

**Simulated, by design:** thermal state and DVFS. The simulator is driven by
*measured* per-query CPU time — the query really runs, and the simulator converts
the measured work into thermal and frequency consequences. Measured `t_s` at
`freq_ratio = 1.0` is scaled by `1/freq_ratio` to produce simulated latency. This
hybrid gives real recall and real work counts with reproducible thermal dynamics,
which is what makes 50,000-query experiments feasible and repeatable.

**Modelled:** energy. See the hardware caveat below.

Phase 9 validates a subset of all of this on a real Android device and reports
the fit in `results/real/sim_validation.md`.

### Hardware caveat on the current development machine

The development machine is Windows 11 on an Intel i5-13420H. It has **no
readable CPU temperature and no RAPL energy counters**:

| Source | Availability here |
|---|---|
| CPU package temperature | ✗ — WMI `MSAcpi_ThermalZoneTemperature` returns *Access denied*; WSL2 exposes no `thermal_zone*` at all |
| RAPL energy (`/sys/class/powercap`) | ✗ — not on Windows; WSL2's powercap directory is empty |
| CPU frequency ratio | ✓ — `\Processor Information\% Processor Performance` |
| Battery %, charging state | ✓ — WMI `Win32_Battery` |
| Battery discharge power (mW) | ✓ — WMI `BatteryStatus`, but only while unplugged, refreshing every ~15–60 s |
| CPU / memory utilisation | ✓ — `psutil` |

Consequently `energy_mj_measured` is `None` on this machine and
`temperature_c` comes from the simulator. Per Section 3 rule 1, no code path
substitutes a modelled value for a measured one. Real thermal and energy ground
truth comes from Phase 9 only, which is therefore **required, not optional**
(ADR-0007): it is the sole source of measured temperature and battery-current
energy, and the only thing that can calibrate the simulator's thermal constants.

`hnswlib` ships no Windows wheels for any Python version. This is resolved
without a compiler by `chroma-hnswlib`, a redistribution of the same library,
pinned on CPython 3.12 and verified by 21 contract tests (ADR-0017). The MSVC
Build Tools are still required for Phase 7's `llama-cpp-python`, which has no
wheel and no equivalent fallback.

## Datasets

The registry (`smartann/data/registry.py`) is the single source of truth for
every dataset fact, most importantly the distance metric. Phase 2's index config
reads `space` from the registry rather than declaring its own -- two independent
declarations of the metric is the bug that silently halves recall (ADR-0010).

| dataset | kind | dim | metric | normalised | search space | base vectors | queries | Nc |
|---|---|---|---|---|---|---|---|---|
| `sift1m` | ann | 128 | euclidean | no | `l2` | 1,000,000 | 10,000 | 2000 (pinned) |
| `nytimes` | ann | 256 | angular | yes | `ip` | 289,761 | 9,991 | auto |
| `squad` | qa | 384 | cosine | yes | `ip` | (chunked) | 1,000 | auto |

`search_space` is derived, never stored: on unit-norm vectors cosine and inner
product induce the same ordering, so a normalised cosine dataset is searched as
`ip`, and k-means becomes spherical k-means.

**NYTimes counts differ from its file counts.** The distributed HDF5 holds
290,000 base and 10,000 query vectors, but 239 base rows and 9 query rows have
exactly zero norm -- it is a bag-of-words-derived set and those rows retain no
terms. They are dropped during preparation and the shipped ground truth is
remapped into the compacted index space; the drop is lossless because no
neighbour list references a dropped row. See ADR-0014 for why keeping them would
have displaced correct results rather than merely wasted index slots.

**NYTimes contains exact duplicate vectors.** Brute-force top-10 disagrees with
the shipped `neighbors` array on about 2.6% of positions, and every one of those
disagreements is an exact cosine tie (measured gap 0.000e+00 in float64), with
our result never worse. Recall against this ground truth therefore has a small
tie-breaking artifact; Phase 8's recall metric needs to be tie-aware.

**SQuAD chunk counts do not match MobileRAG's reported 87,599** and are not
tuned to. That figure is SQuAD's question count, and the paper does not state
its chunking policy; ours is a 3-sentence sliding window with 1 sentence of
overlap over the union of unique train and dev contexts (ADR-0009).

## Quickstart

Every command, from a fresh checkout to figures, is in
[docs/RUNBOOK.md](docs/RUNBOOK.md). The essentials:


```bash
# Windows, from the repo root. The venv lives outside OneDrive on purpose.
python -m venv C:/Users/<you>/.venvs/smartann
C:/Users/<you>/.venvs/smartann/Scripts/python -m pip install -U pip uv
C:/Users/<you>/.venvs/smartann/Scripts/uv pip install -e ".[dev]"

smartann --help
smartann paths          # where bulk data is written
smartann ask --budget-ms 120             # type questions in a loop
smartann ask --budget-ms 120 --replay c5 --sensor hottest   # under measured phone conditions
smartann dashboard --open                # one HTML page of every result
smartann kb add-wiki --starter           # add current Wikipedia articles
smartann serve                           # the live web demo
smartann thermal explain --prefix c5_   # why the device got hot, and the score
pytest
ruff check .
mypy smartann
```

Bulk artefacts default to `C:/smartann-data` and `C:/smartann-results`, outside
the OneDrive-synced source tree — background sync contends for I/O and CPU
during timed loops and would corrupt the latency measurements this project
exists to produce. Override with `SMARTANN_DATA_ROOT` / `SMARTANN_RESULTS_ROOT`.

## Reproducibility

Every run is reproducible from `(config.yaml, seed)` and writes a
`manifest.json` recording git SHA, config hash, library versions, hostname,
seed and wall-clock bounds. Every experiment writes one `QueryRecord` row per
query to `queries.parquet`; every figure and summary statistic is derived from
that file and nowhere else.

## Known limitations

- Thermal behaviour on the dev machine is simulated, not measured (above).
- The simulator's `R_th`, `C_th`, `P_idle`, `P_dyn` require calibration against
  a real sustained-load trace (Phase 9). Until then they are uncalibrated.
- The i5-13420H is a hybrid P-core/E-core part; thread migration adds latency
  variance that inflates cost-model error unless affinity is pinned.

## Layout

See `smartann/` for the library, `scripts/` for entry points, `configs/` for
experiment definitions, and `docs/decisions.md` for the ADR log.

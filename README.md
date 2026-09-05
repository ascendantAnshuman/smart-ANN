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
| C1 | Runtime-adaptive ANN effort | **demonstrated** in three independent experiments |
| C2 | Adaptive SCR aggressiveness | mechanism + cost only; quality needs a generator |
| C3 | Joint vs split control | both dimensions run, cost-only |
| C4 | Budget-constrained retrieval | **half** — latency budgets honoured at 0.2% violation; energy unmeasurable here |
| C5 | Battery / device-state policy | not started — needs the phone |

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

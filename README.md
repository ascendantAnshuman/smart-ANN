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
                        │               k, rerank_cands)          ctx_ext, top_w, τ)
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
| 1 | Datasets, embeddings, ground truth | not started |
| 2 | EcoVector index | not started |
| 3 | Telemetry and cost model | not started |
| 4 | Device simulator | not started |
| 5 | Selective Content Reduction | not started |
| 6 | Controller framework and policies | not started |
| 7 | Full RAG pipeline | not started |
| 8 | Benchmark harness and experiments | not started |
| 9 | Android / real-device validation | not started |
| 10 | Paper artifacts | not started |

Unimplemented entry points raise `NotImplementedError` and are listed in
[TODO.md](TODO.md). Nothing in this repository returns a plausible dummy value.

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

Building the index additionally requires the MSVC C++ Build Tools, since
`hnswlib` ships no Windows wheels -- see TODO.md for the one-line install.

## Quickstart

```bash
# Windows, from the repo root. The venv lives outside OneDrive on purpose.
python -m venv C:/Users/<you>/.venvs/smartann
C:/Users/<you>/.venvs/smartann/Scripts/python -m pip install -U pip uv
C:/Users/<you>/.venvs/smartann/Scripts/uv pip install -e ".[dev]"

smartann --help
smartann paths          # where bulk data will be written
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

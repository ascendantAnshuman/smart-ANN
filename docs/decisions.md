# Architecture Decision Record

One short entry per non-obvious design decision, with the alternatives that were
considered and why they lost. Append; do not rewrite history.

---

## ADR-0001 — Repository root is the working directory, not a nested `smart-ann/`

**Date:** 2026-09-03 · **Phase:** 0 · **Status:** accepted

Section 5 draws the layout under a `smart-ann/` root. The working directory was
already an empty project folder, so nesting would have produced
`RAG ANN/smart-ann/smartann/`: three levels that mean the same thing.

**Decision.** Treat the working directory as the repository root. The import
package remains `smartann`.

**Alternatives.** Create the nested directory for literal fidelity to Section 5
— rejected as pure ceremony.

---

## ADR-0002 — `types.py` stays dependency-free; pydantic validates it from `config.py`

**Date:** 2026-09-03 · **Phase:** 0 · **Status:** accepted

Section 6 permits either frozen dataclasses or pydantic models for the core
contracts. `QueryRecord` is constructed once per query and `PipelineConfig` once
per decision, both on the hot path, where pydantic's validation cost would be
charged to exactly the latency numbers being measured. But configs loaded from
YAML genuinely do need validation.

**Decision.** `smartann/types.py` holds plain stdlib frozen dataclasses with no
third-party imports (enforced by a test). `smartann/config.py` validates *those
same classes* through `pydantic.TypeAdapter`, so field names and types have one
definition. Value-range rules that a dataclass cannot express (`nprobe >= 1`,
`rerank_candidates >= k`, `overlap_size < sliding_window_size`) live in
`config.py` as explicit `validate_*` functions.

**Alternatives.**
- Parallel `BaseModel` classes mirroring each dataclass — rejected: two field
  lists drift, and nothing catches the drift.
- pydantic models everywhere — rejected: validation cost on the hot path.
- `Annotated[int, Field(ge=1)]` inside `types.py` — rejected: it would pull
  pydantic into the hot-path module purely for metadata.

---

## ADR-0003 — Bulk data and results live outside the OneDrive-synced tree

**Date:** 2026-09-03 · **Phase:** 0 · **Status:** accepted

The repository sits in `OneDrive/Desktop/RAG ANN`. OneDrive sync competes for
I/O and CPU with no warning and no schedule the benchmark can observe. For a
project whose entire output is latency and energy measurement, a background
syncer touching multi-gigabyte index files mid-run is a correctness problem, not
an inconvenience. Concurrent sync of `.git` objects is additionally a known
source of repository corruption.

**Decision.** `PathsConfig` defaults `data_root` to `C:/smartann-data` and
`results_root` to `C:/smartann-results`, both overridable by the
`SMARTANN_DATA_ROOT` / `SMARTANN_RESULTS_ROOT` environment variables. The
virtualenv is created at `C:/Users/HP/.venvs/smartann`, also outside OneDrive.
Source stays in place, since it is small and version-controlled.

**Alternatives.**
- Keep everything in the repo and ask the user to exclude folders from sync —
  rejected: it depends on a manual step that silently reverts.
- Move the whole repository out of OneDrive — better, but that is the user's
  call; raised as a recommendation instead.

---

## ADR-0004 — Config hash is sorted-key JSON over a canonical form, never `hash()`

**Date:** 2026-09-03 · **Phase:** 0 · **Status:** accepted

Section 3 rule 4 requires every result to be reproducible from
`(config.yaml, seed)`, which requires config identity to be stable across
processes and machines.

**Decision.** `config_hash()` reduces any object to JSON primitives
(`canonical_dict`), serialises with `sort_keys=True` and `allow_nan=False`, then
takes SHA256[:12]. Two consequences are deliberate: `Path` objects normalise to
posix form so a hash does not change between Windows and Linux, and `NaN` raises
rather than hashing, because it has no canonical JSON form. A test runs the hash
in subprocesses under three different `PYTHONHASHSEED` values to catch any
accidental dependency on Python's salted `hash()` or on set iteration order.

**Alternatives.** `hash(frozenset(...))` — rejected: salted per process.
Pickle digest — rejected: not stable across Python or library versions.

---

## ADR-0005 — `dirty` git state is recorded, not raised

**Date:** 2026-09-03 · **Phase:** 0 · **Status:** accepted

A run started from a dirty working tree is not identified by its SHA.

**Decision.** `Manifest` records `git.dirty` rather than refusing to run, so
exploratory runs stay cheap. Any number destined for the paper must come from a
manifest with `dirty: false`; this will be asserted in the Phase 8 runner, not
here.

---

## ADR-0006 — Windows-native over WSL2 as the development target

**Date:** 2026-09-03 · **Phase:** 0 · **Status:** provisional

WSL2 Ubuntu is available and would build `hnswlib` immediately with gcc 13,
whereas native Windows currently cannot build it at all. It is nevertheless the
worse target for this project: WSL2 exposes **no** `thermal_zone*` entries, an
**empty** `/sys/class/powercap` (no RAPL), and no `cpufreq` — so every telemetry
signal the controller consumes would be unavailable. Native Windows at least
provides frequency ratio, battery percentage, charging state and battery
discharge power.

**Decision.** Develop on native Windows and install the MSVC C++ Build Tools
(`Microsoft.VisualStudio.2022.BuildTools`, VCTools workload) to unblock
`hnswlib`. Confirmed 2026-09-03; status raised from provisional to accepted.
This also unblocks `llama-cpp-python` for Phase 7, which has the same build
requirement.

---

## ADR-0007 — The dev machine is model-only; measured energy and temperature come from Phase 9

**Date:** 2026-09-03 · **Phase:** 0 · **Status:** accepted

This laptop can measure neither CPU temperature nor energy. WMI's
`MSAcpi_ThermalZoneTemperature` returns *Access denied*; there is no RAPL on
Windows without a kernel driver; WSL2 exposes no thermal zones and an empty
powercap directory. LibreHardwareMonitor would recover both, but only by
installing a signed kernel driver (WinRing0) running with admin privileges.

**Decision.** Decline the driver. On the dev machine `energy_mj_measured` stays
`None` and `temperature_c` comes from the simulator; no code path substitutes a
modelled value for a measured one (Section 3, rule 1). An Android device is
available, so Phase 9 becomes **required** rather than optional: it is the sole
source of measured temperature and battery-current energy, and therefore the
only thing that can calibrate the simulator's `R_th`, `C_th`, `P_idle` and
`P_dyn`.

**Consequences.**
- Phase 3's energy MAPE < 25% acceptance check **cannot be evaluated here**. The
  latency MAPE < 15% check can. Both facts get reported in
  `results/calibration/report.md` rather than quietly dropped.
- Energy calibration moves to the Phase 9 device, using integrated battery
  discharge current over sustained unplugged runs.
- Phase 9 targets Termux (Path A): on-device Python reading
  `/sys/class/thermal/thermal_zone*/temp` and
  `/sys/class/power_supply/battery/current_now`. No `adb` needed from the PC.

**Alternatives.** Install LibreHardwareMonitor — rejected: an admin kernel
driver is a heavy dependency outside Section 4's list, to rescue a machine that
is not the paper's target platform anyway. Report modelled energy as if
measured — never considered; it is precisely what rule 1 forbids.

---

## ADR-0008 — Phase 3.5 inserted: on-device trace capture; Phase 4 hard-depends on it

**Date:** 2026-09-03 · **Phase:** 1 (planning) · **Status:** accepted

The dev machine can measure neither temperature nor energy (ADR-0007). The
phone is therefore not the *best* source of thermal ground truth, it is the
*only* source. A simulator whose `R_th`, `C_th`, `P_idle`, `P_dyn` were chosen
rather than fitted would make every downstream thermal result unfalsifiable.

**Decision.** Insert **Phase 3.5, on-device trace capture**, between Phases 3
and 4. Scope is deliberately narrow: Termux, Python, the EcoVector index over
SIFT1M, a sustained-load loop, and a 1 Hz sysfs sampler. No controller, no SCR,
no generation.

Deliverable: `data/traces/<device>_sustained_<timestamp>.parquet` with columns
`t, temp_c, freq_khz, current_ua, voltage_uv, capacity_pct, charging, cpu_util,
query_latency_ms`, plus a 10-minute idle-cooldown trace for the thermal time
constant.

**Consequences.**
- Phase 4 **hard-depends** on 3.5. The thermal constants and the DVFS tier
  thresholds are *fitted to that trace*, not chosen. The tier table in the
  original prompt (45/55/65 °C at 1.00/0.90/0.75/0.55) is a placeholder to be
  overwritten by whatever the device actually does; tiers are fitted by
  clustering observed `freq_khz` against `temp_c`.
- Phase 9 shrinks to end-to-end validation runs plus the simulator-fidelity
  report, since calibration data was already captured in 3.5.
- `LinuxMonitor` must degrade per-source and loudly: each telemetry source is
  independently optional, a capability table is logged at startup and written
  into the run manifest, and an assertion makes it impossible for a manifest to
  claim measured energy while the capability table says unavailable.

---

## ADR-0009 — SQuAD chunking diverges from MobileRAG's reported 87,599 vectors

**Date:** 2026-09-03 · **Phase:** 1 · **Status:** accepted

MobileRAG Table 3 reports 87,599 base vectors for SQuAD. That is the *question*
count of SQuAD v1.1 train, not a context or chunk count, and the paper does not
specify its chunking policy.

**Decision.** State our policy, record whatever count it actually produces, and
do not reverse-engineer a chunk size to reproduce 87,599.

Policy: a **document** is a unique context paragraph; a **chunk** is a sliding
window of 3 sentences with 1 sentence of overlap (step 2), giving SCR something
sentence-shaped to operate on later.

**Alternatives.** Tune the window until the count lands near 87,599 — rejected.
It would be a fabricated parameter chosen to match a number whose meaning we do
not know, and it is indefensible at viva. "The paper does not specify its
chunking and we documented ours" is a complete answer; a fudged parameter is
not.

---

## ADR-0010 — The registry is the sole source of the distance metric

**Date:** 2026-09-03 · **Phase:** 1 · **Status:** accepted

Two independent declarations of the metric — one in the dataset registry, one
in the index config — is the bug that silently halves recall and costs a week
to find.

**Decision.** `DatasetSpec.space` records the dataset's **true** metric and is
asserted at load time against the ANN-Benchmarks HDF5 `distance` attribute
(`euclidean` -> `l2`, `angular` -> `cosine`). Vectors for cosine-metric datasets
are L2-normalised **at write time**, never at query time, and `normalized` is
recorded. The metric the index is actually built with is a *derived* property:

```python
search_space = "ip" if normalized else space
```

Cosine ordering is exactly preserved by normalisation, so shipped `neighbors`
arrays remain valid ground truth, and k-means becomes spherical k-means, which
is the correct clustering under cosine. Phase 2's index config reads the space
from the registry rather than declaring it, and asserts a match at build time.
Zero-norm rows are detected and reported rather than silently producing NaN.

This means `squad` is registered as `space="cosine", normalized=True`
(gte-small's native metric) rather than `space="ip"` as first tabled --
identical in effect, since `search_space` derives to `"ip"`, but it keeps one
fact in one place.

---

## ADR-0011 — ANN-Benchmarks HDF5 is the primary source; TEXMEX readers retained

**Date:** 2026-09-03 · **Phase:** 1 · **Status:** accepted

**Decision.** Fetch both ANN datasets from the ANN-Benchmarks HDF5
distribution, which carries `train`, `test`, `neighbors` (top-100) and
`distances` in one uniform layout. Brute-force ground truth is then needed only
for the QA chunk index.

`read_fvecs` / `read_ivecs` are still implemented: they are short, TEXMEX is the
canonical SIFT source, and they are the fallback if a mirror goes down. They are
simply not the primary path.

Mirrors are **not** silently used. A failed primary URL raises with the manual
download instructions and the expected destination path; `--allow-mirror` opts
in explicitly, and the URL actually used is recorded in `MANIFEST.json`.

---

## ADR-0012 — One sentence splitter, shared by chunking and SCR

**Date:** 2026-09-03 · **Phase:** 1 · **Status:** accepted

Phase 1 chunking builds 3-sentence windows; Phase 5 SCR builds its own
sentence windows over retrieved text. If the two use different splitters, SCR's
windows will not align with chunk boundaries -- a silent correctness bug that
would surface only as inexplicably poor SCR performance, several phases later.

**Decision.** A single `smartann/text.py` provides `split_sentences()`, backed
by `pysbd`. Both `smartann.data.chunking` and (in Phase 5)
`smartann.scr.core` import it. This module is an addition to the Section 5
layout, justified by the coupling it removes.

---

## ADR-0013 — `DatasetSpec` carries an artifact list, not a single URL set

**Date:** 2026-09-03 · **Phase:** 1 · **Status:** accepted

The specified `source_urls: list[str]` + `sha256: str | None` shape assumes one
file per dataset. SQuAD is two files (train and dev), and HotpotQA would be two
more -- so the single-file shape would force a refactor at exactly the point the
brief says adding a dataset should be "a config entry and a chunker call".

**Decision.** `DatasetSpec.artifacts` is a tuple of `DatasetArtifact(key,
filename, urls)`, each independently checksummed and recorded. Single-artifact
datasets are the common case and read naturally; multi-artifact ones need no
special handling.

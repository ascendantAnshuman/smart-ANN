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

**Decision (provisional).** Develop on native Windows and install the MSVC C++
Build Tools to unblock `hnswlib`. Revisit if that install is refused, in which
case WSL2 becomes a compute-only environment and *all* telemetry comes from the
simulator plus the Phase 9 device.

**Status is provisional** pending the toolchain decision; this entry will be
updated to accepted or superseded once made.

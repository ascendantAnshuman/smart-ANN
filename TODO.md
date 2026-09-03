# TODO

Every `NotImplementedError` site in the codebase, with the reason and the phase
that will deliver it (Section 3, rule 2). Nothing here returns a placeholder
value in the meantime.

Keep this file in sync when a stub is filled in or a new one is added.

## Phase 1 — Datasets, embeddings, ground truth

- `smartann/cli.py::data_status` — dataset status table; needs the Phase 1 dataset registry.
- `smartann/cli.py::data_prepare` — download/verify/chunk; needs the Phase 1 loaders.
- `smartann/cli.py::data_groundtruth` — exact top-100 brute force; needs the Phase 1 loaders.

## Phase 2 — EcoVector index

- `smartann/cli.py::index_build` — EcoVector build; needs `hnswlib`, which does **not**
  currently install on this machine (no MSVC toolchain — see README hardware caveat).
- `smartann/cli.py::index_info` — index metadata + memory model; blocked on the same.

## Phase 8 — Benchmark harness

- `smartann/cli.py::run_experiment` — experiment driver; needs Phases 2, 3, 4, 6.
- `smartann/cli.py::figures_make` — figure regeneration; needs the parquet schema from Phase 8.

## Known blockers (not stubs)

- **`hnswlib` cannot be installed here.** No binary wheels exist on PyPI and the
  source build requires MSVC ≥ 14.0, which is not present (the `VS 2019`
  directory on this machine contains no `cl.exe`). Phase 2 is blocked until
  either the C++ Build Tools are installed or the project moves to WSL2.
- **No measured energy or CPU temperature on the dev machine.** Phase 3's energy
  MAPE acceptance check cannot be satisfied here; only the latency check can.
  See README "Hardware caveat".

## Deferred design decisions

- `smartann/types.py` will need `SearchResult` (Phase 2) and `ThermalTier` /
  `Profile` enums (Phases 4 and 6). These are additions to the Section 6
  contracts and require explicit sign-off before being added.

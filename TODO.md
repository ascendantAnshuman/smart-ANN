# TODO

Every `NotImplementedError` site in the codebase, with the reason and the phase
that will deliver it (Section 3, rule 2). Nothing here returns a placeholder
value in the meantime.

Keep this file in sync when a stub is filled in or a new one is added.

## Phase 2 — EcoVector index

- `smartann/cli.py::index_build` — EcoVector build; needs `hnswlib`, which does **not**
  currently install on this machine (no MSVC toolchain — see README hardware caveat).
- `smartann/cli.py::index_info` — index metadata + memory model; blocked on the same.

## Phase 3.5 — On-device trace capture (ADR-0008)

- `smartann/telemetry/android.py` — Termux sysfs sampler; not written yet.
- Trace capture script; blocked on the phone, not on code.

## Phase 8 — Benchmark harness

- `smartann/cli.py::run_experiment` — experiment driver; needs Phases 2, 3, 4, 6.
- `smartann/cli.py::figures_make` — figure regeneration; needs the parquet schema from Phase 8.

## Known blockers (not stubs)

- **`hnswlib` cannot be installed until the MSVC C++ Build Tools are present.**
  No binary wheels exist on PyPI and the source build requires MSVC ≥ 14.0; the
  `VS 2019` directory on this machine contains no `cl.exe`. Resolution agreed
  (ADR-0006): install `Microsoft.VisualStudio.2022.BuildTools` with the VCTools
  workload. Phase 2 is blocked until then. `llama-cpp-python` (Phase 7) has the
  same requirement.

  ```powershell
  # run from an ADMIN PowerShell
  winget install --id Microsoft.VisualStudio.2022.BuildTools `
    --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
  ```

  Verify with: `C:/Users/HP/.venvs/smartann/Scripts/uv pip install hnswlib`

- **No measured energy or CPU temperature on the dev machine** (ADR-0007). Phase
  3's energy MAPE acceptance check cannot be evaluated here; only the latency
  check can. Both outcomes must be reported explicitly in
  `results/calibration/report.md`. See README "Hardware caveat".

## Deferred design decisions

- `smartann/types.py` will need `SearchResult` (Phase 2) and `ThermalTier` /
  `Profile` enums (Phases 4 and 6). These are additions to the Section 6
  contracts and require explicit sign-off before being added.

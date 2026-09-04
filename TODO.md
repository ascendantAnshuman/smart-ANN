# TODO

Every `NotImplementedError` site in the codebase, with the reason and the phase
that will deliver it (Section 3, rule 2). Nothing here returns a placeholder
value in the meantime.

Keep this file in sync when a stub is filled in or a new one is added.

## Phase 2 — EcoVector index

Delivered. `index build` and `index info` are implemented; the index package,
the acceptance harness and the recall metrics are in place. Nothing here raises
`NotImplementedError` any more.

## Phase 3.5 — On-device trace capture (ADR-0008)

- `smartann/telemetry/android.py` — Termux sysfs sampler; not written yet.
- Trace capture script; blocked on the phone, not on code.

## Phase 8 — Benchmark harness

- `smartann/cli.py::run_experiment` — experiment driver; needs Phases 2, 3, 4, 6.
- `smartann/cli.py::figures_make` — figure regeneration; needs the parquet schema from Phase 8.

## Known blockers (not stubs)

- **`llama-cpp-python` (Phase 7) still needs the MSVC C++ Build Tools.** The
  Phase 2 hnswlib blocker is resolved without them (ADR-0017), but the local
  generator is a source build with no wheel. Install from an ADMIN PowerShell:

  ```powershell
  winget install --id Microsoft.VisualStudio.2022.BuildTools `
    --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
  ```

- **Thermal and energy ground truth exist only on the Android phone**
  (ADR-0007, ADR-0008). This machine can measure neither.

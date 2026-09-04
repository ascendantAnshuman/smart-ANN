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

- **Phase 7 (generation) needs the MSVC C++ Build Tools. CONFIRMED, not assumed.**
  `uv pip install --only-binary=:all: llama-cpp-python` on CPython 3.12.14 /
  win-amd64 reports *"all versions of llama-cpp-python have no usable wheels"*.
  Source build only, and unlike hnswlib there is no differently-named
  redistribution to fall back on -- the rung-2 escape that resolved Phase 2
  (ADR-0017) does not exist here. Install from an ADMIN PowerShell:

  ```powershell
  winget install --id Microsoft.VisualStudio.2022.BuildTools `
    --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
  ```

- **Phases 3.5, 4 and 9 are gated on the device feasibility probe**
  (`docs/predeclared/device_feasibility.json`). Six of seven questions are
  BLOCKED on device access. `adb` is now installed at
  `C:/Users/HP/platform-tools/adb.exe` and works, but no phone enumerates on USB
  -- Windows itself sees no Android device, which points at the cable or port
  rather than at adb or developer options. `scripts/device_probe.sh` answers all
  seven in one Termux run once a device is reachable.

  Two of the seven can sink a phase rather than delay it: whether hnswlib builds
  under Termux, and whether battery current is readable without root. If the
  second fails there is no measured energy anywhere in the project, and Phase 3's
  energy criterion becomes unachievable as scoped rather than merely hard.

- **Thermal and energy ground truth exist only on the Android phone**
  (ADR-0007, ADR-0008). This machine can measure neither.

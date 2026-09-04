# TODO

Every `NotImplementedError` site in the codebase, with the reason and the phase
that will deliver it (Section 3, rule 2). Nothing here returns a placeholder
value in the meantime.

Keep this file in sync when a stub is filled in or a new one is added.

## Nothing is stubbed

Every CLI command is implemented; there are no `NotImplementedError` sites left
in `smartann/`. A test asserts this, so a stub added later without a matching
entry here will fail the suite rather than sit unnoticed.

## What remains, and what each needs

| Phase | Needs |
|---|---|
| 3.5 device trace | The phone. `scripts/device_probe.sh` answers all seven questions in one Termux run. |
| 4 simulator | Phase 3.5's trace. Thermal constants must be FITTED from a real trace, never chosen. |
| 7 generation | MSVC Build Tools (elevated install). |
| 9 device validation | The phone, same gate as 3.5. |

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

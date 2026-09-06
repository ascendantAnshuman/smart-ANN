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

Nothing is blocked. Each item below is a known quantity of work.

| Item | What it needs |
|---|---|
| **C4** per-query energy | ONE 21-minute device trace with idle periods. Instrumentation built and tested (ADR-0053). |
| **C5** demonstration | An experiment running `DeviceStatePolicy` against live handset telemetry. |
| **Phase 9** validation | C4 and C5 first. |
| **`freq_ratio`** | A decision, not code. See `docs/review.md` section 7. |

### How to run the C4 trace, start to finish

The handset fights this in two ways, so the order matters.

1. **Play Store -> profile -> Play Protect -> settings -> scan OFF.** Google
   re-enables this on its own, and it blocks the install every time.
2. `adb install --no-streaming -r -g termux-api.apk` — the ROM has deleted
   Termux:API twice, so check `pm list packages | grep termux` shows BOTH
   `com.termux` and `com.termux.api` before going further.
3. Verify the sensor responds in under a second. A 15-second timeout means the
   API is down and the trace will silently produce no energy.
4. **Unplug the cable.**
5. On the phone, in Termux: `cd /sdcard` then `bash go.sh`
6. Confirm it prints `power state OK: DISCHARGING`. If it says REFUSING TO
   START, the cable is still in.
7. **Leave the phone alone for 21 minutes.** Do not plug in to check progress --
   that was tried, and it charged the phone mid-trace and spoiled the run.
   The phone prints `TRACE FINISHED` when it is done.
8. Plug in, then on the host:
   `adb pull /sdcard/smartann/trace_c4/... results/device/` and
   `python scripts/fit_energy_model.py --prefix c4_`

The fit reports `reportable: true` only when 5%+ of intervals sat at zero work.
If it still refuses, the duty cycle did not run and the trace should be redone
rather than argued with.

## Known blockers (not stubs)

- **MSVC is still needed for `llama-cpp-python` specifically**, and that finding
  stands: `uv pip install --only-binary=:all: llama-cpp-python` on CPython
  3.12.14 / win-amd64 reports *"all versions have no usable wheels"*. It is no
  longer a project blocker, because Phase 7 does not require llama.cpp -- only
  a generator. Should a GGUF quantised model ever be wanted for on-device
  parity, install from an ADMIN PowerShell:

  ```powershell
  winget install --id Microsoft.VisualStudio.2022.BuildTools `
    --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
  ```

- **The device feasibility probe is COMPLETE: all seven questions answered**
  (`docs/predeclared/device_feasibility.json`, ADR-0046). Both critical ones
  passed. hnswlib builds and runs on the handset, and battery current is
  readable without root and responds to load. Phase 3.5 is GO.

  Two conditions carry into every device measurement:
  - **`HNSWLIB_NO_NATIVE=1` is mandatory.** Without it the build succeeds and
    the library SIGILLs on import.
  - **Measure unplugged.** The current's sign is unstable on AC (ADR-0047).

  And the centroid graph must be rebuilt on-device: HNSW binaries do not
  transfer between chroma-hnswlib and upstream hnswlib (ADR-0050).

- **Thermal and energy ground truth exist only on the Android phone**
  (ADR-0007, ADR-0008). This machine can measure neither.

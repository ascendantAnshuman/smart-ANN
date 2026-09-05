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
| 3.5 device trace | NOT RUN. Everything is staged: index on the handset at recall parity with the host, thermal sampling working, energy sign calibrated. `scripts/device_trace.py` runs it. |
| 4 simulator | Phase 3.5's trace. Thermal constants must be FITTED from a real trace, never chosen. |
| 9 device validation | Phase 3.5's trace first. |

**Phase 7 is done and was never actually blocked** (ADR-0043). The blocker was
recorded against `llama-cpp-python`, which does need MSVC, rather than against
the capability. `transformers` and `torch` have been installed since Phase 1 as
transitive dependencies of `sentence-transformers`, so a CPU generator was
available the whole time. `scripts/phase7_qa.py` runs SQuAD end-to-end on
`google/flan-t5-base` and scores EM/F1.

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

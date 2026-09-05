# Runbook

Every command needed to go from a fresh checkout to results.

## Where to run these

Open a terminal **in the project folder**:

- **VS Code**: press `Ctrl + ~` (the key above Tab). It opens already in the project.
- **Otherwise**: Start menu → "Windows PowerShell", then `cd` into the project.

Which shell you have is visible in the prompt: `PS C:\...>` is PowerShell,
`HP@machine MINGW64 ...$` is Git Bash. **The two syntaxes are not
interchangeable** — pick a column and stay in it.

| | PowerShell (Windows default) | Git Bash |
|---|---|---|
| set the interpreter | `$PY = "C:\Users\HP\.venvs\smartann312\Scripts\python.exe"` | `PY=C:/Users/HP/.venvs/smartann312/Scripts/python.exe` |
| run it | `& $PY -m smartann.cli data status` | `$PY -m smartann.cli data status` |
| path separator | backslash | forward slash |
| chain on success | `A; if ($?) { B }` | `A && B` |

**PowerShell needs the leading `&`** whenever the program path is held in a
variable — without it PowerShell prints the string instead of running it. Git
Bash must *not* have the `&`, where it means "run in the background".

Commands below are given for **PowerShell**. For Git Bash, drop the leading `&`
and use forward slashes.

**Paste only the lines inside the code blocks.** Prose describing what a command
should print is not a command; PowerShell will try to run it and report
`The term 'every' is not recognized as the name of a cmdlet`.

There are two equivalent ways to invoke the CLI. The module form is used
throughout this document:

```powershell
& $PY -m smartann.cli data status                             # module form
& C:\Users\HP\.venvs\smartann312\Scripts\smartann.exe data status   # console script
```

Until 2026-09-05 the module form was a **silent no-op that exited 0** -- no
`__main__` guard, so it imported the module and ran nothing (ADR-0042). If a CLI
command ever prints nothing at all, that is the failure mode to suspect;
`verify_all.py` now fails any step that produces no output.

---

## 0. The interpreter

**Python 3.12, not 3.13.** The project pins `chroma-hnswlib==0.7.5`, whose only
Windows wheel is cp312 (ADR-0017). On 3.13 the install fails outright.

```powershell
cd "c:\Users\HP\OneDrive\Desktop\RAG ANN"
$PY = "C:\Users\HP\.venvs\smartann312\Scripts\python.exe"
& $PY -c "import sys, hnswlib; print(sys.version.split()[0], hnswlib.__file__)"
```

That should print `3.12.14` and a path ending `hnswlib.cp312-win_amd64.pyd`.
If it errors, recreate the environment:

```powershell
uv python install 3.12
uv venv --python 3.12 C:\Users\HP\.venvs\smartann312
uv pip install --python $PY -e ".[dev]"
```

None of this needs an elevated prompt.

---

## 1. One rule before running anything

**Run experiments alone** — nothing else on the machine, including analysis of a
previous run.

Three separate measurements in this project were silently corrupted by
concurrent work: twice by another experiment, once by this project's own
figure-generation code perturbing the latencies it was plotting. Each was caught
only because a number looked wrong. `phase2_acceptance.py` now refuses to start
above 25% CPU; `--allow-busy` overrides that, and then its timings are not
trustworthy and must not be quoted.

---

## 2. Data — once, ~4 hours, mostly embedding

Already done on this machine. Check first:

```powershell
& $PY -m smartann.cli data status     # exits NON-ZERO if anything is incomplete
```

From scratch:

```powershell
& $PY -m smartann.cli data prepare sift1m      # ~1 GB download
& $PY -m smartann.cli data prepare nytimes
& $PY -m smartann.cli data prepare squad       # ~3.7 h: embeds 46k chunks
& $PY -m smartann.cli data groundtruth squad   # brute force; ANN sets ship theirs
& $PY -m smartann.cli data gt-dists sift1m
& $PY -m smartann.cli data gt-dists nytimes
& $PY -m smartann.cli data verify sift1m       # re-checksum against the registry
```

`data status` exiting zero is the gate for everything below.

---

## 3. Indexes — ~5 minutes total

```powershell
foreach ($d in "sift1m","nytimes","squad") {
  & $PY -m smartann.cli index build "configs\index\${d}_ecovector.yaml"
}

# Baselines, needed by acceptance checks 4 and 10
foreach ($d in "sift1m","nytimes") {
  & $PY -m smartann.cli index build "configs\index\${d}_flat_hnsw.yaml"
  & $PY -m smartann.cli index build "configs\index\${d}_ivf_disk.yaml"
  & $PY -m smartann.cli index build "configs\index\${d}_ecovector_graphs.yaml"
}
```

Inspect one, including its predicted memory model:

```powershell
& $PY -m smartann.cli index info C:\smartann-data\index\sift1m\<config-hash>
```

`--force` rebuilds, clearing the directory first — a cluster changing from graph
to flat would otherwise leave a stale file that `bytes_on_disk` counts.

---

## 4. Verify the whole project — ~15 min, run alone

```powershell
& $PY scripts\verify_all.py
```

Seven steps through the documented entry points: tests, lint, types, data
status, index acceptance, cost-model fit, figures. Writes
`results/verification.json`.

**Expect 7/7 with the acceptance step exiting 1.** That exit code is correct:
one check genuinely fails (NYTimes check 4) and it is a property of the dataset,
not a defect. A verification demanding exit 0 there could only be satisfied by
hiding a real result.

---

## 5. Experiments

Each writes `queries.parquet` + `manifest.json` under `C:\smartann-results\`.

```powershell
& $PY scripts\phase2_acceptance.py --datasets sift1m nytimes   # index checks, ~13 min
& $PY -m smartann.cli run experiment contention                # ~9 min
& $PY -m smartann.cli run experiment target-step               # ~9 min
& $PY -m smartann.cli run experiment profile                   # ~8 min
& $PY -m smartann.cli run experiment matrix                    # ~20 min
& $PY scripts\fit_cost_model.py                                # ~5 s
```

---

## 6. Figures

```powershell
& $PY -m smartann.cli figures make
& $PY -m smartann.cli figures make C:\smartann-results\E_DEMO_contention
```

Every number comes from stored parquet or a summary JSON. A missing input fails
loudly rather than drawing an empty axis — no plotting script here contains a
hardcoded result (Section 3, rule 1).

---

## 7. Where things live

| Path | Contents |
|---|---|
| `C:\smartann-data\` | datasets, ground truth, indexes, `MANIFEST.json` |
| `C:\smartann-results\` | one directory per run: parquet + manifest |
| `results\` (in repo) | figures, profile table, cost model, verification |
| `docs\review.md` | the report: status, results, findings, what is blocked |
| `docs\decisions.md` | 42 ADRs — the reasoning behind every non-obvious choice |

Neither `data/` nor `results/` is committed.

---

## 8. What is blocked, and what unblocks it

| Phase | Needs |
|---|---|
| 3.5 device trace, 9 validation | A phone reachable over USB. `adb` is installed at `C:\Users\HP\platform-tools\adb.exe`; Windows currently enumerates **no** Android device at all, which points at the **cable** before anything else. Then USB debugging, then tap "Allow" on the handset. Termux must be the F-Droid or GitHub build — the Play Store one's `pkg install` is broken. |
| 4 simulator | Phase 3.5's trace. Thermal constants must be **fitted** from a real trace, never chosen. |
| 7 generation | MSVC C++ Build Tools, elevated install. Confirmed necessary: `llama-cpp-python` has no wheel for any version on cp312/win-amd64, and unlike hnswlib there is no differently-named redistribution to fall back on. |

Once the phone is connected, one command answers all seven blocking questions:

```powershell
& C:\Users\HP\platform-tools\adb.exe push scripts\device_probe.sh /sdcard/
# then, in Termux ON THE PHONE:
#   bash /sdcard/device_probe.sh 2>&1 | tee /sdcard/device_probe.log
```

Two of those seven can **sink** a phase rather than delay it: whether hnswlib
builds under Termux, and whether battery current is readable without root. If
the second fails there is no measured energy anywhere in this project, and
C4/C5 need rescoping rather than postponing.

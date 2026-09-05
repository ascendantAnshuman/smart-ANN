# Runbook

Every command needed to go from a fresh checkout to results. Commands are shown
for Git Bash on Windows; the interpreter path is the only Windows-specific part.

## 0. The interpreter

**Python 3.12, not 3.13.** The project pins `chroma-hnswlib==0.7.5`, whose only
Windows wheel is cp312 (ADR-0017). Using 3.13 fails at install, not at runtime.

```bash
cd "c:/Users/HP/OneDrive/Desktop/RAG ANN"
PY=C:/Users/HP/.venvs/smartann312/Scripts/python.exe
$PY -c "import sys, hnswlib; print(sys.version.split()[0], hnswlib.__file__)"
```

If that environment is missing, recreate it:

```bash
uv python install 3.12
uv venv --python 3.12 C:/Users/HP/.venvs/smartann312
uv pip install --python C:/Users/HP/.venvs/smartann312/Scripts/python.exe -e ".[dev]"
```

`uv python install` needs no elevation. Neither does anything else below.

---

## 1. One rule before you run anything

**Run experiments alone.** Nothing else on the machine, including analysis of a
previous run.

Three separate measurements in this project were silently corrupted by
concurrent work — twice by another experiment, once by this project's own
figure-generation code perturbing the latencies it was plotting. Each was caught
only because a number looked wrong. `scripts/phase2_acceptance.py` now refuses
to start above 25% CPU (`--allow-busy` overrides it, and then the timings are
not trustworthy and should not be quoted).

---

## 2. Data (once, ~4 hours, mostly embedding)

Already done on this machine — check before re-running:

```bash
$PY -m smartann.cli data status        # exits NON-ZERO if anything is incomplete
```

From scratch:

```bash
$PY -m smartann.cli data prepare sift1m       # ~1 GB download
$PY -m smartann.cli data prepare nytimes
$PY -m smartann.cli data prepare squad        # ~3.7 h: embeds 46k chunks
$PY -m smartann.cli data groundtruth squad    # brute force; the ANN sets ship theirs
$PY -m smartann.cli data gt-dists sift1m
$PY -m smartann.cli data gt-dists nytimes
$PY -m smartann.cli data verify sift1m        # re-checksum against the registry
```

`data status` exiting zero is the gate for everything below.

---

## 3. Indexes (~5 minutes total)

```bash
for d in sift1m nytimes squad; do
  $PY -m smartann.cli index build "configs/index/${d}_ecovector.yaml"
done

# Baselines, needed by acceptance checks 4 and 10
for d in sift1m nytimes; do
  $PY -m smartann.cli index build "configs/index/${d}_flat_hnsw.yaml"
  $PY -m smartann.cli index build "configs/index/${d}_ivf_disk.yaml"
  $PY -m smartann.cli index build "configs/index/${d}_ecovector_graphs.yaml"
done
```

Inspect one, including the predicted memory model:

```bash
$PY -m smartann.cli index info C:/smartann-data/index/sift1m/<config-hash>
```

Add `--force` to rebuild; it clears the output directory first, because a
cluster that changes from graph to flat would otherwise leave a stale file that
`bytes_on_disk` counts.

---

## 4. Verify the whole project (~15–20 min, run alone)

```bash
$PY scripts/verify_all.py
```

Seven steps from the documented entry points: tests, lint, types, data status,
index acceptance, cost-model fit, figures. Writes `results/verification.json`.

**Expect 7/7, with the acceptance step exiting 1.** That exit code is correct:
one check genuinely fails (NYTimes check 4), and it is a property of the dataset
rather than a defect. A verification demanding exit 0 there could only be
satisfied by hiding a real result.

---

## 5. Experiments

Each writes `queries.parquet` + `manifest.json` under `C:/smartann-results/`.

```bash
# Index acceptance on its own (~15 min)
$PY scripts/phase2_acceptance.py --datasets sift1m nytimes

# Adaptive control under CPU contention (~9 min)
$PY -m smartann.cli run experiment contention

# Same mechanism, deterministic, no external pressure (~9 min)
$PY -m smartann.cli run experiment target-step

# Profile table + static/aimd/budget comparison (~8 min)
$PY -m smartann.cli run experiment profile

# Retrieval x SCR matrix on SQuAD (~20 min)
$PY -m smartann.cli run experiment matrix

# Refit the latency cost model on recorded runs (~5 s)
$PY scripts/fit_cost_model.py
```

---

## 6. Figures

```bash
$PY -m smartann.cli figures make                          # newest run
$PY -m smartann.cli figures make C:/smartann-results/E_DEMO_contention
```

Every number comes from stored parquet or a summary JSON. A missing input fails
loudly rather than drawing an empty axis — no plotting script in this repository
contains a hardcoded result (Section 3, rule 1).

---

## 7. Where things live

| Path | Contents |
|---|---|
| `C:/smartann-data/` | datasets, ground truth, indexes, `MANIFEST.json` |
| `C:/smartann-results/` | one directory per run: parquet + manifest |
| `results/` (repo) | figures, profile table, cost model, verification |
| `docs/review.md` | the report: status, results, findings, what is blocked |
| `docs/decisions.md` | 40 ADRs — the reasoning behind every non-obvious choice |

Neither `data/` nor `results/` is committed.

---

## 8. Blocked, and what unblocks it

| Phase | Needs |
|---|---|
| 3.5 device trace, 9 validation | A phone reachable over USB. `adb` is installed at `C:/Users/HP/platform-tools/adb.exe`; Windows currently enumerates no Android device at all, which points at the **cable** before anything else. Then USB debugging, then tap "Allow" on the handset. Termux must be the F-Droid or GitHub build — the Play Store one's `pkg install` is broken. |
| 4 simulator | Phase 3.5's trace. Thermal constants must be **fitted** from a real trace, never chosen. |
| 7 generation | MSVC C++ Build Tools, elevated install. Confirmed necessary: `llama-cpp-python` has no wheel for any version on cp312/win-amd64, and unlike hnswlib there is no differently-named redistribution. |

When the phone is connected, one command answers all seven blocking questions:

```bash
adb push scripts/device_probe.sh /sdcard/
# then in Termux on the device:
bash /sdcard/device_probe.sh 2>&1 | tee /sdcard/device_probe.log
```

Two of those seven can **sink** a phase rather than delay it — whether hnswlib
builds under Termux, and whether battery current is readable without root. If
the second fails there is no measured energy anywhere in this project, and C4/C5
need rescoping rather than postponing.

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
`per_cluster_k >= k`, `overlap_size < sliding_window_size`) live in
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

**Decision.** Develop on native Windows and install the MSVC C++ Build Tools
(`Microsoft.VisualStudio.2022.BuildTools`, VCTools workload) to unblock
`hnswlib`. Confirmed 2026-09-03; status raised from provisional to accepted.
This also unblocks `llama-cpp-python` for Phase 7, which has the same build
requirement.

---

## ADR-0007 — The dev machine is model-only; measured energy and temperature come from Phase 9

**Date:** 2026-09-03 · **Phase:** 0 · **Status:** accepted

This laptop can measure neither CPU temperature nor energy. WMI's
`MSAcpi_ThermalZoneTemperature` returns *Access denied*; there is no RAPL on
Windows without a kernel driver; WSL2 exposes no thermal zones and an empty
powercap directory. LibreHardwareMonitor would recover both, but only by
installing a signed kernel driver (WinRing0) running with admin privileges.

**Decision.** Decline the driver. On the dev machine `energy_mj_measured` stays
`None` and `temperature_c` comes from the simulator; no code path substitutes a
modelled value for a measured one (Section 3, rule 1). An Android device is
available, so Phase 9 becomes **required** rather than optional: it is the sole
source of measured temperature and battery-current energy, and therefore the
only thing that can calibrate the simulator's `R_th`, `C_th`, `P_idle` and
`P_dyn`.

**Consequences.**
- Phase 3's energy MAPE < 25% acceptance check **cannot be evaluated here**. The
  latency MAPE < 15% check can. Both facts get reported in
  `results/calibration/report.md` rather than quietly dropped.
- Energy calibration moves to the Phase 9 device, using integrated battery
  discharge current over sustained unplugged runs.
- Phase 9 targets Termux (Path A): on-device Python reading
  `/sys/class/thermal/thermal_zone*/temp` and
  `/sys/class/power_supply/battery/current_now`. No `adb` needed from the PC.

**Alternatives.** Install LibreHardwareMonitor — rejected: an admin kernel
driver is a heavy dependency outside Section 4's list, to rescue a machine that
is not the paper's target platform anyway. Report modelled energy as if
measured — never considered; it is precisely what rule 1 forbids.

---

## ADR-0008 — Phase 3.5 inserted: on-device trace capture; Phase 4 hard-depends on it

**Date:** 2026-09-03 · **Phase:** 1 (planning) · **Status:** accepted

The dev machine can measure neither temperature nor energy (ADR-0007). The
phone is therefore not the *best* source of thermal ground truth, it is the
*only* source. A simulator whose `R_th`, `C_th`, `P_idle`, `P_dyn` were chosen
rather than fitted would make every downstream thermal result unfalsifiable.

**Decision.** Insert **Phase 3.5, on-device trace capture**, between Phases 3
and 4. Scope is deliberately narrow: Termux, Python, the EcoVector index over
SIFT1M, a sustained-load loop, and a 1 Hz sysfs sampler. No controller, no SCR,
no generation.

Deliverable: `data/traces/<device>_sustained_<timestamp>.parquet` with columns
`t, temp_c, freq_khz, current_ua, voltage_uv, capacity_pct, charging, cpu_util,
query_latency_ms`, plus a 10-minute idle-cooldown trace for the thermal time
constant.

**Consequences.**
- Phase 4 **hard-depends** on 3.5. The thermal constants and the DVFS tier
  thresholds are *fitted to that trace*, not chosen. The tier table in the
  original prompt (45/55/65 °C at 1.00/0.90/0.75/0.55) is a placeholder to be
  overwritten by whatever the device actually does; tiers are fitted by
  clustering observed `freq_khz` against `temp_c`.
- Phase 9 shrinks to end-to-end validation runs plus the simulator-fidelity
  report, since calibration data was already captured in 3.5.
- `LinuxMonitor` must degrade per-source and loudly: each telemetry source is
  independently optional, a capability table is logged at startup and written
  into the run manifest, and an assertion makes it impossible for a manifest to
  claim measured energy while the capability table says unavailable.

---

## ADR-0009 — SQuAD chunking diverges from MobileRAG's reported 87,599 vectors

**Date:** 2026-09-03 · **Phase:** 1 · **Status:** accepted

MobileRAG Table 3 reports 87,599 base vectors for SQuAD. That is the *question*
count of SQuAD v1.1 train, not a context or chunk count, and the paper does not
specify its chunking policy.

**Decision.** State our policy, record whatever count it actually produces, and
do not reverse-engineer a chunk size to reproduce 87,599.

Policy: a **document** is a unique context paragraph; a **chunk** is a sliding
window of 3 sentences with 1 sentence of overlap (step 2), giving SCR something
sentence-shaped to operate on later.

**Alternatives.** Tune the window until the count lands near 87,599 — rejected.
It would be a fabricated parameter chosen to match a number whose meaning we do
not know, and it is indefensible at viva. "The paper does not specify its
chunking and we documented ours" is a complete answer; a fudged parameter is
not.

---

## ADR-0010 — The registry is the sole source of the distance metric

**Date:** 2026-09-03 · **Phase:** 1 · **Status:** accepted

Two independent declarations of the metric — one in the dataset registry, one
in the index config — is the bug that silently halves recall and costs a week
to find.

**Decision.** `DatasetSpec.space` records the dataset's **true** metric and is
asserted at load time against the ANN-Benchmarks HDF5 `distance` attribute
(`euclidean` -> `l2`, `angular` -> `cosine`). Vectors for cosine-metric datasets
are L2-normalised **at write time**, never at query time, and `normalized` is
recorded. The metric the index is actually built with is a *derived* property:

```python
search_space = "ip" if normalized else space
```

Cosine ordering is exactly preserved by normalisation, so shipped `neighbors`
arrays remain valid ground truth, and k-means becomes spherical k-means, which
is the correct clustering under cosine. Phase 2's index config reads the space
from the registry rather than declaring it, and asserts a match at build time.
Zero-norm rows are detected and reported rather than silently producing NaN.

This means `squad` is registered as `space="cosine", normalized=True`
(gte-small's native metric) rather than `space="ip"` as first tabled --
identical in effect, since `search_space` derives to `"ip"`, but it keeps one
fact in one place.

---

## ADR-0011 — ANN-Benchmarks HDF5 is the primary source; TEXMEX readers retained

**Date:** 2026-09-03 · **Phase:** 1 · **Status:** accepted

**Decision.** Fetch both ANN datasets from the ANN-Benchmarks HDF5
distribution, which carries `train`, `test`, `neighbors` (top-100) and
`distances` in one uniform layout. Brute-force ground truth is then needed only
for the QA chunk index.

`read_fvecs` / `read_ivecs` are still implemented: they are short, TEXMEX is the
canonical SIFT source, and they are the fallback if a mirror goes down. They are
simply not the primary path.

Mirrors are **not** silently used. A failed primary URL raises with the manual
download instructions and the expected destination path; `--allow-mirror` opts
in explicitly, and the URL actually used is recorded in `MANIFEST.json`.

---

## ADR-0012 — One sentence splitter, shared by chunking and SCR

**Date:** 2026-09-03 · **Phase:** 1 · **Status:** accepted

Phase 1 chunking builds 3-sentence windows; Phase 5 SCR builds its own
sentence windows over retrieved text. If the two use different splitters, SCR's
windows will not align with chunk boundaries -- a silent correctness bug that
would surface only as inexplicably poor SCR performance, several phases later.

**Decision.** A single `smartann/text.py` provides `split_sentences()`, backed
by `pysbd`. Both `smartann.data.chunking` and (in Phase 5)
`smartann.scr.core` import it. This module is an addition to the Section 5
layout, justified by the coupling it removes.

---

## ADR-0013 — `DatasetSpec` carries an artifact list, not a single URL set

**Date:** 2026-09-03 · **Phase:** 1 · **Status:** accepted

The specified `source_urls: list[str]` + `sha256: str | None` shape assumes one
file per dataset. SQuAD is two files (train and dev), and HotpotQA would be two
more -- so the single-file shape would force a refactor at exactly the point the
brief says adding a dataset should be "a config entry and a chunker call".

**Decision.** `DatasetSpec.artifacts` is a tuple of `DatasetArtifact(key,
filename, urls)`, each independently checksummed and recorded. Single-artifact
datasets are the common case and read naturally; multi-artifact ones need no
special handling.

---

## ADR-0014 — Zero-norm vectors are dropped from cosine datasets, not repaired

**Date:** 2026-09-03 · **Phase:** 1 · **Status:** accepted

Preparing NYTimes surfaced a property of the dataset rather than a bug in the
code: **239 of its 290,000 base vectors and 9 of its 10,000 query vectors have
exactly zero norm.** It is a bag-of-words-derived set, so some rows carry no
surviving terms. Measured facts, not assumptions:

- Non-zero norms cluster tightly around 1.0 (median 0.999154, range
  0.807-1.204) -- the set is *approximately* but not exactly normalised.
- The zero rows appear **nowhere** in the shipped top-100 `neighbors` array:
  0 occurrences across all 10,000 x 100 entries.
- The shipped `distances` array contains **NaN**, produced by the 9 zero-norm
  *queries*. ANN-Benchmarks did not resolve this either.

A zero vector has no direction. It cannot be normalised, its cosine distance to
everything is undefined, and -- the part that actually bites -- under an
inner-product search it scores exactly 0.0, which ranks it *above* every
genuinely anti-correlated vector. Left in the index it does not waste a slot, it
displaces correct results and depresses measured recall for reasons unrelated to
the retrieval system being studied.

**Decision.** Drop them explicitly during preparation, and remap the shipped
ground truth into the compacted index space. Because no neighbour list
references a dropped row, the remapping is lossless and no neighbour list
becomes ragged; `remap_neighbors` raises rather than truncating if that ever
stops being true. Zero-norm *queries* are dropped along with their ground-truth
rows, since they are unanswerable by construction.

`l2_normalize` continues to *raise* on zero-norm input. Dropping is an explicit,
recorded step in `prepare.py`, never a silent side effect of normalisation, and
the dropped counts go into `MANIFEST.json` and the prepare report.

**Consequences.** NYTimes' usable counts differ from its file counts. The
registry's `expected_n_base` / `expected_n_queries` continue to describe the
*file* (that is what the reader asserts against), while the processed artifacts
and the manifest record the usable counts. Both numbers must appear in the
paper's dataset table.

**Alternatives.**
- Substitute a unit vector for each zero row -- rejected outright: it fabricates
  data, which Section 3 rule 1 forbids.
- Keep them and search with hnswlib's `cosine` space -- rejected: it divides by
  the zero norm, and k-means over non-unit vectors would disagree with the
  search metric anyway.
- Leave them un-normalised in an `ip` index -- rejected: this is the case that
  silently displaces correct results.

---

## ADR-0015 — No test may assert on unimplemented behaviour without the isolation fixtures

**Date:** 2026-09-03 · **Phase:** 1 · **Status:** accepted

A Phase 0 test asserted that `smartann data prepare squad` raises
`NotImplementedError`. When Phase 1 implemented that subcommand, the assertion
did not merely start failing -- the test **executed the command**. It downloaded
35 MB, began writing a chunk database into the developer's real data root, and
left two orphaned pytest processes running for hours holding a SQLite lock. A
test that encodes "this does not work yet" becomes a live-fire command the
moment it does.

**Decision.** `tests/conftest.py` applies two autouse fixtures to every test:

- `isolate_data_roots` points `SMARTANN_DATA_ROOT` and
  `SMARTANN_RESULTS_ROOT` at a per-test temporary directory, so no test can
  write to real data even if it invokes a fully implemented command.
- `no_accidental_network` replaces `requests.get`/`head` with a raiser unless
  the test carries the `slow` marker. Tests that mock `requests.get` themselves
  are unaffected, since their patch applies after this one.

**Standing rule.** No test may assert on unimplemented behaviour without those
fixtures in force, and any name removed from the `STUBS` list in
`tests/test_cli.py` must gain real coverage in the same commit. The `STUBS` list
carries a comment recording this incident so the reason survives.

---

## ADR-0016 — MANIFEST.json merges on write; concurrent preparations are safe

**Date:** 2026-09-03 · **Phase:** 1 · **Status:** accepted

`DataManifest` read the file at construction and overwrote it at the end. Two
preparations running concurrently therefore raced: NYTimes and SQuAD were
prepared in parallel, SQuAD constructed its manifest before NYTimes finished,
and SQuAD's write erased all four of NYTimes' artifact entries. Phase 1
acceptance check 7 is what caught it -- a manifest completeness check earns its
keep the first time something like this happens.

**Decision.** `write()` re-reads the on-disk artifacts and merges its own
entries over them, so a concurrent writer's records survive. The write goes to a
temporary file and is then renamed, so an interrupted write cannot leave a
partial manifest. `_read_artifacts` tolerates a corrupt file rather than making
the whole pipeline unrunnable: recording what we know beats recording nothing.

Tests reproduce the exact lost-update sequence, the corrupt-file recovery, and
the absence of a leftover temp file.

---

## ADR-0017 — hnswlib comes from `chroma-hnswlib` on Python 3.12, not a compiler

**Date:** 2026-09-03 · **Phase:** 2 (gate) · **Status:** accepted ·
**Supersedes the toolchain half of ADR-0006**

Phase 2's gate is `import hnswlib`. Resolving it without elevation required
knowing which of several assumptions were actually true, so all of them were
measured:

- **Upstream `hnswlib` publishes no binary wheels at all.** Not for cp311,
  cp312 or cp313, on any platform: `pip download hnswlib --only-binary=:all:`
  reports "from versions: none" for every one. It is source-only, so it cannot
  be installed anywhere without a C++ toolchain. This kills the "just pick a
  different Python version" idea outright.
- **MSVC cannot be installed from this session.** `winget install
  Microsoft.VisualStudio.2022.BuildTools` downloads and hash-verifies, then
  fails with **exit code 1602 (user cancelled)** because the session is not
  elevated and the UAC prompt cannot be answered.
- **WSL2 cannot install a toolchain unattended either.** `sudo -n true` fails:
  no passwordless sudo, so `apt install build-essential` needs a human.
- **`chroma-hnswlib` ships a cp312 Windows wheel.** Version 0.7.5 does; 0.7.6
  is source-only on Windows. There is no cp313 wheel for any version.

**Decision.** Depend on `chroma-hnswlib==0.7.5` and target **Python 3.12**.
Chroma's redistribution imports as `hnswlib` and is the same library. The
standalone CPython 3.12.14 was installed by `uv python install 3.12`, which
needs no elevation.

**Why this is not a compromise on the science.** `tests/test_hnswlib_contract.py`
pins every behaviour Phase 2 depends on, verified on the actual binary:

- the full API surface (`init_index`, `add_items`, `save_index`, `load_index`,
  `set_ef`, `knn_query`, `get_items`, `get_ids_list`, ...) and all three spaces
- **per-query `set_ef` trades recall for work monotonically**: recall@10 runs
  0.527 / 0.655 / 0.811 / 0.935 / 0.988 / 0.998 across ef 8 -> 256. This is the
  premise the entire adaptive layer rests on, now measured rather than assumed.
- `save_index`/`load_index` round-trips to bit-identical labels and distances,
  which is what per-cluster graph load/unload requires
- `knn_query` returns float32 **distances**, needed for score-equality recall
- custom int64 labels survive serialisation, so per-cluster graphs can store
  global vector ids directly instead of needing a second mapping layer

**One semantic trap, now pinned by a test.** hnswlib's `ip` space returns
**`1 - inner_product`**, and its `l2` space returns **squared** distance. Our
ground truth stores raw similarities for `ip` datasets. Comparing the two
directly inverts the ranking, and nothing in the types would reveal it. Both
conventions are asserted in the contract tests, and the Phase 1 ground-truth
addendum stores distances in the library's own units.

**Consequences.**
- `requires-python` becomes `>=3.11,<3.13`; the 3.13 environment is abandoned.
- The paper should cite hnswlib and note the binary distribution used.
- Re-embedding was not needed: the SQLite embedding cache is raw float32 keyed
  by SHA-256, and gte-small embeddings re-produced under 3.12 match those
  written under 3.13 to cosine >= 0.999. Cross-version reproducibility, free.
- If MSVC is installed later, switching to upstream `hnswlib` is a one-line
  dependency change and the contract tests guard the swap.

**Alternatives.** `usearch` ships a cp313 wheel and is a capable HNSW
implementation with memory-mapped views that would suit per-cluster loading --
rejected because Section 4 mandates hnswlib and a different graph
implementation would make the MobileRAG comparison inexact. `faiss-cpu` ships
wheels -- rejected by Section 4 for the primary index. WSL2 -- available as
rung 4 but it needs a password *and* costs every telemetry signal (no thermal
zones, no RAPL, no cpufreq), so it is strictly worse when a no-elevation path
exists on Windows.

---

## ADR-0018 — Ground-truth distances are *gathered*, stored in hnswlib's units

**Status.** Accepted (Phase 1 addendum, before Phase 2).

**Context.** Score-equality recall compares a retrieved distance against the
ground-truth distance at depth k. That needs stored distances, and Phase 1
stored only neighbour *ids*. Three problems had to be settled at once: which
values, in which units, aligned to what.

**Decision.**

1. **Gather, do not recompute.** `gather_distances(base, queries, neighbors,
   space)` computes the distance from each query to *its own stored neighbours*.
   Re-running `compute_topk` would produce its own neighbour ordering, and any
   tie resolved differently would leave the distances silently misaligned with
   the ids they describe. Gathering makes
   `out[q, j] == dist(queries[q], base[neighbors[q, j]])` true by construction,
   and costs `n_queries * depth` comparisons instead of `n_queries * n_base`
   (1.6 s for SIFT1M, versus hours).

2. **Store in hnswlib's own convention** — squared L2 for `l2`, `1 - ip` for
   `ip`/`cosine`, recorded per artifact as `dists_convention` in `MANIFEST.json`.
   The alternative is a conversion at every comparison site, and one missed
   conversion inverts a ranking with no symptom. Verified directly against the
   library: `test_gather_distances_agrees_with_hnswlib_itself` builds an index,
   queries it, and asserts our numbers equal the ones it returns.

3. **Assert row monotonicity over every query, not a sample.** The ids were
   shipped by the dataset and remapped onto a compacted base (ADR-0014); these
   distances were computed independently from the compacted vectors. If the
   remap had put an id on the wrong row, the ordering would break. Worst
   violations measured: SIFT `+0.00e+00`, NYTimes `-5.96e-08`, SQuAD
   `-4.17e-07`, all float32 noise against a `1e-5` tolerance.

**A bug this found.** `compute_topk` computed L2 as `||b||^2 - 2qb`, dropping
`||q||^2` because it is constant per query and cannot change the ordering. True
for ranking, wrong for a value that leaves the function: every stored distance
was off by a per-query constant, and score-equality recall would have compared
against a meaningless threshold. Every ordering test the module had passed. The
regression test is now a *value* comparison against
`scipy.spatial.distance.cdist(..., "sqeuclidean")`.

**Independent confirmation.** Our recomputed distances reproduce the ones
published in the ANN-Benchmarks HDF5 files: SIFT to `1.14e-02` absolute on
values of order `10^4` (float32 epsilon, 1,000,000/1,000,000 inside the band),
NYTimes to `3.58e-07`. This is evidence from outside the repository that the
compaction, remapping and normalisation pipeline is correct end to end. Their
angular convention was determined empirically, not assumed — it is exactly
`1 - cos`, identical to ours; the first guess of `sqrt(2 - 2cos)` was wrong by a
median of 0.48 and was discarded rather than fitted around.

**A tolerance mistake worth recording.** The first version of the cross-check
used pure relative error and failed NYTimes at `3.58e+05`, because `1 - cos`
approaches zero for near-duplicate vectors. Same trap as signed similarities in
score-equality recall, and the same fix: `band = atol + rtol * |x|`.

---

## ADR-0019 — There is no rerank pass; `per_cluster_k` replaces `rerank_candidates`

**Status.** Accepted (before any Phase 2 index code).

**Context.** The Section 6 `AnnConfig` carried `rerank_candidates`: a candidate
pool gathered across probed clusters, then re-scored exactly against the raw
vectors to produce the final top-k.

**Decision.** Drop the exact-rerank stage. `hnswlib.knn_query` is approximate in
*which* candidates it returns but **exact in the distance to each one**
(`test_l2_space_returns_squared_distance`, `test_ip_space_returns_one_minus_ip`).
Merging per-cluster results by those distances therefore already yields the
exact best-k among everything examined. A rerank pass would recompute numbers we
already hold.

**Consequences.**
- `AnnConfig.rerank_candidates` → **`per_cluster_k`**: the retrieval depth
  requested *from each probed cluster* before merging. This is a real knob —
  asking a cluster for only `k` loses results a deeper pull would have surfaced —
  where the old one had no effect at all. `per_cluster_k >= k` is validated.
- `QueryRecord.t_rerank` → **`t_merge`**.
- Had this not been caught, `rerank_candidates` would have entered the Phase 8
  Pareto grid as a dimension with no measurable effect, wasting profiling compute
  and adding a dead axis to the controller's action space.
- Flat clusters (below `min_graph_size`) must return distances in the *same*
  convention as graph clusters, or a merged list sorts wrong across cluster
  types. Asserted by test in Phase 2.
- `per_cluster_k` joins `ef_list` in the Phase 6 profile table, which matters if
  the probe-cost check (Phase 2, check 8) comes back high.

**Memory consequence.** Reranking against a global `base.npy` would have put the
full base array back on the search path. Instead the merge uses the distances
hnswlib already returned, so search touches the centroid graph and one cluster
graph at a time — which is what MobileRAG's memory expression actually describes.

---

## ADR-0014 (amendment) — NYTimes duplicates are real, and they are not small

**Status.** Accepted. Amends the zero-norm finding with a second, independent
quirk of the same dataset.

**The question.** Check 9 measured that 13.55% of NYTimes queries have a
neighbour indistinguishable from the k-th at k=10, one of them with 85 of them.
Two explanations, with opposite consequences:

- **Real duplicate vectors.** Score-equality recall is correct, the band is
  doing its job, and the tie rate is a property of the corpus.
- **Float32 cancellation.** `one_minus_ip` computes `1 - cos`, and float32 eps
  near 1.0 is 1.19e-07, so *any* pair with cosine within that of 1.0 collapses
  to the same stored value whether or not the vectors are the same. Ground truth
  would then be unreliable exactly where it is being used as a threshold, and
  the metric would credit genuinely-worse neighbours — inflating recall in the
  direction that flatters us.

**Settled by measurement, three independent ways.** Byte comparison cannot be
fooled by cancellation:

| dataset | rows that repeat a previous row | groups | largest group | ties at k=10 | max extra |
|---|---|---|---|---|---|
| sift1m  | 14,538 / 1,000,000 (1.454%)  | 14,538 | 2  | 1.37%  | 1  |
| nytimes | 40,793 / 289,761 (14.078%)   | 26,557 | 95 | 13.55% | 85 |
| squad   | 11 / 46,369 (0.024%)         | 11     | 2  | 0.30%  | 1  |

1. **Bytes.** One in seven NYTimes base vectors is a byte-identical repeat.
2. **Full precision.** The worst query (1264) has 95 tied neighbours which are
   *one* distinct byte pattern; recomputing in float64 gives a spread of
   2.22e-16, i.e. float64 rounding on identical inputs, not a real difference.
3. **Correspondence.** The tie rate tracks the duplicate rate on all three
   datasets independently, and the maximum tie count tracks the maximum
   duplicate-group size. Cancellation would predict no such relationship.

**Verdict: the duplicates are real.** Score-equality recall is measuring a true
property of the corpus. Both numbers go in the dataset table.

Note also that the raw pre-normalisation NYTimes array has a largest duplicate
group of **239** — exactly the 239 all-zero rows of the original zero-norm
finding. After dropping them the largest group is 95. The two quirks are
consistent with each other.

**Consequence for Phase 6 — a known constraint, not solved here.** The planned
`recall_proxy` uses score margin (the gap between the k-th and (k+1)-th
candidate) as a runtime confidence signal. On NYTimes that margin is **exactly
zero by construction for one query in seven**, so the proxy is degenerate on
precisely those queries and the bandit would be learning from a structurally
uninformative signal. Recorded now because rediscovering it in Phase 6 as
mysterious single-dataset controller underperformance would cost days. Not
solved now because the shadow-sampling path may cover it and the right fix
depends on Phase 6 numbers that do not exist yet.

**Precision consequence.** Ground-truth distances are stored as **float64**
(8 MB per dataset), computed in float64 throughout. Retrieved distances come
back from hnswlib as float32 by necessity, and the asymmetry is deliberate: it
puts the precision where the recall band needs it, so `atol` covers the float32
noise floor once rather than being swamped by noise on both sides. Recorded per
artifact as `dtype` / `retrieved_side_dtype`. Switching to float64 halved
NYTimes' worst monotonicity violation, from -5.96e-08 to -2.61e-08.

---

## ADR-0020 — An acceptance check a human has to read is not an acceptance check

**Status.** Accepted. Applies to every phase from here.

**Context.** Phase 1 closed with SQuAD having no ground truth at all. Check 1
was "run `smartann data status` and confirm all three are fully prepared" — the
table printed `gt: x`, it was read past, and the phase passed. Nothing caught it
until an unrelated Phase 2 check went looking for a file that should have been
beside it. The cost was never the compute; recomputing took five seconds. The
cost was that only luck surfaced it.

**Decision.**

1. `smartann data status` **exits non-zero** when any registered dataset is
   incomplete, naming the missing artifacts. `DatasetStatus.complete` and
   `.missing()` define completeness in one place.
2. `scripts/phase1_acceptance.py` runs the command as a subprocess and asserts
   its **exit code**. It does not parse the table.
3. Every Phase 2 acceptance check produces a pass/fail verdict of its own.
   Measurements that are genuinely not pass/fail (probe cost, tie structure,
   resident footprint) still print their numbers, but each carries an explicit
   verdict against a stated threshold rather than leaving the judgement to the
   reader.

**Consequence.** `data status` now fails on a fresh checkout with no data, which
is correct — nothing is prepared. The Phase 0 test that asserted exit 0 on an
empty root was updated to assert exit 1 plus a rendered table; its real intent
was "does not crash", which still holds.

---

## ADR-0022 — k-means settings chosen by measurement: `batch_size=10*nc`, `n_init=1`

**Status.** Accepted (Phase 2).

**Context.** The first SQuAD build spent **295 s of its 366 s in k-means**, on
46,369 vectors with nc=215. Extrapolated to SIFT1M at nc=2000 that is hours per
build, and there are six builds. Before making it faster, the question was what
the slowness was buying -- inertia is the k-means objective and the thing that
governs cluster-miss error, so the trade had to be measured against it rather
than against wall-clock alone.

**Measurement.** SQuAD, 46,369 x 384, nc=215, fixed `random_state=42`:

```
  batch    n_init    inertia   relative      sec
   4096         3     8604.7    1.0040x    412.4
   4096         1     8570.2    1.0000x    207.2
   2150 (10*nc) 1     8620.4    1.0059x      4.7
  32768         1     8600.9    1.0036x     38.3
  65536         1     8607.6    1.0044x     34.1
```

**Two findings.**

1. **The entire inertia spread is 0.59%**, while wall-clock spans 88x. Every one
   of these partitions is, for practical purposes, the same partition. There is
   no quality being bought by the slow settings.

2. **`n_init=1` beat `n_init=3` on both time and inertia** (207 s / 8570.2
   versus 412 s / 8604.7), which looks impossible until you read what `n_init`
   does in this estimator. `MiniBatchKMeans` does not run the full fit `n_init`
   times and keep the best; it evaluates candidate *initialisations* on a
   subsample and keeps the one with the lowest inertia there, then fits once.
   A start that looks best on a subsample can converge worse on the full data,
   which is exactly what happened. More inits is not more search -- it is a
   different, noisier way of picking one start.

**Why the small batch is the fast one.** `max_no_improvement=10` counts
*mini-batches* without improvement in the smoothed inertia. Smaller batches
accumulate that counter far sooner in wall-clock terms, so convergence is
detected earlier. The larger batches are not doing better work, only more of it
before noticing they are done.

**Decision.** `kmeans_batch_size="auto"` resolving to `10 * nc` (sklearn's own
guidance), `kmeans_n_init=1`. Both live on `IndexBuildConfig`, and therefore in
the config hash, because they change the artifact: leaving them as function
defaults would let two materially different partitions share one hash, which is
the collision the hash exists to prevent (Section 3, rule 4). The resolved
values are also written into `meta.json` as `kmeans_algorithm`, so an index is
self-describing without its config.

**Caveat, stated rather than buried.** Inertia is a proxy for what actually
matters, which is recall on the built index. It is the right proxy -- it is the
objective k-means optimises and it maps directly onto cluster-miss error -- but
a 0.59% inertia spread was not separately confirmed to be a negligible recall
difference. It is small relative to the recall effects Phase 2 measures (the M
sweep moved recall by 3-6 points), so the risk of it mattering is low, and
acceptance check 1b measures real recall on the artifact these settings produce.

---

## ADR-0023 — The distance-computation model is capped at containment

**Status.** Accepted. Bug fix, not a tune.

**Context.** `n_dist_modeled` estimated graph traversal as `ef * M` with no
upper bound. EcoVector's clusters are small: SIFT's mean cluster holds 500
vectors while `ef=512, M=32` claims 16,384 comparisons — 33x more than the
cluster contains. A graph of `n` elements cannot produce more than `n` distinct
distance computations, so the figure was not an approximation but a wrong number.

**Measured against the true containment ceiling** (total membership of the
probed clusters, the hard maximum):

| config | uncapped model | containment ceiling | overstatement |
|---|---|---|---|
| sift nprobe=8, ef=64 | 20,480 | 4,641 | 4.4x |
| sift nprobe=32, ef=256 | 266,240 | 18,124 | 14.7x |
| sift nprobe=64, ef=512 | 1,052,672 | 35,709 | 29.5x |

**Decision.** Per graph cluster, `min(max(ef_list, per_cluster_k) * M, size)`;
for the centroid phase, `min(max(ef_centroid, nprobe) * M, n_live)`.

**Still a model, still an upper bound.** HNSW revisits nodes, so the true count
is lower again. The cap removes an error of up to 29x; it does not make the
number a measurement. Phase 3 regresses its cost model on this field, which is
why it had to be right *before* Phase 3 rather than discovered during it.

---

## ADR-0024 — Check 4 is posed against ivf_disk; and the per-cluster graphs buy nothing here

**Status.** Accepted, with a finding that was not anticipated.

**Context.** Check 4 originally required EcoVector to reach a recall target in
fewer distance computations than flat HNSW. That asks an IVF index to win the
exact thing it deliberately trades away: flat HNSW holds the whole corpus
resident, EcoVector spends extra comparisons to bound its footprint. The
criterion was ill-posed, not merely unflattering.

**Decision.** The verdict is EcoVector against **ivf_disk** — the same
partition, the same files, the same load path, brute force instead of graphs —
which isolates precisely what the per-cluster graphs contribute and is a claim
the index can win or lose on its merits. flat_hnsw is retained as a reported
measurement carrying no verdict, printed beside the check 5 footprint numbers it
is trading against.

**The finding.** With the model capped, the two variants come out *identical*:

```
sift1m   nprobe=32,ef=256   ecovector 19,323 comps / 0.9537 / 72.20 ms
                            ivf_disk  19,323 comps / 0.9537 / 30.96 ms
         nprobe=64,ef=512   ecovector 36,323 comps / 0.9907 / 144.46 ms
                            ivf_disk  36,323 comps / 0.9907 /  39.35 ms
```

Same recall to four decimals, same modelled work, and **ivf_disk is 2.3x-3.7x
faster in wall-clock**. The per-cluster HNSW graphs contribute nothing at these
cluster sizes and cost real time.

**Why, and when it would change.** With `M=32` and `per_cluster_k=32`, the
effective `ef` is at least 32, so `ef * M >= 1024` — already above SIFT's mean
cluster of 500 and NYTimes' 539. Every per-cluster graph search is therefore
*exhaustive anyway*, and the graph is pure overhead: extra bytes on disk (762 MB
versus 499 MB on SIFT), extra reconstruction time per query, zero recall.

The graphs only earn their place when `ef * M < cluster size`, i.e. with
substantially larger clusters (lower `nc`) or smaller `M`. At `nc=128` on
NYTimes the mean cluster is 2,264 and the graphs would do real work. This is
the same axis as ADR-0025's ceiling question, which is why the two should be
decided together rather than separately.

**Caveat on the wall-clock column.** ivf_disk's per-query times in this run are
noisy and non-monotonic (30.1 ms at nprobe=2, 8.6 ms at nprobe=16), consistent
with first-touch page-cache effects. The distance-computation comparison is
solid; the ivf_disk timings should be re-measured before being quoted.

---

## ADR-0025 — NYTimes' recall shortfall is cluster-miss; ef_centroid is free

**Status.** Measurement recorded. No default changed.

**Context.** Acceptance check 1b measured 0.8751 score-equality recall on
NYTimes at `nprobe=64, ef_list=512`, against a 0.95 threshold. Before arguing
with anything, the shortfall had to be split into cluster-miss error (the true
neighbours were never in a probed cluster) and in-cluster error (they were
there and the search missed them).

**Method.** Pure numpy over `assignments` and the ground-truth ids, no search.
For each query, map its true top-10 to their clusters and count how many fall in
the probed set. That is the maximum recall@10 reachable at that `nprobe` with a
*perfect* in-cluster search. Computed twice: against the exact nearest
centroids, and against what `probe()` actually returns.

```
 nytimes  nc=538, mean cluster 539        sift1m  nc=2000, mean cluster 500
 nprobe   exact    probe      gap         nprobe   exact    probe      gap
      8  0.7151   0.7151   0.0000              8  0.7672   0.7672   0.0000
     16  0.7649   0.7649   0.0000             16  0.8779   0.8779   0.0000
     32  0.8158   0.8158   0.0000             32  0.9515   0.9515   0.0000
     64  0.8709   0.8710  -0.0001             64  0.9879   0.9879   0.0000
    128  0.9232   0.9232   0.0000            128  0.9977   0.9977   0.0000
    256  0.9715   0.9715   0.0000            256  1.0000   1.0000   0.0000
```

**Finding 1: the shortfall is almost entirely cluster-miss.**

| | NYTimes | SIFT |
|---|---|---|
| ceiling at nprobe=64 | 0.8709 | 0.9879 |
| measured (index-overlap) | 0.8603 | 0.9871 |
| **in-cluster loss** | **0.0106** | **0.0008** |
| cluster-miss loss | 0.1291 | 0.0121 |

In-cluster search accounts for 1.1 points of a 13.9-point shortfall. `ef_list=512`
on a 539-element cluster is exhaustive, so there is essentially nothing left for
the in-cluster search to recover. Tuning `ef_list` or `M` cannot fix this.

**Finding 2: the centroid graph is exact, so `ef_centroid` is not a lever.**
The exact-versus-probe gap is `0.0000` at every `nprobe` on both datasets; the
single `-0.0001` is a centroid-distance tie broken differently, where the graph
*beat* brute force. At `ef_centroid=128` the centroid search costs no recall at
all. Direct Phase 6 consequence: `ef_centroid` has no error to trade against and
should be pinned rather than carried in the controller's action space. It also
retires the check 8 contingency — the probe is cheap *and* accurate.

**Finding 3: lower `nc` raises the ceiling, and the price is the exact resource
under test.** Measurement only; nothing built, no default changed.

```
    nc    mean     max  peakMB     np=8    np=16    np=32    np=64   np=128
    64    4528   10559   12.89   0.7675   0.8660   0.9547   1.0000        -
   128    2264    6220    7.59   0.7444   0.8173   0.8897   0.9594   1.0000
   256    1132    4278    5.22   0.7193   0.7850   0.8456   0.9102   0.9673
   538     539    2342    2.86   0.7151   0.7649   0.8158   0.8709   0.9232
```

`nc=128` reaches a 0.9594 ceiling at `nprobe=64` for a `cluster_peak` of 7.59 MB
against 2.86 MB — total predicted resident rising from 4.62 MB to about 9.3 MB,
still inside MobileRAG's own 3.53-9.67 MB range. It is also *cheaper* in
wall-clock, because per-query cost is dominated by the number of cluster loads
(ADR-0026) and `nc=128` needs 64 loads where `nc=538` would need roughly 200.

Note this interacts with ADR-0024: at `nc=128` the mean cluster is 2,264, above
`ef * M = 1024`, so the per-cluster graphs would begin doing real work instead of
being exhaustive overhead. The `nc` question and the "do the graphs earn their
place" question have the same answer axis and should be decided together.

**Open.** The threshold stands at 0.95 and `nc` stays at `auto`. Three honest
options — raise `nprobe`, lower `nc`, or revise the threshold with a recorded
reason — and the measurement above is what a decision should be made against.

---

## ADR-0026 — 83% of per-query time is graph reconstruction, not I/O

**Status.** Finding recorded. Deliberately not fixed.

**Measurement.** Per-query breakdown at `nprobe=32` on SIFT1M, warm page cache:

```
load 59.75 ms | search 2.93 ms | merge 0.11 ms | probe 0.08 ms | total 71.88 ms
```

Across the acceptance run, cluster load is **84.1%** of `t_total` on SIFT and
**84.3%** on NYTimes, at roughly 1.9 ms per cluster. Actual graph search is 4%.

**What this is.** The page cache is warm and this platform cannot evict it, so
the file read is a memcpy. The 1.9 ms is `hnswlib.load_index` allocating and
reconstructing the graph structure in memory. EcoVector as implemented pays
per-query *deserialisation*, not per-query disk I/O.

**Consequence 1 — it changes what Phase 9 measures.** A phone's flash is far
slower than this SSD, so the naive expectation is that load cost grows there.
But the dominant term here is CPU-bound reconstruction, and a phone's CPU is
also slower. Phase 9 must separate the two — read time versus reconstruction
time — or it will produce a single load number that cannot be attributed to
either, and no cost model can be fitted to it.

**Consequence 2 — it inflated check 8's denominator.** The probe fraction
`t_centroid / t_total` was reported as p50 0.0037, comfortably HEALTHY, but part
of that comfort came from an unrelated cost sitting in the denominator. With
cluster load excluded the figure is p50 **0.0237** (p95 0.0298) on SIFT and
0.0235 on NYTimes. Both are still far inside the 0.15 HEALTHY band, so the
verdict is unchanged — but the check now reports both numbers and takes its
verdict from the load-excluded one, which is the figure not flattered by
something else being expensive.

**Not fixed: no cluster cache.** Caching reconstructed graphs across queries
would remove most of this cost, and would trade directly against the bounded
resident footprint that checks 5 and 6 exist to defend. That is a design change
with its own consequences, not a performance tweak, and it needs its own
decision rather than being slipped in as an optimisation.

**Related.** ADR-0024 shows the per-cluster graphs currently buy no recall at
these cluster sizes while costing this reconstruction time, so "drop the graphs"
and "cache the graphs" are competing answers to the same measurement.

---

## ADR-0027 — SIFT1M's `nc` is pinned to 2000, not `auto`

**Status.** Accepted retroactively. The decision was made and reasoned in Phase
1; it was recorded only in the dataset registry's `notes` field, which is not
where decisions live.

**The discrepancy.** `default_nc="auto"` resolves to `round(sqrt(n_base))`,
which is 1000 for SIFT1M — but SIFT built at nc=2000. The registry carries
`default_nc=2000` explicitly for SIFT and `"auto"` for NYTimes (538) and SQuAD
(215), so the two datasets are governed by different rules.

**The reason, as recorded in the registry notes.** MobileRAG's Figure 11 sweeps
`Nc` over 1000-4000 on SIFT, and 2000 sits in the middle of that range; pinning
it makes our EcoVector build directly comparable to the paper's. `sqrt(N)` is a
generic heuristic, and matching the baseline we are measuring against is worth
more than internal consistency with a rule of thumb.

**Why this ADR exists anyway.** A per-dataset override that changes cluster
count by 2x is exactly the kind of thing that later looks like a bug, and a
reason living in a string field of a registry entry is not discoverable from the
decision log. It is now in both places. It also matters for reading Phase 2's
results: SIFT's mean cluster is 500 rather than the ~1000 that `auto` would
have produced, which feeds directly into ADR-0024's finding that `ef * M`
exceeds the cluster size and the per-cluster graphs are therefore exhaustive.

**Consequence for the open `nc` question.** Any decision to change NYTimes' `nc`
should say whether SIFT's pin stays. They are independent knobs today, and the
comparability argument applies only to SIFT.

---

## ADR-0025 (amendment) — the `nc` finding was an artifact; the axis was wrong

**Status.** Supersedes finding 3 of ADR-0025. The original conclusion was wrong.

**The error.** The `nc` table held `nprobe` fixed, but `nprobe` is not a
fixed-cost axis across `nc`. At nc=538 a probe of 64 touches 64 x 539 = 34,470
vectors (11.9% of the corpus); at nc=128 the same `nprobe` touches 64 x 2,264 =
144,896 (50%). The lower-`nc` rows looked better because they examined roughly
4x more data. Two rows were worse still: nc=64 at nprobe=64 and nc=128 at
nprobe=128 probe *every* cluster, so their 1.0000 ceilings were brute force with
extra steps rather than operating points.

**Recomputed against corpus coverage** — vectors contained in the probed
clusters — which is the shared-cost axis:

```
    nc  peakMB     5.0%    10.0%    12.5%    20.0%    30.0%    40.0%    50.0%
    64   12.89   0.6209   0.7114   0.7434   0.8162   0.8740   0.9139   0.9428
   128    7.59   0.6951   0.7771   0.8020   0.8516   0.8971   0.9293   0.9515
   256    5.22   0.7440   0.8134   0.8336   0.8780   0.9160   0.9429   0.9609
   538    2.86   0.7872   0.8440   0.8619   0.8983   0.9307   0.9542   0.9691
  1024    2.14   0.8225   0.8682   0.8848   0.9165   0.9443   0.9630   0.9706
```

**The ordering reverses completely, at every coverage level.** A *finer*
partition localises NYTimes better per unit of data examined, monotonically from
nc=64 to nc=1024 — and it simultaneously costs *less* memory, since
`cluster_peak` falls from 12.89 MB to 2.14 MB. Both axes point the same way, the
opposite way from the original reading. The corrected finding is that coarsening
`nc` is strictly worse for NYTimes on both recall-per-work and memory.

The wall-clock claim in the superseded section ("nc=128 is also cheaper because
64 loads beat 200") is withdrawn: it was never measured, and it now looks
backwards on the recall side.

**But finer is not free, because load cost is per-cluster, not per-vector.**
ADR-0026 measured ~1.9 ms per cluster load dominating query time. Reaching 40%
coverage needs roughly 192 loads at nc=538 and roughly 410 at nc=1024 — about
365 ms against 780 ms. So the partition trade is genuinely two-sided: finer wins
on recall-per-vector-examined and on memory, coarser wins on recall-per-second.
Which one governs depends entirely on whether cluster loads stay expensive,
i.e. on the cache decision. The three open questions are one question.

**The number that actually matters.** At nc=538, the ceiling reaches:

```
 nprobe   coverage   ceiling          ceiling 0.95 -> 38.0% coverage (nprobe ~185)
     64     13.94%    0.8709          ceiling 0.96 -> 43.9% coverage (nprobe ~216)
    128     26.86%    0.9232
    192     39.41%    0.9533
    256     51.58%    0.9715
```

**NYTimes needs ~40% of the corpus examined to reach recall@10 of 0.95, with a
perfect in-cluster search.** That is a property of 256-dimensional angular data
against IVF partitioning, not a tuning failure and not something `nc`, `M` or
`ef_list` can repair. An IVF index that must touch 40% of the corpus is barely
an index on this dataset. SIFT reaches the same ceiling at nprobe=32 of 2000,
i.e. under 2% coverage — a 20x difference in how well the data partitions.

This is the finding to report about NYTimes, and it is what any decision on the
0.95 threshold should be made against.

---

## ADR-0024 (amendment) — the timing half re-measured; the claim strengthens

**Status.** Replaces the wall-clock numbers in ADR-0024. The recall half is
unchanged and was never in doubt.

**Why it needed redoing.** The original check 4 timed each config once, with no
warmup. It reported 27.076 ms and 8.589 ms for two configs doing provably
identical work (10,724 modelled comparisons each) -- a 3x spread at zero work
difference, which is first-touch page-cache behaviour being read as a
performance result. The ADR asserted "2.3-3.7x faster" while its own caveat said
the numbers were unreliable, so it contradicted itself.

**Harness fix.** Recall and timing are now measured separately, because they
need different things: recall over every query once (breadth), timing over 50
queries with a warmup pass and 5 repeats taking the per-query minimum, reported
as median with inter-quartile range (repetition). A wall-clock figure without a
dispersion estimate beside it is not quotable.

**Re-measured, sift1m:**

```
                       config  dist_comps   recall  ms median      IQR
 ecovector      nprobe=2,ef=32      3,087   0.4473      4.507    0.393
                nprobe=32,ef=256   19,323   0.9537     62.901    5.457
                nprobe=64,ef=512   36,323   0.9907    139.114    8.958
 ivf_disk       nprobe=2,ef=32      3,098   0.4490      1.060    0.112
                nprobe=32,ef=256   19,323   0.9537     15.470    1.295
                nprobe=64,ef=512   36,323   0.9907     30.905    2.334
```

**Re-measured, nytimes:**

```
 ecovector      nprobe=2,ef=32      1,918   0.6330      5.423    2.040
                nprobe=64,ef=512   41,008   0.8820    147.977   10.623
 ivf_disk       nprobe=2,ef=32      2,124   0.6380      1.887    0.931
                nprobe=64,ef=512   40,992   0.8823     45.436    4.615
```

**The dispersion is now sane** -- IQRs of 0.1-10 ms on values of 1-148 ms,
proportional to magnitude -- and both variants are monotone in work, which the
single-shot version was not. The 3x-at-identical-work pathology is gone.

**The claim strengthens rather than weakens.** ivf_disk is **4.1x** faster at the
matched operating point (15.470 ms against 62.901 ms), not the 2.3x originally
claimed, and the ratio is stable at 4.2-4.5x across the whole sweep on both
datasets. The earlier figure understated the gap because the noise it rested on
happened to run the wrong way.

**Combined finding, now on solid footing both halves.** At these cluster sizes
the per-cluster HNSW graphs deliver identical recall to brute force -- four
decimal places, seven configurations, two datasets, two different algorithms --
and cost roughly 4x the wall-clock plus 263 MB of extra disk on SIFT. Two
independent algorithms agreeing that precisely is strong evidence both are
exhaustive within each cluster, which is exactly what `ef * M >= 1024` against a
mean cluster of 500 predicts.

---

## ADR-0026 (amendment) — the reconstruction claim is now measured, not inferred

**Status.** Confirms ADR-0026 by direct measurement. Adds a caveat the original
did not have.

**What was wrong with the original argument.** ADR-0026 concluded "reconstruction,
not I/O" from an inference: the page cache is warm, therefore the read is cheap,
therefore the cost must be CPU. Sound reasoning, but the whole cache decision
rested on it, so it needed testing rather than deriving.

**Method.** Load one representative cluster file 60 times in a single process,
timing each. By load 50 the page cache is certainly warm, so any cost remaining
is not I/O. Separately, read the same file with no parsing at all, which
isolates the byte-read component.

```
sift1m   cluster 98,  466 vectors, 361 KiB
  raw file read (warm)  0.064 ms =  1.2% of load time
  reconstruction        5.345 ms = 98.8%

nytimes  cluster 359, 472 vectors, 601 KiB
  raw file read (warm)  0.438 ms =  5.2% of load time
  reconstruction        7.979 ms = 94.8%
```

**Confirmed.** Reading the bytes is 1-5% of the cost; 95-99% is
`hnswlib.load_index` rebuilding the graph structure in memory. The cache
decision's premise holds: a cache would remove the great majority of the ~84% of
query time that cluster loading consumes.

**Eviction is not achievable on this platform, stated plainly.** win32 exposes
no page-cache eviction primitive -- no `posix_fadvise`, and `EmptyWorkingSet`
requires elevation. These numbers are **warm-only**. What they establish: the
cost that remains when the file is certainly cached, which is a floor. What they
do not establish: what a genuinely cold first touch adds on this SSD, or on a
phone's flash, where the read component will be far larger relative to CPU.
Phase 3.5 supplies the cold number and it can only move the ratio toward I/O,
never away from it -- so "reconstruction dominates" is safe as a lower bound on
this machine and must be re-checked on the phone.

**An anomaly worth recording rather than hiding.** The per-load absolute here
(5.4-8.4 ms) is higher than the ~1.9 ms derived from `t_cluster_load` in the
search loop, load 1 is *faster* than the steady state (ratios 0.79x and 0.40x),
and NYTimes is noisy (IQR 6.656 against a median of 8.417). That pattern is the
opposite of a cold-cache effect and looks like heap fragmentation accumulating
across repeated allocate-and-free cycles of graph structures. It means the
absolute per-load figure depends on allocator state and should not be quoted as
a constant. The read-versus-reconstruct *ratio* is unaffected by it, and the
ratio is what the decision turns on.

---

## ADR-0028 — The per-cluster HNSW graphs are removed; no M makes them pay

**Status.** Accepted, by a pre-committed rule applied to Measurement A.

**The rule, fixed before the data.** Keep the graphs if some `M` gives EcoVector
a lower median wall-clock than `ivf_disk` at matched recall (within 0.005) on
either dataset; drop them if no `M` wins.

**Measurement A.** `nc`, `ef_construction` and seed held fixed; `M` swept over
{4, 8, 16, 32}; `ivf_disk` included as the M-to-zero limit.

```
sift1m      M   nprobe/ef   recall      dist   ms med     load   search   diskMB
            4      64/512   0.9927    35,602  544.293  439.606   22.309    550.1
            8      64/512   0.9953    36,089  437.816  352.524   22.583    579.2
           16      64/512   0.9953    37,066  443.281  356.393   23.350    640.1
           32      64/512   0.9953    37,066  131.654  110.408    6.679    762.1
      ivf(M->0)    64/512   0.9953    37,066   43.351   31.946    5.030    499.4

nytimes    32      64/512   0.8680    40,441  528.816  432.862   29.021    362.8
      ivf(M->0)    64/512   0.8680    40,435  222.119  182.886   15.604    286.6
```

**No `M` wins anywhere.** At every matched-recall pair on both datasets,
`ivf_disk` is faster. Lowering `M` does make the graph do partial rather than
exhaustive work -- distance computations fall from 37,066 to 35,602 on SIFT --
but recall falls with it (0.9953 to 0.9927) and wall-clock gets *worse*, not
better. The graphs also cost 263 MB of extra disk on SIFT.

**An order confound, checked rather than assumed.** In Measurement A the
variants got faster with position and `ivf_disk` ran last, and a 4x
discontinuity between M=16 and M=32 is not a plausible property of graph degree.
A control re-ran the two variants with `ivf_disk` **first**, interleaved over
four rounds in one process: 51.5 / 53.1 / 51.3 ms against 194.6 / 186.6 / 203.0
ms, a **3.80x** ratio matching the ~4x from an earlier independent run. The
advantage is real; the M-to-M absolute times in Measurement A are drift-affected
and should not be quoted.

**The confound the ruling asked about: it won on load, not on search.** At SIFT
`nprobe=64`, `ivf_disk` saves 78.5 ms of cluster load and only 1.7 ms of search.
So the honest statement is **"`.npy` maps faster than hnswlib graphs
reconstruct"**, which is a finding about serialisation, not about search
algorithms. "Sparser graphs search faster" would have been a finding about the
algorithm and is *not* what was observed.

**Decision.** `min_graph_size` defaults to a value above any cluster size, so
every cluster is flat. The graph path is retained in the code and reachable by
lowering `min_graph_size`, which keeps the ablation available -- this comparison
is itself a result the paper should carry.

**Reframed around MobileRAG's own cost model, which predicts this result.**

Dropping the per-cluster graphs does not produce a generic IVF index. It
produces **IVF-HNSW**, a baseline the paper defines and positions itself
against. The paper's memory expressions differ by exactly one term:

```
IVF-HNSW    4Nc(d + M'/(1-p0)) + 8N + 4d
EcoVector   4Nc(d + M'/(1-p0)) + 8N + 4(d + M'/(1-p0))
```

That last term is one cluster's graph. Remove it and EcoVector *is* IVF-HNSW.
Figure 9 puts IVF-HNSW at 2.0 mJ against EcoVector's 1.2 mJ on SIFT, a 40%
power gap attributed to that term.

**And the paper states the condition under which the graph pays.** Its
search-time models are

```
IVF-HNSW    ef_c * M' + n_P * (N/Nc)
EcoVector   ef_c * M' + n_P * ef_L * M'
```

so EcoVector beats IVF-HNSW **iff `ef_L * M' < N/Nc`** -- the per-cluster graph
is worth walking only when walking it costs less than scanning the cluster.
Against our builds:

| dataset | N/Nc | ef_L * M' | condition |
|---|---|---|---|
| sift1m  | 1,000,000 / 2000 = 500 | >= 32 x 32 = 1024 | **violated by 2.0x** |
| nytimes | 289,761 / 538 = 539    | >= 32 x 32 = 1024 | **violated by 1.9x** |

**MobileRAG's own arithmetic predicts our measurement.** This is not a
contradiction of the paper. We built outside the regime where EcoVector's
distinguishing component can pay for itself, and inside that regime EcoVector
reduces to the baseline it is measured against.

The finding is therefore not "the graphs did not help for us" but the sharper:
*the default configuration -- `Nc = sqrt(N)` on NYTimes, `Nc = 2000` on SIFT,
with M=32 -- sits outside the condition `ef_L * M' < N/Nc` that EcoVector's own
search-time model requires, so its distinguishing component cannot pay for
itself there and the index degenerates to IVF-HNSW.*

**Instantiation caveat, now narrower.** The recall half is
implementation-independent: two different algorithms agreeing to four decimal
places across seven configurations and two datasets is evidence about cluster
sizes, not about a library. The wall-clock half is hnswlib-specific -- the 4x is
`.npy` mapping against hnswlib deserialisation, and a different serialisation
could change it. The *condition* is the paper's, not ours, and holds regardless
of either.

## ADR-0029 — No cluster cache; memory-map the flat clusters instead

**Status.** Accepted.

**Not because the benefit is small.** Measurement B (ADR-0026 amendment) shows
95-99% of load time is reconstruction, so a cache would remove most of the ~84%
of query time that loading consumes. The benefit is large. A cache is the wrong
instrument for it.

**Why a cache is wrong here.** It makes query cost depend on cache contents,
which depend on query locality and history. Phase 3's cost model would then be
fitting a function of the query *sequence* rather than of the query, and every
latency and energy prediction would be conditional on a warm-up state the
controller cannot observe. The dominant cost term would change from
`n_cluster_loads` to `n_cache_misses` -- different random variables with
different distributions -- so a model fitted on one is not merely imprecise for
the other, it is invalid.

**mmap instead.** With the graphs gone (ADR-0028), flat clusters are stored as
two plain `.npy` files -- vectors and ids -- rather than one `.npz`, because
`np.load` can memory-map a `.npy` and cannot map inside a zip archive. Mapping
gives zero-copy access with no reconstruction at all: most of what the cache
would have bought, while keeping cost a function of bytes touched rather than of
history, and handing eviction policy to the OS, which is better at it and is not
our variable.

**A footprint consequence worth stating.** Mapped pages are file-backed, so they
do not count toward anonymous RSS. Check 5 already reports anonymous separately
from total RSS, so the measurement infrastructure needed no change. A
file-backed footprint is arguably *more* faithful to a phone than an anonymous
one, since the OS can evict it under memory pressure -- which is precisely the
pressure this project exists to adapt to.

**If the graphs ever come back**, mmap is unavailable for them (hnswlib
deserialises into its own structures) and this decision should be reopened with
the M-sweep numbers in hand.

---

## ADR-0030 — Standing rule: interleave timing comparisons, and report dispersion

**Status.** Accepted. Standing rule, alongside the test-isolation rule
(ADR-0015).

**A wall-clock comparison between two variants must interleave them within a
single process, alternating on every repeat, and must report a dispersion
estimate beside every median.**

**Why.** A blocked sweep -- all of variant A, then all of variant B -- measures
the machine's drift as much as it measures the code. This project has now hit
that twice:

1. Single-shot timings with no warmup reported 27.076 ms and 8.589 ms for two
   configurations doing provably identical work (10,724 modelled comparisons
   each). A 3x spread at zero work difference.
2. A blocked M sweep produced a 4x discontinuity between M=16 and M=32 that is
   not a plausible property of graph degree, with the fastest variant happening
   to run last. An interleaved control put the true ratio at 3.80x.

In both cases the number looked like a result and was partly an artifact of
position in the run. Drift is not noise that averages out: it is monotone within
a run, so it biases whichever variant runs late.

**What it does not fix.** Interleaving equalises drift between variants; it does
not make absolute timings portable. Absolutes still depend on allocator state,
thermal headroom and page-cache contents, which is why the dispersion estimate
is mandatory rather than decorative -- an IQR comparable to the median is a
signal not to quote the median.

**Applied.** `_sweep_cost_recall` in `scripts/phase2_acceptance.py` interleaves
every variant within each config. The regime test does the same. Any future
comparison that cannot interleave -- because the variants are separate processes,
say -- must say so and must carry a control.

---

## ADR-0028 (regime test) — the graphs lose inside their own condition too

**Status.** Closes ADR-0028. The conclusion hardens; it does not change.

**The question a reviewer would ask.** ADR-0028 found the graphs do not pay in
the default configuration, and explained it with MobileRAG's own condition
`ef_L * M' < N/Nc`, which the default violates. The obvious next question is
whether they pay where the condition *holds*. "Out of scope" is a weak answer
when the test is one build pair.

**Setup.** SIFT only, `nc=250` so `N/Nc = 4000`, against `ef_L * M' = 1024`
(M=16) and `2048` (M=32) -- both comfortably inside. Interleaved timing per
ADR-0030. A probe, not a proposed default: ADR-0027's `nc=2000` pin stands.

```
 nprobe          eco M=16                  eco M=32                  ivf_disk
        recall   dist      ms       recall   dist      ms      recall   dist      ms
      2 0.6487   2,298   24.67      0.6493   4,343   24.66     0.6507   8,784    9.41
      4 0.8173   4,346  132.94      0.8183   8,436  143.70     0.8197  17,298   58.07
      8 0.9217   8,442  101.23      0.9230  16,625  102.74     0.9243  34,071   34.77
     16 0.9780  16,634  190.87      0.9790  32,998  200.78     0.9807  67,287   63.68
```

**The condition works exactly as the paper says it should.** Inside the regime
the graphs really do cut distance computations -- 3.8x at M=16, 2.0x at M=32,
consistently across every `nprobe`. MobileRAG's cost model is correct about the
quantity it models.

**And the graphs still lose.** Wall-clock is **2.3x to 3.0x worse** with graphs
at every operating point, and recall is very slightly worse too (ivf_disk is
marginally highest in all four rows -- the graph's approximation costs a little
accuracy that the full scan does not).

**Why: the paper's cost model has no reconstruction term.** Its search cost is
distance computations plus disk I/O
(`n_seek * (T_seek + T_cmd + n_byte * T_transfer)`). Nothing in it charges for
deserialising a graph structure, which measurement B put at 95-99% of load time
(ADR-0026 amendment). So EcoVector wins the quantity the model counts and loses
the quantity that matters, because the dominant real cost is not in the model at
all. That is a gap in the model rather than a violation of it, and it is the
same omission measurement B found from the other direction.

**Verdict.** The graphs stay removed. The finding is now stronger than "the
default configuration is outside the regime": *even inside the regime where
EcoVector's own cost model predicts its distinguishing component should pay, it
does not, because the model omits the cost that dominates.*

**Caveat, unchanged and still narrow.** The reconstruction cost is hnswlib's.
A serialisation format that could be mapped rather than parsed might reverse the
wall-clock result while leaving the distance-computation result intact. What is
implementation-independent: the condition itself, the 3.8x reduction in distance
computations, and the recall equivalence.

**And a note on which side is the faithful one.** Our `ivf_disk` memory-maps its
flat clusters, so warm "loading" is a page-table operation. The paper's model
charges for I/O and distance computations but never for a memcpy or a parse, so
an mmap'd scan path is *closer* to what the paper models than a read-and-copy
path would be. The divergence from the paper's model is on the **EcoVector**
side -- hnswlib's `load_index` introduces a per-query cost the model does not
have -- not on the baseline side. The precise description of what won is
"IVF-HNSW with an mmap'd scan path".

**A measurement-quality note.** Absolute timings in the table are not monotone
in `nprobe` (the `nprobe=4` row is slower than `nprobe=8`), which is the same
machine drift ADR-0030 documents. Interleaving equalises drift *between*
variants within a row, which is what the comparison needs, but not *across*
rows. The within-row ratios are the trustworthy quantity and they are stable at
2.3-3.0x; the absolute column should not be quoted.

---

## ADR-0031 — The missing materialisation term, and a contradiction inside MobileRAG's design

**Status.** Accepted. This is the sharpest form of the ADR-0028 finding and the
one the paper section should be written around.

**The dose-response confirms the model's mechanics.** Inside the condition, at
`nc=250` on SIFT:

| M | `ef_L · M'` | distance computations vs ivf_disk |
|---|---|---|
| 16 | 1024 | **4.0x fewer** |
| 32 | 2048 | **2.0x fewer** |

Halving `M` halves the distance computations, exactly as `ef_L · M'` predicts.
MobileRAG's search-time model is *mechanically correct* about the quantity it
counts. What it predicts from that quantity -- that EcoVector therefore wins --
is false by 2.3-3.0x on wall-clock at every operating point measured.

**The missing term, stated precisely.** The paper prices cluster access as pure
I/O: `n_seek · (T_seek + T_cmd + n_byte · T_transfer)`. There is no term for
turning bytes on disk into a *searchable* structure. That omission is not
uniformly wrong -- it is exactly right for one storage format and badly wrong for
another:

* **Flat array**: the stored form *is* the traversable form. The materialisation
  term is genuinely zero, and the paper's model is exact.
* **HNSW graph**: the stored form must be parsed into nodes and link lists
  before a single comparison happens. Measured at **95-99%** of access cost
  (ADR-0026 amendment).

**The contradiction is inside MobileRAG's own design, not between the paper and
our build.** Section 3.2.3 requires unloading the cluster graph after every
query -- that is what bounds the resident footprint, which is the paper's central
claim. But an access model with no materialisation term is only valid if the
serialised form is traversable in place. **Unload-per-query and pure-I/O access
cost cannot both hold unless the on-disk format is directly searchable.** The
memory design and the latency model contradict each other, and our measurement
is where that contradiction surfaces.

**The constructive version, which is the useful half.** A directly-traversable
on-disk graph format would satisfy both requirements at once: unload-per-query
keeps the footprint bounded, and zero materialisation makes the pure-I/O cost
model correct. That is precisely why our mmap'd flat clusters win -- a flat array
is traversable in place, so it already has the property the design needs.

The finding is therefore not "graphs are bad". It is: **a format that must
survive unload-per-query has to be traversable without materialisation, and
hnswlib's serialisation is not.** An implementation with a mappable graph format
could plausibly recover EcoVector's advantage, and that is a concrete direction
rather than a complaint.

**Scope of the instantiation caveat, now narrow.** The 2.3-3.0x wall-clock
figure is hnswlib-specific. The *argument* is not: it follows from the paper's
own equations plus the measured fact that materialisation is non-zero for any
format requiring a parse. The distance-computation dose-response and the recall
equivalence are implementation-independent.

**A note on measurement quality that supports the method.** The `nprobe=4` row
is non-monotonic in absolute time (132.94 ms against 101.23 ms at `nprobe=8`),
and it moves that way for **all three variants simultaneously** -- a system
disturbance during that block, not a variant effect. The eco/ivf ratios across
the four rows are 2.62, 2.29, 2.91, 3.00: stable while the absolutes are
visibly corrupted at one point. That is exactly what within-config interleaving
is for (ADR-0030). The ratios are the result; the absolute column is
drift-affected and is not quoted.

---

## ADR-0032 — A control loop must act no faster than its feedback delay

**Status.** Accepted. Standing rule for every policy in this project, present
and future.

**The rule.** *Control actions must be paced no faster than the feedback delay
of the signal they are read from. Verify this explicitly for any controller
before tuning anything.*

**What it cost to learn.** The first AIMD policy decided once per query while
reading a 50-sample windowed p95. An action was therefore invisible to its own
signal for roughly 50 queries, so the controller acted on up to fifty superseded
decisions. Multiplicative decrease under a violated stability condition ratchets
to the floor by construction: seven consecutive cuts take effort from 1.0 to
below 0.01, and seven queries pass long before fifty do.

The result was a controller that collapsed to `nprobe=1` in **every phase of two
independent experiments**, including a target-step run with no external
disturbance at all — giving up half its recall while missing its latency target
*more often than the tuned static baseline it was supposed to improve on*.

**TCP paces `cwnd` once per RTT, not once per ACK, precisely because the RTT is
the feedback delay.** The first implementation had AIMD's arithmetic without its
clock.

**Confirmed by a pre-declared test.** `control_period_s = 4.05` (50 samples ×
the measured quiet latency at `nprobe=64`), declared and committed before the
code existed, one run, one setting. Quiet-phase effort changes fell from roughly
150 to **4**, and the controller settled at `nprobe=20` with recall 0.9115
against the static baseline's 0.8806.

**Why this is a design rule and not a tuning preference.** It is checkable from
the estimator's definition before any policy code is written, it applies to
every controller regardless of its control law, and it makes the fix
pre-registrable rather than found by search. A bandit is not exempt: its reward
is a delayed observation in exactly the same sense, and a bandit updating faster
than its reward window will exhibit the same pathology with a different shape.

**A second-order consequence, recorded because it is easy to miss.** A window
fixed in *samples* has a delay in *time* that varies with the controller's own
output — 4.07 s at `nprobe=64` against 0.055 s at `nprobe=1`, a 74× swing across
the action space. A controller whose feedback delay is a function of its own
action is a harder object than one with a fixed delay, and this argues for a
time-based window independently of the ratchet.

---

## ADR-0033 — The latency model holds within its regime and cannot extrapolate; the state feature is censored

**Status.** Accepted as the Phase 3 result. Not a failure to be fixed by more
features; a boundary to be respected and reported.

**The result.** Fitted per-stage on 21,388 quiet queries, validated on 44,067
held-out contended ones:

```
                          quiet (fitted)    contended (held out)
total latency MAPE            14.75%              35.87%
                          PASS (<15%)         FAIL (<15%)

per stage, contended:   t_centroid 33.7%   t_cluster_load 28.7%
                        t_cluster_search 31.2%   t_merge 71.9%
```

Every stage **underpredicts** under contention. `t_cluster_load` predicts 2.08 ms
against 4.89 ms measured.

**First diagnosis, and the one principled correction made.** A work-only model
cannot express latency, because latency is work divided by available machine.
Between phases `cpu_util` went 0.418 → 1.000 while `t_cluster_load` went
4.111 → 10.597 ms **for identical work**. `dev_cpu_util` was therefore added to
every CPU-competing stage. This was a specification correction rather than a
search: the variable was named from the mechanism before the refit, it is the
same quantity the controller already reads, and it was validated on held-out
data rather than on the data that motivated it.

**It barely helped: 36.44% → 35.87%. That is the finding.**

**Why, measured rather than assumed.** `cpu_util` is *censored*. Across the
contended data, **12,469 rows sit at exactly 1.000**, and within those rows
`t_cluster_load` spans **0.67 ms to 27.59 ms — a 41x spread while the feature
does not move at all**. A feature pinned at its ceiling cannot express how far
past the ceiling the machine is, and no coefficient on it can recover that.

**A second mechanism, also measured.** The contended-phase median `cpu_util` is
0.446, *lower* than one would expect from a saturated machine. The telemetry
sampler is a background thread at 10 Hz, and under full contention it is itself
starved and scheduled late. So the feature is not merely censored at the top: it
is **undersampled precisely when it matters**. This was predicted when the
monitor was written and is now measured.

**Consequences, stated rather than patched.**

1. **The Phase 3 latency criterion is met within the regime it was fitted on and
   is not met outside it.** Reported as two numbers, never as one. A single
   averaged MAPE over both phases would have read as roughly 25% and hidden that
   the model is excellent in one regime and unusable in the other.
2. **Phase 6 must not use this model to predict contended latency.** It is sound
   for choosing effort on a quiet machine and unsound for anticipating a
   disturbance. That is precisely why the AIMD governor is *reactive* -- it
   measures latency rather than predicting it -- and this result is an argument
   for keeping the controller reactive rather than model-based.
3. **The fix is a load metric that does not saturate**, such as run-queue length
   or a load average, not another coefficient on a censored feature. Recorded as
   a Phase 3.5 item because the phone needs the same treatment and its
   `/proc/loadavg` is readable where Windows' is emulated.
4. **`cfg_ann_ef_centroid` is unidentifiable**: constant at 128 in every recorded
   run, so `t_centroid`'s dependence on it cannot be fitted. The first fit
   returned R2 = 0.0000 -- an intercept wearing a regression's clothes. Reported
   in `UNIDENTIFIABLE` rather than silently fitted. ADR-0025 found `ef_centroid`
   to be free, so it may never need to vary.

**Energy: specified, implemented, deliberately not fitted.** The model is
`E = V·[I_cpu·t_cpu + I_io·t_io]` with `t_cluster_load` on the **CPU** side
(ADR-0026 amendment, ADR-0031). `assert_energy_measurable()` refuses to fit or
record it on this platform and the fit script reports BLOCKED rather than a
number. Phase 3 closes with **no energy MAPE at all**, which is the honest
outcome: a criterion that cannot be evaluated here is not marked PASS.

**Addendum — two coefficients that must not be read separately.** In the
recorded data `n_cluster_loads` and `bytes_read` correlate at **r = 0.966**
(design-matrix condition number 4.9e6), because EcoVector's clusters are of
similar size. Their *combined* effect is well determined — `t_cluster_load`
reaches R² = 0.99 — but the individual coefficients are unstable, and quoting
"0.69 ms per cluster load" as a standalone constant would be over-reading the
fit. This surfaced from a unit test whose synthetic `bytes_read` was exactly
proportional to `n_cluster_loads`: the fit put all the weight on one feature and
none on the other, which is what perfect collinearity does and what near-perfect
collinearity does more quietly.

---

## ADR-0034 — SCR reduces generation cost, and Phase 5 cannot measure whether that is worth it

**Status.** Accepted. Phase 5 delivers the mechanism and its cost; the benefit is
not measurable until Phase 7 exists.

**What was built.** Sentence windows over a retrieved chunk, scored against the
query embedding, top-`k` kept above a threshold, neighbours re-attached,
overlapping spans merged. Driven entirely by the five fields already on
`ScrConfig` -- no new knobs, because those fields are the Section 6 contract.

**Three properties the implementation is built around.**

1. **Disabled means untouched.** `enabled=False` returns the input byte for byte
   *and does not call the encoder at all*. The C2 ablation compares against this
   baseline, so an "off" path that still normalised whitespace would measure the
   normaliser as part of the reduction, and one that still embedded would charge
   the ablation for a cost it does not incur.
2. **The decision is reproducible from stored data.** Kept spans are sentence
   indices into the source, so a `QueryRecord` is enough to reconstruct exactly
   what was dropped without re-running the embedder.
3. **Overlapping kept windows are merged.** Windows overlap by construction, and
   emitting a sentence twice would inflate the very token count SCR exists to
   reduce -- a bug that would have made aggressive settings look *worse* than
   mild ones.

**A deliberate refusal to return nothing.** When the threshold excludes every
window, the best window is kept anyway. An empty context would surface downstream
as a generation failure rather than as the retrieval miss it actually is, and
those need different fixes.

**The scores are SIMILARITIES, not distances.** Larger is better here, which is
the opposite of the convention the index stores (ADR-0018). Conflating them would
rank every window backwards while still producing plausible-looking output, so
the direction is asserted by test rather than left to a comment.

**What Phase 5 cannot deliver, stated rather than implied.** SCR's value is
answer quality per token, and answer quality needs a generator. Phase 7 is
blocked on the MSVC toolchain (confirmed: no `llama-cpp-python` wheel exists for
any version on this platform). So Phase 5 reports **reduction ratio and token
counts only** -- there is no EM, no F1, and no claim that reduction preserves
answer quality. `approx_tokens()` is characters over a documented constant, named
`approx_` precisely so nothing downstream reads it as a token measurement.

**Consequence for C2 and C3.** The mechanism now exists, so the *action space*
for C3 is genuinely two-dimensional (`nprobe` x SCR aggressiveness) for the first
time -- ADR-0033 having reduced C1 to one dimension on its own. But C2's claim,
"adaptive SCR aggressiveness improves the quality-cost trade-off", cannot be
evaluated until a generator exists to measure quality. Until then C3 can be run
as a cost-only study and must be labelled as one.

---

## ADR-0035 — SCR's applicability is a property of the chunking, not of the reducer

**Status.** Accepted. Found by a smoke run before the real experiment, which is
the only reason it is a configuration note rather than a wasted result.

**What happened.** The first matrix smoke run reported **0.0% reduction** with
SCR enabled, while still paying its full embedding cost -- 662 ms per query
against 14.8 ms with SCR off. The reducer was not broken; it had nothing to work
with.

**Measured, not guessed.** Over 400 SQuAD chunks:

```
chunk length      p50 385 chars   p95 683
sentences/chunk   p5 2   p50 3   p95 3   max 4
fraction with <= 3 sentences: 0.993
```

At `sliding_window_size=3`, a 3-sentence chunk yields **exactly one window**, so
"keep the top 2 windows" keeps everything. Reduction is zero by construction, and
no threshold or extension setting can change that.

**The general statement.** SCR selects *among* windows, so it can only reduce
when a chunk contains more windows than are kept: `n_sentences > window_size`
must hold with room to spare. Our Phase 1 chunking targets roughly 3-sentence
chunks, and MobileRAG's SCR implicitly assumes chunks substantially larger than
its windows. **The two were set independently and are incompatible at their
defaults.**

**Decision.** The experiment uses `window_size=2, overlap=0, top_windows=1,
context_extension=0`, which yields two windows per chunk and a measured **~29%**
reduction. The alternative -- larger chunks -- would change the index, the ground
truth and every recall number already recorded, for a benefit that is
unmeasurable until a generator exists.

**Two consequences worth stating.**

1. **A zero-reduction result would have looked like a null finding about SCR**
   rather than a mismatch between two configurations. It would have been reported
   as "adaptive SCR does not help", which is a claim about the contribution
   rather than about the setup. Smoke-running before the real run is what
   separated those.
2. **SCR is expensive here, and that is a real measurement.** It embeds every
   window of every retrieved chunk on the query path: with k=5 and two windows
   per chunk that is ten forward passes through gte-small per query, costing
   roughly 45x the retrieval it follows. On this corpus SCR does not trade a
   little latency for fewer tokens; it trades a great deal of latency for ~29%
   fewer tokens. Whether that trade is ever worth it depends on generation cost,
   which Phase 7 would supply and cannot.

---

## ADR-0036 — The profile table is measurements, and a blocked sweep corrupted it

**Status.** Accepted. Phase 6's remaining deliverable, plus the fourth occasion
this project has been bitten by a blocked measurement.

**Why a table and not the cost model.** Phase 3's latency model holds at 14.75%
MAPE inside its fitted regime and fails at 35.87% outside it because `cpu_util`
is censored (ADR-0033). A budget policy that *predicted* latency from it would
be most confident exactly where it is most wrong -- under load, which is the
condition this project exists to handle. So `ProfileTable` records what each
effort level actually cost and returned when it ran, and carries the conditions
of measurement so quiet-machine numbers cannot be silently applied to a loaded
one.

**The sweep was blocked, and it inverted the table.** The first attempt ran all
of `nprobe=1`, then all of `nprobe=2`, and so on. The result:

```
 nprobe    p50 ms    p95 ms   recall        (BLOCKED -- WRONG)
      1    38.793    54.524   0.3075
      4     4.807    51.462   0.6235
      8     8.172    39.320   0.7623
```

Latency *fell* as effort rose, because the first levels paid the cold-start and
the later ones did not. The derived Pareto front then declared **nprobe 1, 2, 4
and 8 "dominated and never worth choosing"** -- exactly the cheap levels a
controller must reach for under pressure. A budget policy reading that table
would have been structurally unable to back off, and the failure would have
looked like a policy bug rather than a corrupted input.

**Interleaved and warmed, the table is monotone in both columns:**

```
 nprobe    p50 ms    p95 ms   recall        n
      1     1.171     1.562   0.3189    1,990
      8     7.921     9.461   0.7849    1,990
     16    15.811    18.295   0.8906    1,990
     32    31.421    35.611   0.9586    1,989
pareto front: every level. dominated: none.
```

Every level on the front is the correct answer for a well-behaved index: each
buys recall for latency and none is strictly worse than another.

**ADR-0030 already said this** -- "a wall-clock comparison must interleave" --
and it was written after the same mistake in a different harness. The rule was
recorded and then not applied to a new sweep. The lesson is that a rule stated
once in a document does not propagate to code written later; `profile()` now
carries the reason in its own docstring, where someone writing the next sweep
will see it.

**Policy comparison, one declared budget, fixed from the table before any policy
ran:**

```
budget = p95 of the median swept level (nprobe=12) = 14.01 ms

  static   p95 14.20 ms   viol 6.0%   recall 0.8480   nprobe 12
  aimd     p95 13.44 ms   viol 4.3%   recall 0.8103   nprobe 10
  budget   p95  9.51 ms   viol 0.2%   recall 0.7797   nprobe  8
```

**The budget policy honours its declared latency budget at 0.2% violation, 30x
better than static, for 0.068 recall.** It is deliberately conservative -- 0.9
headroom means it selects the level whose measured p95 fits with margin, because
the table was measured on a machine in one state and aiming at the budget
exactly would put half the error on the wrong side of it.

**C4 is half-delivered, and the half is named.** Latency-budget-constrained
retrieval works and is measured. Energy-budget-constrained retrieval has the
machinery and no measurement: `BudgetPolicy.honours_energy_budget()` returns
**False** and every run manifest records it, because acting on a modelled energy
as if measured is the precise failure the capability gate exists to prevent.
Phase 3.5 supplies the measurement that would complete it.

---

## ADR-0037 — A tolerance flag that ignores the verdict is a check that cannot fail

**Status.** Accepted. The third instance of one root pattern, and the one worth
naming as a pattern rather than an incident.

**What happened.** `scripts/verify_all.py` runs the documented entry points end
to end. Its acceptance step has one *expected* failure -- NYTimes check 4 is a
dataset property, not a defect -- so demanding exit 0 could only be satisfied by
hiding a real result. The escape hatch was `expect_zero=False`.

`expect_zero=False` does not mean "tolerate exit 1". It means **tolerate
anything**. The step timed out when the host slept mid-run, returned the `-1`
sentinel, and the harness reported:

```
  OK    phase 2 acceptance     exit  -1  26010.6s      <- 7.2 h against a 1800 s timeout
7/7 steps reproduced.
```

Six steps had reproduced. One had never finished. The summary said seven.

**Fixed by replacing the boolean with the verdict itself:** `expect_codes=(0, 1)`
for that step, `(0,)` elsewhere. A timeout is `-1`, in no step's expected set,
and can never be reported as reproduced. Unit-checked in both directions before
re-running.

```
  OK    phase 2 acceptance     exit   1    762.8s
7/7 steps reproduced.
```

762.8 s against 26,010 s also settles what the earlier duration was: the host
sleeping, not a measurement. A warning now fires whenever wall-clock exceeds the
timeout without the timeout firing, because the exit code can be fine and the
duration still meaningless.

**The pattern, stated once so it is recognisable next time.** Three defects in
this project share one root: *a check whose outcome does not depend on what it
is checking.*

1. Check 4 compared an index with itself and reported PASS.
2. `headline_sentence()` carried a hardcoded conclusion beside computed numbers,
   and printed "the control law does not work" next to a negative recall cost.
3. This: a tolerance flag that accepted every outcome.

Each produced a **green line in a summary** — the line nobody re-reads. None was
caught by a test, because in every case the code did exactly what it was written
to do. All three were caught by reading a number that looked wrong.

**The design rule.** An escape hatch should say *what is expected*, never *ignore
the result*. `expect_codes=(0, 1)` and `expect_zero=False` differ only when
something unexpected happens, which is precisely the case the check exists for.

---

## ADR-0042 — The CLI entry point was a no-op, and the verification scored it as a pass

**Status.** Accepted. Found 2026-09-05, by a user pasting a documented command
into their own terminal and getting nothing back.

**What was wrong.** `smartann/cli.py` defined a Typer `app` and thirty-odd
commands, and never called `app()`. There was no `if __name__ == "__main__"`
guard and no `smartann/__main__.py`. So:

```
> python -m smartann.cli data status
>                                     # no output
> echo $LASTEXITCODE
0
```

Python imported the module, executed every decorator, bound every command, ran
none of them, and exited **0**. Every command in `docs/RUNBOOK.md` used this
form. All of them did nothing, silently, successfully.

The console script worked the whole time — `pyproject.toml` declares
`smartann = "smartann.cli:app"`, and that path invokes `app` directly. So the
indexes and experiments this project reports were genuinely built and genuinely
run. What was broken was the *documented* invocation, and only that.

**Why it survived thirty-nine commits.** `tests/test_cli.py` uses Typer's
`CliRunner`, which calls `app` in-process. That exercises the one path that was
never broken. The bug lived exclusively in the gap between how the tests invoke
the CLI and how the docs tell a human to.

`scripts/verify_all.py` should have caught it and instead certified it. Its
data-status step ran `[PY, "-m", "smartann.cli", "data", "status"]` and checked
`returncode in expect_codes`. The no-op returned 0. The step went green. **The
one check aimed at "do the documented entry points still work" was itself
routed through the broken entry point**, and reported 7/7.

**What the fix exposed.** With the entry point repaired, `data status` ran for
the first time and exited **1**: 13,154 files on disk with no manifest entry,
all inside built index directories. That check had never once executed.

Indexes are build outputs — reproducible from `(config.yaml, seed)`, rebuilt
routinely, and each root already stores a `meta.json` naming its config hash.
Hashing 13,154 files into the data manifest would record the same provenance
twice and churn on every rebuild. So `unrecorded_files()` now skips an index
root **conditionally on that `meta.json` existing**; a root without one is a
half-written build and its files are still reported. `data status` now exits 0.

**The decision.** Three changes, none of them the obvious one:

1. `if __name__ == "__main__": app()` in `cli.py`, plus `smartann/__main__.py`
   so `python -m smartann` works too. Both forms, because the runbook documents
   one and a reader would try the other.
2. `tests/test_entrypoint.py` runs both entry points **as subprocesses** and
   asserts stdout is non-empty. An in-process runner cannot catch this class of
   bug by construction. The sharpest assertion is that an unknown subcommand
   exits non-zero — the broken entry point exited 0 for every argument list ever
   passed to it, including nonsense, because it never parsed one.
3. `verify_all.py` now fails any step that produces **no output at all**. This
   is the general form: an exit code is a claim, output is evidence, and a step
   that reproduced something says so.

**The pattern, for the fourth time.** ADR-0037 named it: *a check whose outcome
does not depend on what it is checking*. The three prior instances were an index
compared with itself, a hardcoded narrative sentence, and a tolerance flag that
accepted every outcome. This one is worse than all three, because the check was
correct and complete — it simply asked its question through a channel that could
only ever answer yes.

**What it costs to state honestly.** The 7/7 reported in ADR-0037 was 6/7 with
one vacuous pass. No measurement in this project changes: the acceptance checks,
the profile sweep and the policy comparison all run through `scripts/`, not the
CLI. The reported *results* stand. The reported *verification* did not.

**The rule this adds.** Test the interface the documentation names, through the
mechanism the documentation names. A CLI tested only in-process is a library
with a CLI-shaped docstring.

---

## ADR-0043 — Phase 7 was never blocked; the blocker was scoped to the wrong noun

**Status.** Accepted. 2026-09-05.

**What the project believed.** Phase 7 (answer generation) was recorded as
blocked on the MSVC C++ Build Tools, in `TODO.md`, in `docs/review.md`, and in
the runbook's gate table. The evidence was solid and remains correct:

```
uv pip install --only-binary=:all: llama-cpp-python
  -> all versions of llama-cpp-python have no usable wheels
```

on CPython 3.12 / win-amd64. Source build only, and unlike hnswlib there is no
differently-named redistribution — the rung-2 escape that resolved Phase 2
(ADR-0017) genuinely does not exist for llama.cpp.

**What was wrong with it.** The project needs *a generator*. It does not need
llama.cpp. `sentence-transformers>=3.0` has been a declared dependency since
Phase 1 — it is what embedded the 46,369 SQuAD chunks — and it pulls in
`torch` and `transformers`. Both were installed the whole time:

```
torch 2.14.0    transformers 5.16.1    tokenizers 0.23.2    safetensors 0.8.0
```

So a CPU generator was available from the moment Phase 1 finished. The blocker
was written down against the tool rather than against the capability, and once
written it was never re-examined — three phase tables and a runbook repeated it
faithfully.

**Section 4 rule 6 is not engaged.** No dependency is added. `transformers` is
already installed as a transitive dependency of a declared one; this ADR makes
that use explicit rather than incidental.

**The model.** `google/flan-t5-base`, greedy decoding, `do_sample=False`,
CPU, 4 threads. Seq2seq rather than causal because FLAN-T5 answers
extractively without a chat scaffold, and SQuAD's gold answers are spans — a
causal model of similar size spends its budget on conversational framing that
EM punishes, which measures the prompt rather than the retrieval. Greedy
because C2 compares quality *across* reduction settings, and sampling noise
would sit inside exactly the difference being measured.

**What this unblocks.** C2's quality claim and C3's real comparison. Until now
context reduction could be priced only in characters, tokens and milliseconds,
and on those axes a reduction that deletes the answer is indistinguishable
from a free one — both look like a win.

**What it does not unblock.** Nothing about energy or thermal state. Those need
the phone (ADR-0007, ADR-0008), and C4's energy half and C5 entirely remain
gated on hardware. Phase 7 being available does not move them.

**The rule this adds.** Record a blocker against the *capability* that is
missing, not against the first tool tried for it. "Needs MSVC" was true of
`llama-cpp-python` and false of Phase 7, and the difference went unnoticed for
as long as nobody restated the requirement.

---

## ADR-0044 — An ablation ladder must be validated against the corpus before it is run

**Status.** Accepted. 2026-09-05.

**What happened.** The first Phase 7 run swept four SCR settings — `off`,
`mild`, `moderate`, `aggressive` — at `window_size=2` with `top_windows` 4, 3,
2, 1. Three of the four came back **byte-identical to `off`**: same character
count, same token count, same EM, same F1, 0.0% reduction.

The reducer was not broken. The ladder was. Measured over 3,000 sampled SQuAD
chunks:

| sentences per chunk | share |
|---|---|
| 1 | 1.4% |
| 2 | 22.6% |
| 3 | 76.0% |

and the windows those yield:

| window, overlap | chunks yielding exactly one window |
|---|---|
| `w=3, o=1` | **100.0%** — cannot reduce at all (ADR-0035) |
| `w=2, o=1` | 24.0%; the rest yield two, so only `top_windows=1` cuts anything |
| `w=1, o=0` | 1.4% — the only setting with a real ladder |

At `w=2` a chunk yields at most two windows, so "keep the top 3" and "keep the
top 4" keep everything. Three rungs of a four-rung ladder were the same rung.

**Why this is the ADR-0037 pattern again.** Those three arms could not have
differed from the baseline whatever the reducer did. Their outcome was
independent of the treatment being ablated — the fifth instance in this project
of a comparison that cannot fail, after an index compared with itself, a
hardcoded narrative sentence, a tolerance flag accepting every outcome, and a
verification routed through the entry point it was verifying (ADR-0042).

It is also the most seductive of the five, because the output looked *good*:
three arms holding quality exactly flat while reducing context reads as "SCR is
free" rather than "SCR did not run". Had the reduction percentage not been
printed beside the quality, this would have become a headline result.

**The decision.** The ladder is now `w=1, o=0` with `top_windows` 3/2/1 plus a
context-extension rung, and the corpus measurement above is recorded in
`scripts/phase7_qa.py` next to the ladder it justifies. The reduction
percentage is printed for every arm, so a rung that does not move is visible
in the same table as the quality it claims to preserve.

**The rule this adds.** Before running an ablation, verify that each rung can
produce an outcome different from its neighbour **on the actual data**. A
treatment level the corpus cannot express is not a weak condition; it is the
control condition wearing a different name.

---

## ADR-0045 — The generation baseline was truncated, which would have made reduction look free

**Status.** Accepted. 2026-09-05.

**What happened.** The first Phase 7 smoke run reported **66.7% of the `off`
arm truncated**. FLAN-T5's nominal input limit is 512 tokens; the unreduced
context at `k=5` reaches 589. The tokeniser cut it silently.

**Why it matters more than it looks.** The `off` arm is the baseline every
reduction arm is measured against. Truncated, it was not "full context" — it
was "the first 512 tokens", which is itself a context reduction, just a
position-based one instead of a relevance-based one. Comparing relevance-based
reduction against position-based reduction of the same size would have shown
little difference, and the conclusion drawn would have been *"reduction is
free"* when what actually happened is that the comparison never took place.

**The decision.** `max_input_tokens=1024`. T5 uses relative position bias
rather than learned absolute embeddings, so it extrapolates past its training
length; this is the one architecture family where raising the limit is sound
rather than a silent corruption. Truncation is now 0.0% on every arm.

`GenerationResult.truncated` exists for this and is reported per arm. It is
computed by tokenising twice — once unbounded to learn the true length, once
truncated to feed the model — because the difference cannot be recovered after
the fact.

**The rule this adds.** A baseline that has been silently altered is not a
baseline. Any pipeline stage that can drop input must report whether it did,
per arm, in the same table as the result.

---

## ADR-0046 — The device probe: two criticals passed, and a build that succeeds then crashes

**Status.** Accepted. 2026-09-05. All seven pre-declared questions answered;
raw logs in `docs/device/`.

**It was the cable.** The Phase 3.5a probe recorded `device_reachable: false`
with the reasoning that Windows enumerated no Android device *at all*, which
placed the fault upstream of adb and developer options. That held: with a data
cable the handset appeared immediately. vivo I2304, Android 16, arm64-v8a,
Snapdragon.

**Q2 — battery current without root: PASS. This is the consequential one.**

The pre-declaration named this the question that could sink a phase rather than
delay it: *"If [it fails] there is NO measured energy anywhere in this project."*

The sysfs route is dead. Every node under
`/sys/class/power_supply/battery/` — `current_now`, `voltage_now`,
`charge_counter` — is unreadable to the adb shell user under Android 16
SELinux, as are the `uevent` files of `battery`, `bq_bms` and `fg_common`.

`termux-battery-status` works, because it goes through BatteryManager rather
than the filesystem. The declared test was whether the value **moves** under
load, since a constant or zero reading is not an energy source. It moves:

```
idle        current -215000 uA   voltage 4331 mV   35.0 C
under load  current -104000 uA   voltage 4311 mV   35.1 C
```

**There is a measured energy path in this project.** C4's energy half and C5 are
not sunk, and Phase 3's energy criterion is achievable as scoped.

**The sign is not usable yet.** Across Q5's run, under *continuous* load, the
field read −106000, +110000, +46000, +97000 µA at 30-second intervals. Magnitude
tracks load; the sign flips as the charger modulates. Every reading in this probe
was taken while charging, because USB was the only transport at the time. **The
convention must be re-established from an unplugged device before a single joule
is reported**, and wireless adb exists so the cable can come out.

**Q1 — hnswlib under Termux: PASS, and the failure mode is the finding.**

hnswlib 0.8.0 compiles in 19 s with clang 21.1.8 and produces a wheel. pip
reports success. Exit code 0. **And `import hnswlib` dies with SIGILL before
executing a line.** Every later step fails identically because none is reached.

`setup.py` compiles with `-march=native`, and clang targeting this core emitted
an instruction it does not implement — `/proc/cpuinfo` lists no SVE.
`HNSWLIB_NO_NATIVE=1` fixes it completely: import OK, self-recall@5 = 1.0000
against brute force, 11.4 µs per query, and `l2` returns squared distance as
ADR-0018 requires.

**Both builds exit 0.** This pre-declaration required *"a functional
`knn_query` check rather than just an import"*, and that clause is the only
reason this was caught. A build-status check would have certified a library that
crashes on import — the same shape as ADR-0042, where a step's verdict did not
depend on what it was checking. Every device-side build MUST set
`HNSWLIB_NO_NATIVE=1`, in the setup script, not in someone's memory.

**Q3 — CPU topology: ANSWERED, and it contradicts `types.py`.**

Eight cores in **three** frequency domains: cpu0–2 at 2.016 GHz, cpu3–6 at
2.803, cpu7 at 3.187. One instantaneous sample held four different ratios at
once: 1.00, 0.61, 0.22, 0.27.

`DeviceState.freq_ratio` is a single scalar and **this device has no single
clock ratio to report**. Which ratio a query experiences depends on where the
Android scheduler placed the thread — something this project neither controls
nor observes from inside the process. **No fix is applied here**: changing
`types.py` needs the supervisor's go-ahead, and the honest options (per-cluster
vector, the ratio of the core actually used, or a documented worst case) differ
in what the controller would be permitted to claim.

**Q4 — thermal: PASS.** 110 zones readable without root: `cpuss-0..3`
40.6–41.4 °C, `cpu-1-*` 38.6–41.8 °C, `gpuss-*` 35.8–37.0 °C, `skin-msm-therm`
32.7 °C, battery 31.0 °C. Battery temperature rose monotonically under load
during Q5 (36.2 → 36.9 °C), so the signal Phase 4 must fit is real and visible.

Two things recorded rather than guessed. **21 of the 110 zones report −273000**
— absolute zero, i.e. a disabled sensor — and any aggregate that does not filter
them is wrong in the direction that looks like a cold device. And **there is no
single "SoC package" zone**: four `cpuss` and fourteen `cpu-*` sensors disagree
by 3.2 °C at idle. Which one maps to `temperature_c` is a choice to declare and
defend, which is exactly what the pre-declaration said to record rather than
resolve by guessing.

**Q5 — sustained load: PASS at 150 s, and not one second more.** Wake lock
acquired; the child was alive at 30/60/90/120 s and its `exit=0` at 150 s is
*completion*, not a kill — a kill surfaces as 137. Noted because the liveness
probe alone cannot tell those apart at the boundary. This is the 2-minute smoke
test that was asked for. It is **not** evidence for Phase 3.5's 30 minutes, it
spawned one child where Android 12+'s phantom-process killer targets many, and
Termux 0.118.3 predates this handset's Android 16 by four major versions.

**Phase 3.5 decision: GO**, on two conditions — the trace is taken unplugged,
and the 30-minute multi-process case is verified before it is relied on.

---

## ADR-0047 — Energy sign convention, calibrated on battery, cross-checked two ways

**Status.** Accepted. 2026-09-05. Raw samples in `docs/device/calib.log`.

**Why a calibration and not a measurement.** ADR-0046 established that
`termux-battery-status` reports a `current` field that responds to load, but
every reading was taken while charging, where the sign flipped between samples
under *constant* load as the charger modulated. A joule reported before knowing
which direction means "leaving the battery" is a plausible-looking wrong number,
which is the failure mode this project exists to avoid. So the first thing run
on battery was not an experiment.

**Design: idle / load / idle, not before-and-after.** A single pair cannot
distinguish a load effect from drift. Bracketing the load with a baseline on
both sides measures the drift directly — the same reason Phase 8 had to be
re-run interleaved (ADR-0046, defect #20).

**Result.** 40 samples at 3-second intervals, wake lock held, 7 cores loaded for
60 s, `plugged: UNPLUGGED`, `status: DISCHARGING` throughout.

| phase | n | current (median) | range | voltage | battery | skin |
|---|---|---|---|---|---|---|
| idle_pre | 10 | **−334 mA** | −1050 … −265 | 4280 mV | 31.4 °C | 36.6 °C |
| load | 20 | **−753 mA** | −1084 … −221 | 4248 mV | 31.6 °C | **41.4 °C** |
| idle_post | 10 | **−256 mA** | −507 … −213 | 4295 mV | 32.6 °C | 38.1 °C |

**The sign convention: NEGATIVE is discharge.** All 40 samples were negative
while discharging — no exceptions, no flips. This is the fact every later energy
figure depends on, and it is now established rather than assumed.

**Load costs −419 mA against the pre-baseline and −497 mA against the post.**
Idle drift between the two baselines is +78 mA, which is why both were measured;
quoting the load delta against a single baseline would carry that drift silently.

**Independent cross-check, and it agrees.** `charge_counter` fell 1708000 →
1689000 µAh over 148 s. Integrating that gives **462 mA** mean draw. Integrating
the instantaneous current samples over the same window, weighted by phase
duration, gives **524 mA**. Two mechanisms — a coulomb counter and a current
sensor — agreeing within 12% is the external reference this project's defect
taxonomy calls the only category that catches plausible-looking errors. Neither
number is derived from the other.

**Voltage sags under load**, 4280 → 4248 mV, and recovers to 4295. Energy must
therefore be integrated as **V·I·dt**, not I·dt scaled by a nominal voltage.
Treating voltage as constant would understate load power by roughly 0.7% here
and more under heavier draw.

**Power, for scale.** Idle ≈ 334 mA × 4.280 V = **1.43 W**. Loaded ≈ 753 mA ×
4.248 V = **3.20 W**. A 7-core busy loop costs about **1.8 W** on this handset.

**Thermal signal confirmed at useful magnitude.** Skin temperature rose 36.6 →
41.4 °C over 60 s of load and fell back to 38.1 °C — a 4.8 °C swing, an order of
magnitude above sensor noise. This is the signal Phase 4's thermal constants
must be *fitted* from. Battery temperature moved only 31.4 → 32.6 °C, so the
skin zone is the responsive one and the battery zone is not a substitute.

**What this does not settle.** The `current` range within each phase is wide
(idle spans −1050 to −265 mA) because the screen, radios and background apps are
not controlled. Per-query energy attribution will need either a much longer
integration window or an idle-baseline subtraction, and which of those is sound
is a Phase 3.5 question, not settled here.

---

## ADR-0048 — A guard enforced once at t=0 does not hold for thirteen minutes

**Status.** Accepted. 2026-09-05.

**What happened.** A full `verify_all.py` run came back **6/7**: the Phase 2
acceptance step timed out at 4,686 s against a 3,600 s limit, having taken
**796 s** the previous time. A 5.9× degradation.

The cause was not in the project. `VALORANT-Win64-Shipping` was running with
9,729 s of accumulated CPU and 2 GB resident, and the machine sat at 63% busy.

**Why this is worth an ADR anyway.** `assert_machine_quiet()` exists precisely
to prevent this, and it did not. It samples the CPU **four times over one
second, before the run starts**, and then a thirteen-minute measurement proceeds
on the strength of that one-second observation. The game either launched
partway through or the check landed in a quiet moment.

**The near miss is the point.** This run failed loudly because it was 490%
slower and blew a timeout. *Had it been 10% slower it would have been written
into `phase2_summary.json` as a measurement*, and the figures would have plotted
it. The guard's failure mode is silent in exactly the range where corruption is
most plausible.

This is the fourth occurrence of this defect in the project (#14 counted three:
a concurrent experiment, this project's own O(n²) figure generation perturbing
the latencies it was plotting, and leftover load). The rule "run experiments
alone" has now been violated four times *while written down*, which is evidence
about rules rather than about discipline.

**The decision.** `MachineWatch` samples CPU every 2 s for the **whole** run and
reports the distribution — median, p95, max, and the fraction of samples above
threshold — into `phase2_summary.json`. A run with more than **10%** of its
samples over the threshold exits **2**.

Three things about that design were deliberate:

**Exit 2, not exit 1.** Exit 1 already means "a check failed", which is a
*result* — the NYTimes cluster-miss property is a real finding. A contaminated
run is the **absence** of a result, and collapsing the two would let a corrupted
run be certified as the expected outcome. `verify_all.py` expects `(0, 1)` for
this step, so exit 2 fails verification without any further change.

**Fraction of samples, not the mean.** One sample at 100% lifts the mean of 100
quiet samples to 6%, so a mean-based rule near 5% trips on a single spike.
Counting samples over the line means magnitude cannot substitute for duration.

**10%, and a brief spike is tolerated.** A guard that fires on any spike gets
disabled within a week, which is worse than one that fires only on sustained
load. The threshold is a judgement, so it is pinned by tests in
`tests/test_machine_watch.py` — a judgement with no test is a number that
drifts.

**The watch runs even under `--allow-busy`.** Opting out of the *gate* is not
opting out of the *evidence*: a run taken knowingly on a busy machine still
needs its contamination recorded, or the numbers outlive the caveat.

**What did not change.** The pass/fail verdicts from the contaminated run are
still valid — they are recall and memory properties, not timings. Only the
durations are meaningless. The failure message says so, because "re-run
everything" is the wrong instruction when only one column is affected.

---

## ADR-0049 — C4 and C5 built, and a capability table that lied about itself

**Status.** Accepted. 2026-09-05.

**What was built.** Three pieces that existed as claims and now exist as code:

- `smartann/telemetry/android.py` — the first monitor in this project that can
  report `temperature_c` as *available*, reading thermal zones, per-cluster
  frequencies and battery state from the handset over adb.
- `smartann/controller/policies/device.py` — **C5**, a policy that derates
  retrieval effort on temperature, thermal slope, battery level and throttling.
- `smartann/controller/policies/energy.py` — **C4's second half**, the first
  policy in the project whose `honours_energy_budget()` can return True.

**The defect this work produced, and it is the interesting part.** The first
`AndroidMonitor.capabilities()` hardcoded `temperature_c` and `freq_ratio` to
`True`, on the reasoning that ADR-0046 had *proven* the handset reports them.
Run against a disconnected phone it declared them available while `sample()`
returned NaN.

That is a capability table asserting a claim its own monitor cannot satisfy —
the exact failure `assert_manifest_consistent` was written to catch, appearing
inside the class that is supposed to be the source of truth for it. It was
caught by running the monitor against the real device and reading the output
next to the table, not by any test.

The fix: **every `available` is earned by a live read**. `temperature_c` is
available when a zone actually returns a finite number; `freq_ratio` when
cpufreq actually parses. A capability is a statement about *this monitor
reaching this device now*, not about what the hardware can do in principle.

**Three design decisions worth defending.**

*`honours_energy_budget()` is per-instance, not per-class.* It returns True only
when an energy source is wired in **and** the capability is declared. A source
without the flag is a caller who wired a mock; a flag without a source is a
caller declaring a platform capability this policy cannot reach. On the
development laptop it constructs without either and reports False, exactly as
`BudgetPolicy` always has.

*`DeviceStatePolicy.honours_energy_budget()` stays False.* It reads battery
percentage and charging state. Neither is an energy budget. Returning True
because a *related* number became readable is the precise slip the capability
gate exists to prevent, and the two policies are separate classes so that the
distinction is structural rather than remembered.

*Device state can only ever LOWER effort.* A policy able to raise effort on a
hot device could violate the latency SLO in the name of thermal management,
which is not a trade anyone asked for. The latency budget sets the ceiling and
device state derates from it.

**What is deliberately not here.** No thermal model. Phase 4 must fit one from a
measured trace, and choosing constants now would break "fitted, never chosen" in
a new place. `DeviceTiers` encodes only the ordering that survives without a
model — hotter and emptier means less effort, monotonically — and its
thresholds are anchored on the one thermal measurement that exists: skin at
36.6 C idle, 41.4 C after 60 s of load (ADR-0047). 42 C is therefore "sustained
load" rather than a round number, and 46 C is deliberately above anything
observed, because a shield should trip on states the calibration did not visit.

**What this does not make true.** C4 and C5 are *built*, not *demonstrated*.
Neither has run against the handset in an experiment, and until one does the
scoreboard says built-not-demonstrated. 440 tests.

---

## ADR-0050 — HNSW index files are not portable between host and device

**Status.** Accepted. 2026-09-05. Phase 3.5 groundwork.

**The question this settles.** The device probe recorded, as an open risk, that
"index files written by one are not assumed loadable by the other, and Phase 9
must verify that rather than presume it" (ADR-0046). Verified. **They are not.**

**What happened.** The SQuAD EcoVector index was pushed to the handset and
`load_index` refused it:

```
RuntimeError: Index seems to be corrupted or unsupported
```

The obvious hypothesis is a truncated transfer, and it is wrong. The file is
byte-identical on both machines — 391,128 bytes, SHA-256 `a0060e6d4fbe9b92…`
on the laptop and on the phone. Nothing was corrupted; the format is simply not
readable by the other library.

The writer is **chroma-hnswlib 0.7.5** (the redistribution this project uses
because upstream publishes no Windows wheels, ADR-0017). The reader is
**upstream hnswlib 0.8.0**, the only one that builds under Termux.

**Neither can be moved to meet the other.**

- Upstream hnswlib has no 0.7.5. Its releases go 0.7.0 → 0.8.0; the 0.7.5
  version number belongs to chroma's fork alone, so "pin the same version" is
  not available.
- `chroma-hnswlib==0.7.5` does not install under Termux. Its build backend
  (`mesonpy`) fails to bootstrap on aarch64/Android, with and without build
  isolation.

**The resolution: only the graph is unportable, so only the graph is rebuilt.**

An EcoVector index is mostly plain `.npy` — centroids, assignments, and (with
per-cluster graphs disabled, ADR-0028) every cluster. NumPy's format is
portable and transferred without complaint. The single non-portable artifact is
`centroid_graph.bin`, and it holds 215 centroids.

So it is rebuilt on-device from the portable `centroids.npy`, using the
parameters recorded in `meta.json` — M=32, ef_construction=200, space=ip —
rather than defaults. **0.016 seconds.**

**Equivalence was checked, not assumed.** Rebuilding a graph changes its random
structure, so the device index could differ from the host's. Measured against
the same ground truth at the same configuration (nprobe=16, k=10, 300 queries):

```
HOST   recall@10 = 0.9543
DEVICE recall@10 = 0.9543
```

Identical. The trace therefore measures the same retrieval system, not an
approximation of it. Had they differed, every device number would have been
about a different index and the comparison to host results would have been
meaningless.

**Consequences for Phase 9.** Any device deployment must rebuild the centroid
graph locally; shipping the host's binary does not work and fails loudly rather
than silently, which is the one merciful part. Per-cluster graphs would make
this far worse — there would be hundreds of unportable binaries instead of one
— which is an argument for ADR-0028's decision that had not previously been
noticed.

**A related portability fix.** `psutil` does not build under Termux, and three
modules imported it at module scope. It was used for exactly one thing:
reporting resident memory. It is now optional in `ecovector.py`, `baselines.py`
and `builder.py`; without it the memory fields report **NaN**, which is this
project's established meaning for "not measured", rather than 0.0, which would
read as "no memory used" and silently satisfy the Phase 2 footprint check.
Building an index without psutil raises instead, because a build that cannot
record its own peak memory should not pretend to.

---

## ADR-0051 — Phase 4: a thermal model fitted from a measured device trace

**Status.** Accepted. 2026-09-05. `results/thermal_model.json`, trace in
`results/device/`.

**The debt this pays.** ADR-0007 and ADR-0008 recorded in Phase 1 that thermal
ground truth exists only on the handset, and every phase since refused to pick
thermal constants rather than measure them. 19,144 queries over 25.0 minutes on
the vivo I2304, sweeping `nprobe` 1→32 round-robin, with 478 thermal samples.
Skin temperature rose **32.90 → 37.51 °C**.

**The model.** First-order lumped thermal mass, discretised:

```
dT/dt = a·work_rate + b·(T − T_ref) + c
      = +9.143e-04·work − 1.473e-02·(T − 34.57) − 1.421e-01
```

**τ = 68 s.** That is the number a controller needs: how long a change in
retrieval effort takes to appear in temperature.

**The differencing window is an analysis choice, and it moves R² sixfold.**

| window | n | R² | a | τ |
|---|---|---|---|---|
| 3 s | 446 | 0.109 | 6.08e-04 | 107 s |
| 15 s | 89 | 0.230 | 9.43e-04 | 67 s |
| **30 s** | **44** | **0.304** | **9.14e-04** | **68 s** |
| 60 s | 22 | 0.187 | 5.94e-04 | 97 s |
| 120 s | 11 | 0.691 | 8.69e-04 | 76 s |

The whole sweep is reported, because presenting one row would conceal that the
answer depends on a choice no reader would think to question. **The 120 s row is
excluded from the headline despite scoring highest**: eleven points against
three free parameters is a number that flatters itself.

**Why short windows fail.** Skin rose 4.6 °C over 25 minutes — about
0.003 °C/s — while differencing adjacent 3-second samples yields a dT/dt with
standard deviation 0.051 °C/s, **sixteen times the signal**. The sensor
quantises at ~0.005 °C, so 3-second differencing measures quantisation rather
than physics.

**The result that matters is the stability, not the R².** τ stays within
67–107 s and `a` within 5.9–9.4e-04 across every window, while R² swings from
0.11 to 0.69. Point estimates that barely move across analysis choices are
better evidence than any single goodness-of-fit, and R² = 0.30 is honestly what
a first-order model achieves against a noisy sensor over 25 minutes.

**A diagnosis that was wrong, recorded because it was plausible.** The first
reading of R² = 0.11 was that round-robin interleaving had flattened the input:
every 3-second interval contains all five effort levels, so work_rate should
barely vary. Checked rather than assumed — work_rate has CV 0.21, range 92–242,
and a cooling-term-only fit scores **R² = 0.0002**, meaning work_rate carries
*all* the explanatory power. The interleaving was not the problem; sensor noise
was. ADR-0030's interleaving requirement and thermal identifiability do not
conflict after all, which is worth knowing before someone "fixes" the trace
design.

**What this model cannot support.** The handset was **charging** throughout, as
93 host-side power samples confirm (`usb: true`, no state change). Charging heat
is absorbed into `c`, so `c` is not the device's idle thermal state. All 478
battery reads failed — Termux:API was not installed — so **no energy was
measured**, and C4's energy half stays undemonstrated. The first 120 s are
trimmed: the cable was pulled and reattached during them.

**Per-effort latency, on the device, for the first time:**

| nprobe | p50 | p95 |
|---|---|---|
| 1 | 5.59 ms | 13.82 ms |
| 4 | 20.50 ms | 52.26 ms |
| 8 | 40.46 ms | 101.46 ms |
| 16 | 82.61 ms | 189.82 ms |
| 32 | 169.21 ms | 354.10 ms |

Clean linear scaling in `nprobe`, and roughly **13× slower than the laptop** at
matched settings — which is the gap that makes adaptive effort worth having on
a phone at all.

**The consequence for C1, stated plainly.** τ = 68 s means a thermally-aware
controller must act on a period of that order or longer. The Phase 6 collapse
happened because the control period was shorter than the observation delay
(ADR-0032); this measurement says the thermal loop's delay is **68 seconds**,
not milliseconds. Any thermal controller sampling faster than that will
reproduce the same collapse, and now there is a measured number saying so
instead of an argument.

---

## ADR-0052 — C4: energy measured on the device, and the part that is refused

**Status.** Accepted. 2026-09-05. `results/energy_model.json`, trace in
`results/device/e2_*`.

**The first measured energy in this project.** 15,425 queries over 20.0 minutes
on the vivo I2304, **unplugged**, `nprobe` swept 1→32, 209 telemetry samples,
**zero read failures**. All 209 samples `DISCHARGING`; all 209 currents
negative.

### What is measured

| | |
|---|---|
| total energy | **1531.4 J** over 20.0 min |
| mean power | **1.277 W** |
| power p5 / p95 | 0.736 / 2.218 W |
| all-in per query | **99.28 mJ** |
| voltage | 4.022–4.132 V (sagging under load, so V·I·dt, not I·dt) |

**Effort drives power, and the relationship needs no model:**

| work rate (clusters/s) | median power |
|---|---|
| 95–126 | 0.864 W |
| 126–141 | 0.925 W |
| 141–153 | 0.989 W |
| 153–189 | 1.330 W |
| 189–251 | **1.940 W** |

A 2.6× increase in retrieval work costs a 2.2× increase in power, correlation
+0.645. That is C4's premise demonstrated on real hardware: **retrieval effort
has a measurable energy price, so an energy budget is a thing a controller can
actually trade against.**

### What is refused, and why

The obvious next step is `power ~ a·work_rate + b`, giving marginal energy per
cluster. It fits, at R² = 0.415, and returns:

```
a = 9.444e-03 W per cluster/s
b = -0.209 W       <-- a phone idling at NEGATIVE 209 milliwatts
```

**That is not reported as a result.** The cause is visible in the data:
`work_rate` spans 95–251 clusters/s and never approaches zero, so the intercept
extrapolates 95 units past the nearest observation. The arithmetic confirms it
independently — at the trace's mean effort of 12.2 clusters the marginal term
predicts **115 mJ/query** against a measured all-in average of **99.3 mJ/query**.
The marginal exceeds the total, and the negative intercept absorbs the
difference.

A number implying negative idle power is not a slightly-wrong version of the
right answer. `fit_energy_model.py` computes it, records it as
`reportable: false`, and prints why — so the reasoning can be checked rather
than taken on trust.

**The fix is a trace change, not an analysis change.** The workload must pause
long enough to measure the baseline the screen and radios draw, instead of
leaving it to extrapolation. Without idle periods the decomposition is not
identifiable, however the fit is written.

### The guard this produced

An earlier 20-minute run completed with **all 209 samples CHARGING**, because
the cable was never pulled. Its current had a median of **0 mA** and a 103/102
split between negative and positive — the charger modulating, exactly as
ADR-0047 predicted. Integrating over a sign that is a coin flip yields
approximately zero, so that run was not a weak measurement but no measurement.

Asking more carefully was not the fix. `device_trace.py` now **refuses to
start** unless it reads `DISCHARGING` first, and `fit_energy_model.py` refuses
to integrate a trace containing any charging sample or any positive current.
Both failures were silent and both are now loud. This is the same lesson as
`MachineWatch` (ADR-0048): *a precondition asserted once and never verified is
not a precondition.*

### C4's status

**Half demonstrated, and the half is the important one.** Energy is measurable
on the target device, responds monotonically to retrieval effort, and
`EnergyBudgetPolicy.honours_energy_budget()` can now return True against a real
source. What is not demonstrated is a controller holding a per-query energy
budget in a live run, because per-query attribution needs the idle baseline
this trace lacks.

---

## ADR-0053 — Duty-cycle the trace so the energy baseline is measured, not extrapolated

**Status.** Accepted. 2026-09-06. Follows ADR-0052.

**The problem, restated in one line.** `power ~ a·work_rate + b` returned
**b = −0.209 W** because the trace never stopped working: `work_rate` spanned
95–251 clusters/s, so the intercept sat 95 units outside the data. No estimator
recovers a baseline that was never observed.

**The fix is in the experiment, not the analysis.** `device_trace.py` now
alternates WORK and IDLE — `--work-s 60 --idle-s 30` by default. During idle it
runs no queries and lets the sampler watch the screen and radios alone. That
turns the intercept from an extrapolation into an interpolation: the regression
reads the baseline off measured samples sitting at work_rate = 0.

Phase transitions are recorded in the trace metadata, so a reader can see when
the workload stopped rather than inferring it from a gap in the query log.

**The fit now requires the baseline rather than hoping for it.** Two conditions
must both hold before per-query attribution is reported:

1. the fitted intercept is non-negative, and
2. **at least 5% of intervals are at zero work.**

The second is the one that matters. A positive intercept fitted *without* idle
data is a lucky extrapolation, and it would be reported with exactly the same
confidence as a measured one — which is the failure ADR-0052 caught, wearing a
friendlier face. Below 5% the intercept is still resting on a handful of points
and the old failure mode is a rounding error away.

**Tested end-to-end against synthetic traces with known answers.** Seven tests
in `tests/test_energy_fit.py` plant a known idle power (0.80 W) and slope
(0.004 W per cluster/s) and check the fit recovers both, refuses a trace with
no idle time, refuses a charging trace, and refuses a positive current.

**A fixture bug worth recording, because it looked exactly like an estimator
bug.** The first synthetic generator emitted each row's queries *after* that
row's timestamp, while the fit attributes work in `(t[i-1], t[i]]` to the power
at `t[i]`. Power and work ended up one interval apart, and a perfectly linear
dataset fitted at **a third of its true slope**. Read without checking, that
says the estimator is broken; it was the test data. The alignment requirement is
now stated in the generator's docstring, because the next person to write a
synthetic trace will reach for the same wrong shape.

**Status of C4.** Unchanged until the trace is re-run: energy is measured and
effort demonstrably drives power, but per-query attribution stays
`reportable: false`. The machinery to produce it correctly now exists and is
tested; what remains is 20 minutes of handset time.

---

## ADR-0054 — C4 demonstrated: per-query energy, measured rather than extrapolated

**Status.** Accepted. 2026-09-06. `results/energy_model.json`, trace in
`results/device/c4_*`.

**What changed since ADR-0052.** Nothing about the estimator. The trace was
duty-cycled (ADR-0053) so the baseline is *observed* instead of extrapolated,
and the number that was unreportable is now reportable.

9,670 queries over 21.0 minutes, **unplugged**, 218 samples, zero read
failures, **60 idle intervals (27.6%)**, 14 clean work/idle alternations.

### The result

```
power = 3.201e-03 W per cluster/s  +  0.7091 W baseline
```

| | value | 95% CI (10,000 bootstrap resamples) |
|---|---|---|
| marginal cost | **3.201 mJ per cluster** | [2.145, 4.201] — excludes zero |
| fitted baseline | **0.7091 W** | [0.5957, 0.8385] — positive |
| **directly measured idle** | **0.6571 W** | median of 60 observed idle intervals |

**The fitted intercept and the directly measured idle power agree to within
7.9%**, from the same data but by different routes — one a regression
extrapolated to zero work, the other the median of intervals that actually sat
at zero work. That agreement is the check the previous trace could not perform
at all, and it is why the number is now reported.

Compare with ADR-0052's attempt on the continuous trace: intercept **−0.209 W**,
a phone consuming negative power. Same code, same estimator; the difference is
entirely in the experiment.

### The finding that changes how C4 should be read

| | energy | share of all-in |
|---|---|---|
| all-in per query | 130.99 mJ | 100% |
| retrieval @ nprobe=1 | 3.20 mJ | **2.4%** |
| retrieval @ nprobe=8 | 25.61 mJ | **19.5%** |
| retrieval @ nprobe=32 | 102.43 mJ | 78.2% |

**The screen and radios are 65% of mean power.** Retrieval is a *minority* of
what a query costs at any sane operating point — at the profile table's
budget-fitting level it is about a fifth.

That bounds C4 honestly. An energy-budget controller adjusting retrieval effort
is moving roughly 20% of the total, not 100%, so the achievable saving is
bounded by that share. The dial is real and its range is now known, which is
more useful than a larger number that ignored the baseline. It also says
something for the wider design: on a phone, retrieval effort is worth
controlling, but the display is the bigger lever and no ANN parameter touches it.

### What is still weak, stated plainly

**R² = 0.165.** The relationship is genuinely noisy: uncontrolled background
activity moves power between samples, and 218 samples over 21 minutes is not a
lot. The slope's CI is a factor of two wide, [2.1, 4.2] mJ/cluster. What
supports the estimate is not its R² but that it excludes zero, that its
intercept is physical, and that the intercept is corroborated by direct
measurement. A tighter number wants a longer trace with the screen off and
flight mode on — which would also make it less representative of real use.

---

## ADR-0055 — A C5 experiment that measured its own headroom

**Status.** Accepted. 2026-09-06. Defect #23.

**What it looked like.** The first on-device C5 run reported a clean win:

```
arm             n    p95 ms   viol %   recall  nprobe p50
static      19719     84.68     1.1%   0.9056           8
adaptive    19719     43.29     0.0%   0.8058           4
```

Zero violations against static's 1.1%, on a phone that heated from 27.9 to
35.3 °C during the run. Read at face value: C5 demonstrated.

**What was actually happening.** The summary also records how many adaptive
decisions cited a device signal, and that column is why this was caught:

```
adaptive nprobe distribution:            {4: 19719}   -- every query
  decisions citing device state:   542   -> chose 4
  decisions taken blind:        19,177   -> chose 4
```

**Device state changed the decision zero times.** The 542 readings that cited
temperature picked exactly what the 19,177 blind ones picked.

The gap came from a budget mismatch built into the experiment.
`DeviceStatePolicy` applies a `headroom` of 0.9, so the adaptive arm was
selecting against **99 ms** while the static arm used the raw **110 ms**. The
device-measured p95 for `nprobe=8` is 101.5 ms — above 99, below 110. So the
two arms were separated before device state was consulted at all, and the
experiment compared two budgets.

**Why it survived writing.** The script was built carefully: interleaved arms,
a tuned baseline, a falsification criterion stated in its own docstring. All of
that was about how the arms were *run*. The defect was in what they were run
*against*, and no amount of care in the loop reaches that.

This is the fifth instance of ADR-0037's pattern — *a comparison whose outcome
does not depend on the thing being tested* — after an index compared with
itself, a hardcoded narrative sentence, a tolerance flag accepting every
outcome, and a verification routed through the entry point it was verifying.

**What caught it.** The `adaptive_decisions_with_a_reason` counter, written
into the script specifically so that "the policy ran but never saw anything"
could not masquerade as a result. It was added defensively and it fired on the
first run. The instinct to instrument the *absence* of an effect is worth more
than the instinct to instrument its presence.

**The fix.** `--headroom`, defaulting to **1.0**, and the static baseline now
selects against `budget × headroom` — the same effective budget as the adaptive
arm. Any remaining difference is device state or nothing.

**A finding the failed run established anyway.** Every reason ever cited was
`temperature rising` at +0.06 to +0.13 °C/s. The absolute tiers — 42 °C warm,
46 °C hot (ADR-0049) — **never fired**, because skin peaked at 35.3 °C. On this
workload the handset does not get hot enough for a temperature-level policy to
act, and the thermal *slope* is the only device signal with anything to say.
That bounds what C5 can claim on this device regardless of the re-run's outcome.

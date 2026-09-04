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

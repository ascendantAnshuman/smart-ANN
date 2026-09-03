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

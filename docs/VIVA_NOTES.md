# SMART-ANN — viva notes

*Generated 2026-10-09 by `scripts/phase15_viva_pack.py`. Every number below is read from a result file at build time; none is typed in. Re-run the script after any experiment to refresh it.*

## 1. The project in one page

**The problem.** Retrieval-augmented generation on a phone has to choose how hard to search. Searching deeply gives better answers and costs latency, battery and heat; searching shallowly is cheap and misses things. The right choice is not fixed — it depends on what the device can afford at that moment.

**The baseline.** MobileRAG proposes EcoVector: an IVF partition where each cluster carries its own small HNSW graph, plus Selective Content Reduction to cut the context handed to the generator.

**What we found.** Under MobileRAG's own build condition (`ef_L * M' < N/Nc  (MobileRAG search-time models)`, N/Nc = 4000), the per-cluster graphs lost. At matched recall the flat, memory-mapped clusters were faster at every point measured — for example 0.9243 recall in 114.1 ms against 0.9227 in 305.5 ms for M=32. The graphs were dropped and the index now uses flat mmap'd clusters (ADR-0028).

**Our contribution.** A control layer that chooses retrieval effort per query from the device's measured condition and a measured profile of what each effort level costs and returns. Plus the discipline around it: every policy decision is explainable, every number carries its provenance, and the system refuses to report what it did not measure.

## 2. What ran on the phone, and what did not

**125,711 queries executed on the vivo I2304 across 6 recorded runs, 67,423 of them with the controller deciding on-device.** Those runs are in `results/device/`, and every count below is read from them rather than quoted from memory.

| run | queries | minutes | effort levels | controller decided on-device |
|---|---|---|---|---|
| `c4` | 9,670 | 21 | 1/4/8/16/32 | no |
| `c5` | 39,438 | 22 | 4/8 | yes |
| `c5b` | 27,985 | 22 | 4/8 | yes |
| `e2` | 15,425 | 20 | 1/4/8/16/32 | no |
| `energy` | 14,049 | 20 | 1/4/8/16/32 | no |
| `trace` | 19,144 | 25 | 1/4/8/16/32 | no |

Stage by stage:

| pipeline stage | measured on | evidence |
|---|---|---|
| embed the query | **host only** | scripts/device_trace.py loads pre-computed queries.npy and imports no embedder; gte-small has never run in Termux |
| probe the centroid graph | **handset** | EcoVectorIndex.probe, in Termux, on the handset's own rebuilt centroid graph (ADR-0050) |
| controller decides the effort level | **handset** | DeviceStatePolicy in scripts/device_c5.py, reading the phone's thermal zones and battery; its reasons are recorded per query |
| search the chosen clusters | **handset** | EcoVectorIndex.finish, the same code path the host runs |
| reduce the context (SCR) | **host only** | no device run enabled SCR; the recorded runs are retrieval only |
| generate the answer | **host only** | no device script imports a generator; flan-t5-base has never run on the handset |

**The honest summary.** Retrieval and the controller are measured on the handset, including the adaptive policy deciding from the phone's own thermal zones and battery. Embedding and generation have never run there, and the Phase 11-15 demonstration runs on the host with recorded handset telemetry replayed into it. Every host number in this document says it is a host number; this section exists so that the other half is stated too.

In one sentence: *retrieval and the controller are measured on the handset across 125,711 queries; embedding and generation run on the host, and no handset number is ever inferred from a host one.*

## 3. The control loop, and the rule it taught us

The first AIMD controller decided once per query while reading a 50-sample windowed p95. An action was invisible to its own signal for ~50 queries, so it acted on superseded decisions and ratcheted to its floor. Replayed on measured handset latency it sits at `nprobe=1` with 59% of queries on the floor, swinging across 90% of the action range.

**ADR-0032: a control loop must act no faster than its feedback delay.** Pacing control actions to the window's fill time (8.46 s, derived as 50 samples x measured p50 at nprobe=32 (169.21 ms)) removes it: median `nprobe=11`, never on the floor, 29% of the range, p95 125.7 ms against a 152 ms target.

The pre-declared criterion for that run was a reversal count, and it failed. The criterion was wrong, not the loop: AIMD reverses direction once per tooth by construction, and a controller pinned to its floor reverses *never* — it scores a perfect 0.00. The criterion was amended to amplitude, with both versions and both outcomes recorded (ADR-0032 amendment).

## 4. Why on-device RAG and an adaptive ANN, not a cloud API

Measured over the 20 labelled demonstration queries:

| | on-device | cloud API |
|---|---|---|
| user data leaving the device | **0 bytes** (measured) | 2,406 bytes per query, 48,129 in total (computed from the exact request body) |
| what is in that payload | nothing leaves | the question verbatim, plus the retrieved passages |
| works with no network | **yes**, proved with every outbound socket raising | no — an API call needs one |
| latency | p50 1660 ms, p95 2956 ms (measured, this laptop) | **not measured** — no API key was set, so no call was made |
| API cost per query | 0 | not computed |
| answers cite a passage you can open | 4 of 20 cite a local chunk id | a hosted model returns prose; any citation is not checkable locally |

**The honest other side.** A 250M-parameter local model is much weaker than a large hosted one at open-ended reasoning and general knowledge — this is not close. And on-device inference spends the phone's battery and thermal budget, which is precisely the cost this project's controller exists to manage.

## 5. How the phone-heating question is answered

Walking through one real saved trace, `docs/viva_trace.json`:

1. **The question arrives.** “Why does my phone get hot even when it is not charging?”
2. **It becomes a vector.** gte-small embeds it into the same 384-dimensional space as every passage in the corpus.
3. **The device is read.** trace-replay — measured trace c5, sample 1/237 at t+0s, hottest sensor (the hottest thermal zone inside). Temperature 55.5 °C, battery 75%.
4. **The controller decides.** temperature 55.5C >= hot 46C -> cut effort: nprobe 11 -> 3, giving up 0.1780 measured recall
5. **The index is searched.** 3 of the index's clusters are opened and 81 vectors compared. The rest of the corpus is never read.
6. **Passages come back.** Best similarity 0.935, top passage “Why a phone gets warm even when it is not plugged in”.
7. **The answer is written from them.** google/flan-t5-base, 533 prompt tokens, rephrasing rather than quoting: no passage contains the answer verbatim, which the trace records rather than claiming a citation.
8. **Nothing left the device.** 0 bytes sent, 0 connections opened — counted by watching this process's own sockets, not asserted.

> **None of this is required for a phone to get hot. Almost all of the electrical energy a phone draws from its battery ends up as heat,**

Total 2405 ms on this machine. Estimated retrieval energy 9.6 mJ (estimate — retrieval only: opening and scanning clusters).

### The link worth highlighting

The demonstration question is *why does my phone get hot when it is not charging*, and the answer the corpus gives is: because work makes heat, and heavy computation is work. Running a language model on the handset **is** that heavy computation. The system is therefore answering a question about the problem it is itself subject to, and the controller is the part that keeps the answering from making the problem worse. The demo and the contribution are the same story told twice.

## 6. Key results

| what | result | how it was obtained |
|---|---|---|
| retrieval, 10 paraphrases | recall@5 0.450, relevant@5 0.800, 0 distractors in top 5 | hand-labelled before any retrieval was run |
| answerable vs unanswerable | gap of 0.096 (0.779 to 0.874) | measured over 20 labelled queries |
| grounded-answer threshold | 0.826 | midpoint of that measured gap, not chosen |
| semantic cache threshold | 0.861 | between the paraphrase floor (0.875) and the unrelated ceiling (0.805) |
| 12.4 fixed | nprobe 16, p95 29.8 ms, recall@10 1.0000 | 300 queries over a replayed handset trace |
| 12.4 adaptive-hottest | nprobe 3, p95 7.1 ms, recall@10 0.8143, giving up 0.1857 recall for 76% lower p95 | 300 queries over a replayed handset trace |
| 12.4 adaptive-skin | nprobe 11, p95 21.5 ms, recall@10 0.9920, giving up 0.0080 recall for 28% lower p95 | 300 queries over a replayed handset trace |
| effort ladder | recall@10 0.50 at nprobe=1 rising to 1.00 at nprobe=16 | exact, against brute force |
| offline | passed 2026-10-09T04:46:42Z | every outbound socket patched to raise |

## 7. Limitations, stated plainly

- **Latency here is a laptop's.** The handset numbers come from recorded traces and are labelled as such. Nothing mixes the two.
- **Energy is an estimate.** No energy sensor exists on this machine. The per-query figure is clusters opened times a coefficient measured on the phone, and that fit is weak (r² ≈ 0.16). It is an order of magnitude, not a figure.
- **The thermal signal is real but not actionable through these thresholds.** The handset exposes two thermometers that disagree by tens of degrees. The policy's tiers were anchored on skin readings; fed the hottest internal zone it derates on every query, fed the skin it never derates at all. Same controller, same trace, same queries — recall@10 0.8143 against 0.9920 purely from the choice of sensor (ADR-0056).
- **No thermal benefit is claimed.** The heat in every trace was produced by the phone's own recorded run. Showing that the controller keeps a phone cooler would need both arms run on the phone, which has not been done.
- **The demonstration corpus is small.** 71 authored documents, 255 chunks, 16 clusters. Recall saturates by nprobe=14, so the effort ladder is short compared with the 46,369-chunk SQuAD index the benchmarks use.
- **Retrieval is not diversified per document.** One document can take several of the five slots, which is the measured cause of the recall@5 figure above.
- **The models were downloaded once.** The offline proof shows the pipeline runs with the network blocked; it does not show the models were obtained without one.

## 8. Future work

- Per-document diversification in the retrieval merge.
- Re-fit the device policy's thresholds against the sensor it will actually read.
- Run both arms of the 12.4 comparison on the handset, to close out C5.
- A time-based observation window, so the feedback delay stops being a function
  of the controller's own output (the second-order note in ADR-0032).

## 9. Ten questions an examiner is likely to ask

**1. You built on MobileRAG but dropped its per-cluster graphs. Does that not undermine the whole paper?**

It removes one mechanism and keeps the architecture. Under MobileRAG's own stated build condition we measured the graphs losing to flat mmap'd clusters at matched recall, at every point we swept. We report that as a negative result with the sweep behind it (ADR-0028) rather than shipping a component that did not earn its place.

**2. How do you know your controller is stable and not just tuned?**

The control period is derived, not searched: 50 samples times the trace's own measured latency at the top effort level. We pre-declared the stability criterion before running, published all three arms including the one that collapses, and when our own criterion failed we reported that and said why it was the wrong criterion rather than changing the setting. The paced loop holds nprobe 11 and never reaches its floor.

**3. Your refusal threshold looks like a magic number. Where did it come from?**

From a measurement. Twenty labelled queries: the answerable ones never scored below 0.874 and the unanswerable ones never above 0.779. The threshold is the midpoint of that gap. If the two groups ever overlap the calibration raises instead of returning a number.

**4. Why not just call GPT? It would answer better.**

On open-ended reasoning it would, and we say so in the comparison. What it cannot do is answer with no network, and what it must do is send the user's question and the retrieved passages off the device — we measured that at 2,406 bytes per query against a measured zero, for exactly the same questions.

**5. Is the energy number a measurement?**

No, and it is labelled an estimate everywhere it appears. This laptop has no energy sensor. The coefficient was measured on the phone by regressing its power against its cluster-access rate; the fit is weak, r² ≈ 0.16, which travels with the number. Generation energy is excluded entirely because nothing measured it.

**6. How do you know nothing is sent off the device?**

Two instruments. Every query runs inside a witness that counts this process's own socket sends, and the trace records the count. Separately, the offline proof patches every outbound path to raise and runs the whole pipeline — load included — so a hidden fetch fails instead of being noticed later. That proof caught a real one: the ML libraries check the model hub at load time even for a cached model.

**7. What does the controller actually buy?**

Over 300 queries on a replayed handset trace, against a fixed maximum depth: adaptive-hottest gives up 0.1857 recall@10 for 76% lower p95 and 81% less estimated retrieval energy; adaptive-skin gives up 0.0080 recall@10 for 28% lower p95 and 32% less estimated retrieval energy. The two differ only in which thermometer the policy reads.

**8. Your demo corpus is tiny and you wrote it yourself. Is that not circular?**

The corpus is authored and labelled as such in every record, and it is used for the demonstration, not for the benchmarks. Recall, latency and the index comparisons are measured on SQuAD, NYTimes and SIFT1M with published ground truth. The demo corpus also carries 24 deliberate distractors, several of them near misses, so retrieval has to discriminate rather than just rank.

**9. What happens when the question has no answer in the corpus?**

It refuses, before generating. The guard reads the best retrieval score, compares it against the calibrated threshold and returns 'I couldn't find this in the on-device knowledge base'. Refusing on the retrieval scores costs nothing; refusing after generating would pay for the most expensive stage and then throw it away.

**10. What is the single most important thing you learned?**

That a measurement is only as good as its provenance. Nearly every real problem we hit — the ratchet, the sensor divergence, the profile table whose latency ladder was non-monotonic because a sync client was running, the model that phones home at load — was invisible until something in the system was built to make it visible and to refuse to report what it had not measured.

## 10. Figures

Exported at 1920 px wide to `docs/figures/`:

- `comparison.png`
- `controller_check.png`
- `phase12_fixed_vs_adaptive.png`
- `benchmark_fig1_headline.png`
- `benchmark_fig_device.png`
- `benchmark_fig_footprint.png`
- `benchmark_fig_joint.png`
- `benchmark_fig_recall_effort.png`
- `benchmark_fig_regime.png`
- `benchmark_fig_scr_quality.png`
- `benchmark_fig_thermal_attribution.png`

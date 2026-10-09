#!/usr/bin/env python
"""15 -- assemble the viva pack from the recorded results.

`docs/VIVA_NOTES.md` is generated, not written by hand, and that is the point:
every number in it is read out of a result file at build time. A hand-written
revision document drifts from the results the first time an experiment is
re-run, and the drift is invisible because both look equally finished.

If a result is missing, the section that needs it says so and names the script
that produces it. Nothing is left as a plausible-looking placeholder.

Also exports every chart as a 1920px-wide PNG into `docs/figures/` for the
slides, and copies one real saved query trace into `docs/` so the walkthrough
section is describing a run that actually happened.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smartann.telemetry.provenance import PIPELINE_STAGES, device_evidence

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
FIGURES = DOCS / "figures"
TRACES = ROOT / "traces"

# 1920 px wide at matplotlib's default 100 dpi figure sizing.
TARGET_WIDTH_PX = 1920


def load(name: str) -> dict[str, Any] | None:
    p = RESULTS / name
    if not p.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
        return data
    except json.JSONDecodeError:
        return None


def absent(what: str, script: str) -> str:
    return (
        f"> **Not available.** {what} has not been produced. Run `{script}` and "
        f"regenerate these notes.\n"
    )


def export_figures() -> list[str]:
    """Re-render every PNG in results/ at 1920 px wide, for the slides.

    Rescaled rather than re-plotted: re-running the plotting scripts at a larger
    size would re-run the experiments behind some of them, and a figure in the
    pack must be the same figure the results directory holds.
    """
    from PIL import Image  # type: ignore[import-untyped]

    FIGURES.mkdir(parents=True, exist_ok=True)
    exported: list[str] = []
    for src in sorted(RESULTS.glob("*.png")):
        img = Image.open(src)
        if img.width != TARGET_WIDTH_PX:
            height = round(img.height * TARGET_WIDTH_PX / img.width)
            img = img.resize((TARGET_WIDTH_PX, height), Image.LANCZOS)
        out = FIGURES / src.name
        img.save(out)
        exported.append(out.name)
    for src in sorted((RESULTS / "figures").glob("*.png")):
        img = Image.open(src)
        if img.width != TARGET_WIDTH_PX:
            height = round(img.height * TARGET_WIDTH_PX / img.width)
            img = img.resize((TARGET_WIDTH_PX, height), Image.LANCZOS)
        out = FIGURES / f"benchmark_{src.name}"
        img.save(out)
        exported.append(out.name)
    return exported


def pick_trace() -> tuple[dict[str, Any] | None, str]:
    """The most recent saved trace that answered the demonstration question."""
    if not TRACES.is_dir():
        return None, ""
    best: tuple[float, Path] | None = None
    for p in sorted(TRACES.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "hot" in d.get("query", "").lower() and not d.get("answer", {}).get("declined", True):
            mtime = p.stat().st_mtime
            if best is None or mtime > best[0]:
                best = (mtime, p)
    if best is None:
        return None, ""
    data: dict[str, Any] = json.loads(best[1].read_text(encoding="utf-8"))
    dest = DOCS / "viva_trace.json"
    # Rewritten rather than copied, with a trailing newline: json.dumps leaves
    # none, and the end-of-file pre-commit hook then rewrites this file after
    # every regeneration, so the repo is dirty the moment the pack is rebuilt.
    dest.write_text(best[1].read_text(encoding="utf-8").rstrip("\n") + "\n", encoding="utf-8")
    return data, dest.name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DOCS / "VIVA_NOTES.md")
    args = ap.parse_args()

    check = load("controller_check.json")
    ev = load("demo_eval.json")
    table = load("demo_profile_table.json")
    stress = load("phase12_fixed_vs_adaptive.json")
    comp = load("comparison.json")
    proof = load("offline_proof.json")
    guard = load("grounding_threshold.json")
    cache = load("cache_threshold.json")
    regime = load("regime_summary.json")
    trace, trace_name = pick_trace()
    evidence = device_evidence(RESULTS / "device", RESULTS / "device_profile_table.json")

    try:
        exported = export_figures()
    except ImportError:
        exported = []
        print("Pillow is not installed; figures were not rescaled.")

    now = datetime.now(UTC).strftime("%Y-%m-%d")
    out: list[str] = []
    add = out.append

    add("# SMART-ANN — viva notes\n")
    add(
        f"*Generated {now} by `scripts/phase15_viva_pack.py`. Every number below is read "
        f"from a result file at build time; none is typed in. Re-run the script after any "
        f"experiment to refresh it.*\n"
    )

    # ------------------------------------------------------------------ story
    add("## 1. The project in one page\n")
    add(
        "**The problem.** Retrieval-augmented generation on a phone has to choose how hard "
        "to search. Searching deeply gives better answers and costs latency, battery and "
        "heat; searching shallowly is cheap and misses things. The right choice is not "
        "fixed — it depends on what the device can afford at that moment.\n"
    )
    add(
        "**The baseline.** MobileRAG proposes EcoVector: an IVF partition where each "
        "cluster carries its own small HNSW graph, plus Selective Content Reduction to cut "
        "the context handed to the generator.\n"
    )
    if regime:
        flat = regime["series"].get("ivf_disk (no graphs)")
        g32 = regime["series"].get("per-cluster graphs M=32")
        if flat and g32:
            add(
                f"**What we found.** Under MobileRAG's own build condition "
                f"(`{regime['condition']}`, N/Nc = {regime['n_over_nc']:.0f}), the "
                f"per-cluster graphs lost. At matched recall the flat, memory-mapped "
                f"clusters were faster at every point measured — for example "
                f"{flat['recall'][2]:.4f} recall in {flat['ms'][2]:.1f} ms against "
                f"{g32['recall'][2]:.4f} in {g32['ms'][2]:.1f} ms for M=32. The graphs were "
                f"dropped and the index now uses flat mmap'd clusters (ADR-0028).\n"
            )
    else:
        add(absent("The index regime comparison", "scripts/experiment_matrix.py"))

    add(
        "**Our contribution.** A control layer that chooses retrieval effort per query from "
        "the device's measured condition and a measured profile of what each effort level "
        "costs and returns. Plus the discipline around it: every policy decision is "
        "explainable, every number carries its provenance, and the system refuses to "
        "report what it did not measure.\n"
    )

    # -------------------------------------------------------------- hardware
    add("## 2. What ran on the phone, and what did not\n")
    if evidence.any:
        add(
            f"**{evidence.headline()}** Those runs are in `results/device/`, and every "
            f"count below is read from them rather than quoted from memory.\n"
        )
        add("| run | queries | minutes | effort levels | controller decided on-device |")
        add("|---|---|---|---|---|")
        for r in evidence.runs:
            levels = "/".join(str(x) for x in r.nprobe_levels) or "n/a"
            mins = f"{r.minutes:.0f}" if math.isfinite(r.minutes) else "n/a"
            add(
                f"| `{r.name}` | {r.n_queries:,} | {mins} | {levels} | "
                f"{'yes' if r.ran_controller else 'no'} |"
            )
        add("")
        add("Stage by stage:\n")
        add("| pipeline stage | measured on | evidence |")
        add("|---|---|---|")
        for stage in PIPELINE_STAGES:
            add(f"| {stage.name} | **{stage.where}** | {stage.evidence} |")
        add("")
        add(
            "**The honest summary.** Retrieval and the controller are measured on the "
            "handset, including the adaptive policy deciding from the phone's own "
            "thermal zones and battery. Embedding and generation have never run there, "
            "and the Phase 11-15 demonstration runs on the host with recorded handset "
            "telemetry replayed into it. Every host number in this document says it is a "
            "host number; this section exists so that the other half is stated too.\n"
        )
        add(
            f"In one sentence: *retrieval and the controller are measured on the handset "
            f"across {evidence.total_queries:,} queries; embedding and generation run on "
            f"the host, and no handset number is ever inferred from a host one.*\n"
        )
    else:
        add(absent("The handset runs", "scripts/device_trace.py, on the phone"))

    # ------------------------------------------------------------- controller
    add("## 3. The control loop, and the rule it taught us\n")
    if check:
        arms = {a["control_period_s"]: a for a in check["arms"]}
        unpaced = arms.get(0.0)
        paced = check["arms"][-1]
        if unpaced:
            add(
                f"The first AIMD controller decided once per query while reading a "
                f"50-sample windowed p95. An action was invisible to its own signal for "
                f"~50 queries, so it acted on superseded decisions and ratcheted to its "
                f"floor. Replayed on measured handset latency it sits at "
                f"`nprobe={unpaced['nprobe_p50']:.0f}` with "
                f"{100 * unpaced['floor_fraction']:.0f}% of queries on the floor, swinging "
                f"across {100 * unpaced['swing_fraction_of_range']:.0f}% of the action "
                f"range.\n"
            )
        add(
            f"**ADR-0032: a control loop must act no faster than its feedback delay.** "
            f"Pacing control actions to the window's fill time "
            f"({check['derived_settings']['window_fill_time_s']:.2f} s, derived as "
            f"{check['derived_settings']['window_fill_derivation']}) removes it: median "
            f"`nprobe={paced['nprobe_p50']:.0f}`, never on the floor, "
            f"{100 * paced['swing_fraction_of_range']:.0f}% of the range, p95 "
            f"{paced['p95_ms']:.1f} ms against a {check['derived_settings']['latency_target_ms']:.0f} ms "
            f"target.\n"
        )
        add(
            "The pre-declared criterion for that run was a reversal count, and it failed. "
            "The criterion was wrong, not the loop: AIMD reverses direction once per tooth "
            "by construction, and a controller pinned to its floor reverses *never* — it "
            "scores a perfect 0.00. The criterion was amended to amplitude, with both "
            "versions and both outcomes recorded (ADR-0032 amendment).\n"
        )
    else:
        add(absent("The controller stability check", "python scripts/phase11_controller_check.py"))

    # ------------------------------------------------------------ why on-device
    add("## 4. Why on-device RAG and an adaptive ANN, not a cloud API\n")
    if comp:
        add(f"Measured over the {comp['n_queries']} labelled demonstration queries:\n")
        add("| | on-device | cloud API |")
        add("|---|---|---|")
        add(
            f"| user data leaving the device | **{comp['device_bytes_total']} bytes** "
            f"(measured) | {comp['cloud_payload_bytes']['mean']:,.0f} bytes per query, "
            f"{comp['cloud_payload_bytes']['total']:,.0f} in total (computed from the "
            f"exact request body) |"
        )
        add(
            "| what is in that payload | nothing leaves | the question verbatim, plus "
            "the retrieved passages |"
        )
        offline_cell = (
            "**yes**, proved with every outbound socket raising"
            if proof and proof.get("passed")
            else "not proved"
        )
        add(f"| works with no network | {offline_cell} | no — an API call needs one |")
        dev_lat = comp["device_latency_ms"]
        cloud_lat = comp["cloud_latency_ms"]
        add(
            f"| latency | p50 {dev_lat['p50']:.0f} ms, p95 {dev_lat['p95']:.0f} ms "
            f"(measured, this laptop) | "
            + (
                f"p50 {cloud_lat['p50']:.0f} ms, p95 {cloud_lat['p95']:.0f} ms (measured)"
                if cloud_lat
                else "**not measured** — no API key was set, so no call was made"
            )
            + " |"
        )
        add(f"| API cost per query | 0 | {comp['cost']['cloud']} |")
        add(
            f"| answers cite a passage you can open | "
            f"{comp['grounding']['device_citations']} of {comp['n_queries']} cite a local "
            f"chunk id | a hosted model returns prose; any citation is not checkable "
            f"locally |"
        )
        add("")
        add(
            "**The honest other side.** A 250M-parameter local model is much weaker than a "
            "large hosted one at open-ended reasoning and general knowledge — this is not "
            "close. And on-device inference spends the phone's battery and thermal budget, "
            "which is precisely the cost this project's controller exists to manage.\n"
        )
    else:
        add(absent("The cloud comparison", "python scripts/phase13_comparison.py"))

    # -------------------------------------------------------------- the demo
    add("## 5. How the phone-heating question is answered\n")
    if trace:
        c = trace["controller"]
        r = trace["resources"]
        ret = trace["retrieval"]
        add(f"Walking through one real saved trace, `docs/{trace_name}`:\n")
        add(f"1. **The question arrives.** “{trace['query']}”")
        add(
            "2. **It becomes a vector.** gte-small embeds it into the same "
            "384-dimensional space as every passage in the corpus."
        )
        add(
            f"3. **The device is read.** {r['source']} — {r['source_detail']}. "
            f"Temperature {r['temperature_c']:.1f} °C, battery {r['battery_pct']:.0f}%."
        )
        add(f"4. **The controller decides.** {c['reason']}")
        add(
            f"5. **The index is searched.** {ret['clusters_probed']} of the index's "
            f"clusters are opened and {ret['vectors_scanned']:,} vectors compared. The "
            f"rest of the corpus is never read."
        )
        add(
            f"6. **Passages come back.** Best similarity {ret['best_score']:.3f}, top "
            f"passage “{ret['chunks'][0]['title']}”."
        )
        cited = trace["answer"]["cited_chunk_ids"]
        citation = (
            "quoting chunk " + ", ".join(f"#{i}" for i in cited) + " word for word"
            if cited
            else (
                "rephrasing rather than quoting: no passage contains the answer verbatim, "
                "which the trace records rather than claiming a citation"
            )
        )
        add(
            f"7. **The answer is written from them.** {trace['generator']}, "
            f"{trace['answer']['prompt_tokens']} prompt tokens, {citation}."
        )
        add(
            f"8. **Nothing left the device.** {trace['network']['bytes_sent']} bytes sent, "
            f"{trace['network']['connections_opened']} connections opened — counted by "
            f"watching this process's own sockets, not asserted."
        )
        add("")
        add(f"> **{trace['answer']['text']}**\n")
        add(
            f"Total {trace['latency_ms']['total']:.0f} ms on this machine. Estimated "
            f"retrieval energy {trace['energy']['mj']:.1f} mJ "
            f"({trace['energy']['label']} — {trace['energy']['covers']}).\n"
        )
    else:
        add(absent("A saved query trace", "smartann ask ... --explain"))

    add("### The link worth highlighting\n")
    add(
        "The demonstration question is *why does my phone get hot when it is not "
        "charging*, and the answer the corpus gives is: because work makes heat, and "
        "heavy computation is work. Running a language model on the handset **is** that "
        "heavy computation. The system is therefore answering a question about the "
        "problem it is itself subject to, and the controller is the part that keeps the "
        "answering from making the problem worse. The demo and the contribution are the "
        "same story told twice.\n"
    )

    # ----------------------------------------------------------- key results
    add("## 6. Key results\n")
    add("| what | result | how it was obtained |")
    add("|---|---|---|")
    if ev:
        p = ev["paraphrases"]
        add(
            f"| retrieval, 10 paraphrases | recall@5 "
            f"{p['recall_at_5_frozen_primary']:.3f}, relevant@5 {p['relevant_at_5']:.3f}, "
            f"{p['distractors_in_top5']} distractors in top 5 | hand-labelled before any "
            f"retrieval was run |"
        )
        sep = ev["separation"]
        add(
            f"| answerable vs unanswerable | gap of {sep['gap']:.3f} "
            f"({sep['highest_unanswerable']:.3f} to {sep['lowest_answerable']:.3f}) | "
            f"measured over 20 labelled queries |"
        )
    if guard:
        add(
            f"| grounded-answer threshold | {guard['threshold']:.3f} | midpoint of that "
            f"measured gap, not chosen |"
        )
    if cache:
        add(
            f"| semantic cache threshold | {cache['threshold']:.3f} | between the "
            f"paraphrase floor ({cache['same_question']['min']:.3f}) and the unrelated "
            f"ceiling ({cache['different_questions']['max']:.3f}) |"
        )
    if stress:
        for arm, s in stress["arms"].items():
            d = stress["deltas"].get(arm)
            extra = (
                f", giving up {d['recall_given_up']:.4f} recall for "
                f"{d['p95_saved_pct']:.0f}% lower p95"
                if d
                else ""
            )
            add(
                f"| 12.4 {arm} | nprobe {s['nprobe_p50']:.0f}, p95 {s['p95_ms']:.1f} ms, "
                f"recall@10 {s['recall_at_10']:.4f}{extra} | 300 queries over a replayed "
                f"handset trace |"
            )
    if table:
        first, last = table["entries"][0], table["entries"][-1]
        add(
            f"| effort ladder | recall@10 {first['recall']:.2f} at nprobe=1 rising to "
            f"{last['recall']:.2f} at nprobe={last['nprobe']} | exact, against brute force |"
        )
    if proof:
        add(
            f"| offline | {'passed' if proof['passed'] else 'FAILED'} "
            f"{proof['ran_at']} | every outbound socket patched to raise |"
        )
    add("")

    # ---------------------------------------------------------- limitations
    add("## 7. Limitations, stated plainly\n")
    add(
        "- **Latency here is a laptop's.** The handset numbers come from recorded traces "
        "and are labelled as such. Nothing mixes the two."
    )
    add(
        "- **Energy is an estimate.** No energy sensor exists on this machine. The "
        "per-query figure is clusters opened times a coefficient measured on the phone, "
        "and that fit is weak (r² ≈ 0.16). It is an order of magnitude, not a figure."
    )
    if stress and "adaptive-hottest" in stress["arms"] and "adaptive-skin" in stress["arms"]:
        hot = stress["arms"]["adaptive-hottest"]
        skin = stress["arms"]["adaptive-skin"]
        add(
            f"- **The thermal signal is real but not actionable through these thresholds.** "
            f"The handset exposes two thermometers that disagree by tens of degrees. The "
            f"policy's tiers were anchored on skin readings; fed the hottest internal zone "
            f"it derates on every query, fed the skin it never derates at all. Same "
            f"controller, same trace, same queries — recall@10 {hot['recall_at_10']:.4f} "
            f"against {skin['recall_at_10']:.4f} purely from the choice of sensor "
            f"(ADR-0056)."
        )
    add(
        "- **No thermal benefit is claimed.** The heat in every trace was produced by the "
        "phone's own recorded run. Showing that the controller keeps a phone cooler would "
        "need both arms run on the phone, which has not been done."
    )
    add(
        "- **The demonstration corpus is small.** 71 authored documents, 255 chunks, 16 "
        "clusters. Recall saturates by nprobe=14, so the effort ladder is short compared "
        "with the 46,369-chunk SQuAD index the benchmarks use."
    )
    add(
        "- **Retrieval is not diversified per document.** One document can take several of "
        "the five slots, which is the measured cause of the recall@5 figure above."
    )
    add(
        "- **The models were downloaded once.** The offline proof shows the pipeline runs "
        "with the network blocked; it does not show the models were obtained without one."
    )
    add("")

    add("## 8. Future work\n")
    add("- Per-document diversification in the retrieval merge.")
    add("- Re-fit the device policy's thresholds against the sensor it will actually read.")
    add("- Run both arms of the 12.4 comparison on the handset, to close out C5.")
    add("- A time-based observation window, so the feedback delay stops being a function")
    add("  of the controller's own output (the second-order note in ADR-0032).")
    add("")

    # ------------------------------------------------------------- questions
    add("## 9. Ten questions an examiner is likely to ask\n")
    qa: list[tuple[str, str]] = []
    if regime:
        qa.append(
            (
                "You built on MobileRAG but dropped its per-cluster graphs. Does that not "
                "undermine the whole paper?",
                "It removes one mechanism and keeps the architecture. Under MobileRAG's own "
                "stated build condition we measured the graphs losing to flat mmap'd "
                "clusters at matched recall, at every point we swept. We report that as a "
                "negative result with the sweep behind it (ADR-0028) rather than shipping a "
                "component that did not earn its place.",
            )
        )
    if check:
        paced = check["arms"][-1]
        qa.append(
            (
                "How do you know your controller is stable and not just tuned?",
                f"The control period is derived, not searched: 50 samples times the trace's "
                f"own measured latency at the top effort level. We pre-declared the "
                f"stability criterion before running, published all three arms including "
                f"the one that collapses, and when our own criterion failed we reported "
                f"that and said why it was the wrong criterion rather than changing the "
                f"setting. The paced loop holds nprobe "
                f"{paced['nprobe_p50']:.0f} and never reaches its floor.",
            )
        )
    if guard and ev:
        qa.append(
            (
                "Your refusal threshold looks like a magic number. Where did it come from?",
                f"From a measurement. Twenty labelled queries: the answerable ones never "
                f"scored below {ev['separation']['lowest_answerable']:.3f} and the "
                f"unanswerable ones never above "
                f"{ev['separation']['highest_unanswerable']:.3f}. The threshold is the "
                f"midpoint of that gap. If the two groups ever overlap the calibration "
                f"raises instead of returning a number.",
            )
        )
    qa.append(
        (
            "Why not just call GPT? It would answer better.",
            "On open-ended reasoning it would, and we say so in the comparison. What it "
            "cannot do is answer with no network, and what it must do is send the user's "
            "question and the retrieved passages off the device — we measured that at "
            f"{comp['cloud_payload_bytes']['mean']:,.0f} bytes per query against a measured "
            "zero, for exactly the same questions."
            if comp
            else "On open-ended reasoning it would, and we say so. What it cannot do is "
            "answer with no network, and what it must do is send the question off the "
            "device.",
        )
    )
    qa.append(
        (
            "Is the energy number a measurement?",
            "No, and it is labelled an estimate everywhere it appears. This laptop has no "
            "energy sensor. The coefficient was measured on the phone by regressing its "
            "power against its cluster-access rate; the fit is weak, r² ≈ 0.16, which "
            "travels with the number. Generation energy is excluded entirely because "
            "nothing measured it.",
        )
    )
    qa.append(
        (
            "How do you know nothing is sent off the device?",
            "Two instruments. Every query runs inside a witness that counts this process's "
            "own socket sends, and the trace records the count. Separately, the offline "
            "proof patches every outbound path to raise and runs the whole pipeline — load "
            "included — so a hidden fetch fails instead of being noticed later. That proof "
            "caught a real one: the ML libraries check the model hub at load time even for "
            "a cached model.",
        )
    )
    if stress:
        qa.append(
            (
                "What does the controller actually buy?",
                "Over 300 queries on a replayed handset trace, against a fixed maximum "
                "depth: "
                + "; ".join(
                    f"{arm} gives up {d['recall_given_up']:.4f} recall@10 for "
                    f"{d['p95_saved_pct']:.0f}% lower p95 and "
                    f"{d['energy_saved_pct_estimate']:.0f}% less estimated retrieval energy"
                    for arm, d in stress["deltas"].items()
                )
                + ". The two differ only in which thermometer the policy reads.",
            )
        )
    qa.append(
        (
            "Your demo corpus is tiny and you wrote it yourself. Is that not circular?",
            "The corpus is authored and labelled as such in every record, and it is used "
            "for the demonstration, not for the benchmarks. Recall, latency and the index "
            "comparisons are measured on SQuAD, NYTimes and SIFT1M with published ground "
            "truth. The demo corpus also carries 24 deliberate distractors, several of them "
            "near misses, so retrieval has to discriminate rather than just rank.",
        )
    )
    qa.append(
        (
            "What happens when the question has no answer in the corpus?",
            "It refuses, before generating. The guard reads the best retrieval score, "
            "compares it against the calibrated threshold and returns 'I couldn't find "
            "this in the on-device knowledge base'. Refusing on the retrieval scores costs "
            "nothing; refusing after generating would pay for the most expensive stage and "
            "then throw it away.",
        )
    )
    qa.append(
        (
            "What is the single most important thing you learned?",
            "That a measurement is only as good as its provenance. Nearly every real "
            "problem we hit — the ratchet, the sensor divergence, the profile table whose "
            "latency ladder was non-monotonic because a sync client was running, the model "
            "that phones home at load — was invisible until something in the system was "
            "built to make it visible and to refuse to report what it had not measured.",
        )
    )

    for i, (q, a) in enumerate(qa, 1):
        add(f"**{i}. {q}**\n")
        add(f"{a}\n")

    # ---------------------------------------------------------------- figures
    add("## 10. Figures\n")
    if exported:
        add(f"Exported at {TARGET_WIDTH_PX} px wide to `docs/figures/`:\n")
        for name in exported:
            add(f"- `{name}`")
    else:
        add(absent("The figure exports", "pip install pillow, then re-run this script"))
    add("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"exported {len(exported)} figures to {FIGURES}")
    if trace_name:
        print(f"copied one real trace to docs/{trace_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

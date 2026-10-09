"""Render one answered question as a readable trace of the whole pipeline.

Separate from `ask.py` for the same reason the thermal report is separate from
its attribution: that module computes and this one formats, so changing the
wording cannot change a number.

The order is the pipeline's own. A reader should be able to follow a question
down through embedding, the controller's choice, retrieval and reduction, and
arrive at the answer having seen what it was built from. The answer is printed
after the evidence rather than before it, because the point of a retrieval
system is that the answer is a consequence of what it found.
"""

from __future__ import annotations

import math
import textwrap

from smartann.pipeline.ask import AskResult

__all__ = ["render"]

WIDTH = 78


def _rule(ch: str = "-") -> str:
    return ch * WIDTH


def _wrap(text: str, indent: str = "      ") -> str:
    return textwrap.fill(
        " ".join(text.split()),
        width=WIDTH - len(indent),
        initial_indent=indent,
        subsequent_indent=indent,
    )


def _num(value: float, unit: str, sign: bool = False) -> str:
    """A measured number, or "not measured" -- never a zero standing in for one."""
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "not measured"
    return f"{value:+.3f} {unit}" if sign else f"{value:.1f} {unit}"


def _bar(fraction: float, width: int = 28) -> str:
    """A proportion drawn in blocks, for reading costs at a glance."""
    if not math.isfinite(fraction) or fraction <= 0:
        return " " * width
    filled = max(1, min(width, round(fraction * width)))
    return "#" * filled + "-" * (width - filled)


def render(r: AskResult, show_context: bool = True, snippet_chars: int = 220) -> str:
    out: list[str] = []
    add = out.append

    add(_rule("="))
    add("  SMART-ANN  --  retrieve, control, generate".ljust(WIDTH))
    add(_rule("="))
    add("")
    add("  QUESTION")
    add(_wrap(r.question, "      "))
    add("")

    add(_rule())
    add("  1. CONTROLLER -- what effort to spend, decided before searching")
    add(_rule())
    add("")
    add(f"    policy              {r.policy_name}")
    if r.budget_ms is not None:
        add(f"    latency budget      {r.budget_ms:.0f} ms")
    else:
        add("    latency budget      none given, so effort was not constrained")
    add(f"    chose nprobe        {r.ann.nprobe}   (clusters to open)")
    add(f"    decision cost       {r.controller_overhead_us:.1f} us")

    if r.device is not None:
        d = r.device
        add("")
        add("    DEVICE CONDITION")
        add(f"      {r.device_source}")
        if not r.device_is_measured:
            add("      HYPOTHETICAL. Not a measurement: this state was typed, to")
            add("      ask what the controller would do in it.")
        temp = d.get("temperature_c", float("nan"))
        add(f"      temperature       {_num(temp, 'C')}")
        slope = d.get("temperature_slope_c_per_s", float("nan"))
        add(f"      rate of change    {_num(slope, 'C/s', sign=True)}")
        pct = d.get("battery_pct", float("nan"))
        charge = "charging" if d.get("charging") else "on battery"
        add(f"      battery           {_num(pct, '%')}   {charge}")
        draw = d.get("discharge_rate_ma", float("nan"))
        add(f"      current           {_num(draw, 'mA')}")
        add("")
        if r.baseline_nprobe is not None and r.derated:
            add(f"      DERATED: the budget alone allowed nprobe={r.baseline_nprobe};")
            add(f"      the device state cut it to nprobe={r.ann.nprobe}.")
            lost = r.recall_given_up
            if math.isfinite(lost):
                add(f"      That gives up {lost:.4f} measured recall, at those two")
                add("      levels over the benchmark set.")
        elif r.baseline_nprobe is not None:
            add(f"      No derate: nprobe={r.ann.nprobe} is what the budget allowed, and")
            add("      no threshold in the policy was crossed.")
        if r.device_signals_absent:
            add("      The policy acted on nothing: every signal it reads was NaN")
            add("      on this platform. That is absence, not a healthy device.")
        elif r.policy_reasons:
            add(f"      signals used: {'; '.join(r.policy_reasons)}")
        else:
            add("      The signals were read and none crossed a threshold. That is a")
            add("      device in good condition, not a device that was not measured.")

    if r.profile_entry is not None:
        e = r.profile_entry
        add("")
        add(f"    picked from the measured profile table, level nprobe={e.nprobe}:")
        add(f"      measured p95      {e.p95_ms:.1f} ms   over {e.n_samples:,} recorded queries")
        add(f"      measured recall   {e.recall:.4f}")
        add(f"      measured under    {r.table_measured_under}")
        add("")
        add("    That recall is the LEVEL's, measured over the benchmark set. It is")
        add("    not this question's recall, which has no ground truth to score")
        add("    against and is therefore not reported.")

    if r.policy_reasons:
        add("")
        # Not "device signals": the query-adaptive policy reports a difficulty
        # signal here, and labelling it as device state would attribute a
        # decision to a thermometer that was never read.
        add(f"    signals used: {', '.join(r.policy_reasons)}")
    add("")

    add(_rule())
    add(f"  2. RETRIEVAL -- top {len(r.chunks)} from {r.dataset}")
    add(_rule())
    add("")
    add(f"    {r.n_cluster_loads} clusters opened, {r.n_dist_exact:,} exact distances computed")
    add("")
    n_kb = sum(1 for c in r.chunks if c.origin == "kb")
    if n_kb:
        add(f"    {n_kb} of {len(r.chunks)} passages came from your knowledge base,")
        add("    ranked against the benchmark index on the same cosine scale.")
        add("")
    for c in r.chunks:
        where = f"{c.title} ({c.kind_label})" if c.title else f"chunk #{c.chunk_id}"
        add(f"    [{c.rank}]  similarity {c.similarity:.3f}   {where}")
        if show_context:
            body = " ".join(c.reduced_text.split())
            if len(body) > snippet_chars:
                body = body[:snippet_chars].rstrip() + " ..."
            add(_wrap(body, "         "))
        add("")

    add(_rule())
    add("  3. CONTEXT REDUCTION")
    add(_rule())
    add("")
    if r.scr.enabled:
        cut = 100.0 * (1.0 - r.chars_out / r.chars_in) if r.chars_in else 0.0
        add(f"    on, top_windows={r.scr.top_windows}")
        add(f"    {r.chars_in:,} chars -> {r.chars_out:,} chars    {cut:.1f}% removed")
        add("")
        add("    Reduction costs quality at every level measured in Phase 7, so it")
        add("    is off unless asked for. Pass --scr to see the trade.")
    else:
        add("    off -- byte-identical passthrough (ADR-0035)")
        add(f"    {r.chars_in:,} chars of context kept in full")
    add("")

    add(_rule("="))
    add("  ANSWER")
    add(_rule("="))
    add("")
    if r.from_model_memory:
        add(_wrap(r.answer, "      "))
        add("")
        add("      FROM THE MODEL'S MEMORY, NOT FROM THE CORPUS. The retrieved")
        add("      passages did not contain the answer, so this was asked with no")
        add("      passages at all. Nothing checks it: the model learned from a fixed")
        add("      snapshot, so anything that changes over time may be out of date,")
        add("      and a small model can simply misremember.")
    elif r.refused_ungrounded:
        # Distinct from a model that declined. Nothing was generated here: the
        # guard stopped before the generator ran, so saying the model "declined"
        # would credit a decision to a component that was never asked.
        add(_wrap(r.answer, "      "))
        add("")
        verdict = r.guard or {}
        add(
            _wrap(
                f"REFUSED BEFORE GENERATING. The best passage scored "
                f"{verdict.get('best_score', float('nan')):.3f}, below the "
                f"{verdict.get('threshold', float('nan')):.3f} threshold, so the corpus was judged "
                f"not to contain the answer and the model was never asked.",
                "      ",
            )
        )
        add("")
        add(
            _wrap(
                "That threshold is measured, not chosen: over the labelled evaluation "
                "set, no answerable question scored below it and no unanswerable one "
                "scored above it.",
                "      ",
            )
        )
    elif r.declined:
        add(_wrap(r.answer, "      "))
        add("")
        if r.overruled:
            add(
                _wrap(
                    f'The extractive step proposed "{r.extracted_answer}", but the '
                    f"grounding check judged that the passages do not contain the answer "
                    f"(p = {r.verify_p_yes:.2f}), so it is withheld rather than shown as "
                    f"a finding.",
                    "      ",
                )
            )
        else:
            add("      The model declined: the passages retrieved do not contain the")
            add("      answer, and it said so rather than inventing one.")
        add("")
        add("      Pass --fallback to also ask the model from its own memory,")
        add("      labelled as such.")
    else:
        add(_wrap(r.answer, "      "))
        if math.isfinite(r.verify_p_yes):  # NaN when the check did not run
            add("")
            add(f"      grounding check: passages contain the answer, p = {r.verify_p_yes:.2f}")
    add("")
    cited = r.cited
    if cited is not None:
        where = cited.title or f"chunk #{cited.chunk_id}"
        add(f"      SOURCE: passage [{cited.rank}], {where}")
        if cited.url:
            add(f"      {cited.url}")
        if cited.added_at:
            add(f"      added to your knowledge base {cited.added_at}")
    elif not r.declined:
        add("      No passage contains this answer word for word, so the model")
        add("      rephrased or combined. Read the passages above before quoting it.")
    add("")
    if r.refused_ungrounded:
        add("      Nothing was generated. The generator was not run.")
    elif r.cache_hit:
        # Not `c`: that name is the chunk in the retrieval loop above, and
        # rebinding it here shadowed a RetrievedChunk with a dict.
        reuse = r.cache or {}
        add(
            f"      REUSED from the semantic cache: this question matched "
            f"{reuse.get('matched_question', '')!r} at similarity "
            f"{reuse.get('similarity', float('nan')):.3f}."
        )
        add(
            f"      The original run cost {reuse.get('saved_ms', float('nan')):.0f} ms, "
            f"which is what this query avoided."
        )
    else:
        add(f"      generated by {r.generator_name}, greedy decoding")
    if r.truncated:
        add("      WARNING: the prompt was truncated; some retrieved context never")
        add("      reached the model.")
    add("")

    add(_rule())
    add("  COST -- measured on this machine, this query")
    add(_rule())
    add("")
    total = r.total_ms
    for s in r.stages:
        share = s.ms / total if total > 0 else 0.0
        add(f"    {s.name:<9} {s.ms:8.1f} ms  {_bar(share)}  {100 * share:4.1f}%")
    add(f"    {'TOTAL':<9} {total:8.1f} ms")
    add("")
    for s in r.stages:
        if s.detail:
            add(f"      {s.name}: {s.detail}")
    add("")
    add("    Laptop numbers, not device numbers. The handset's measured costs are")
    add("    in the profile table above and in results/device_profile_table.json.")
    add("")
    add(_rule("="))
    return "\n".join(out)

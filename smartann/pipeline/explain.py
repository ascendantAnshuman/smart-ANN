"""Render a `QueryTrace` as a step-by-step story of how an answer was made.

`ask_report.render` already prints a finished query as a technical report:
sections, columns, percentages. This prints the same run as a sequence a person
can follow out loud -- what arrived, what the device looked like, what the
controller therefore decided, what that decision opened, and what came back --
because the demonstration has to be narrated to an audience that has not read
the code.

Computing happens in `querytrace.py` and formatting happens here, so changing
the wording cannot change a number. Every value printed is read from the trace;
nothing is recomputed, and anything the trace records as absent prints as
"not measured" rather than as a zero.
"""

from __future__ import annotations

import math
import textwrap

from smartann.pipeline.querytrace import QueryTrace
from smartann.telemetry.provenance import DeviceEvidence

__all__ = ["render_story"]

WIDTH = 78


def _num(value: float, unit: str, places: int = 1, sign: bool = False) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "not measured"
    return f"{value:+.{places}f} {unit}" if sign else f"{value:.{places}f} {unit}"


def _wrap(text: str, indent: str = "        ", first: str | None = None) -> str:
    """Wrap to the page width. `first` marks only the first line.

    Without the split a bullet or an arrow passed as the indent is repeated on
    every continuation line, so one sentence reads as several conclusions.
    """
    return textwrap.fill(
        " ".join(text.split()),
        width=WIDTH,
        initial_indent=first if first is not None else indent,
        subsequent_indent=indent,
    )


def _plural(n: int, singular: str, plural: str | None = None) -> str:
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


def _step(n: int, title: str) -> list[str]:
    return ["", f"  [{n}] {title}", "  " + "-" * (WIDTH - 4)]


def render_story(t: QueryTrace, evidence: DeviceEvidence | None = None) -> str:
    """Render the trace as a story.

    `evidence` is passed in rather than read here, because scanning the
    recorded handset runs costs tens of milliseconds and this is a formatter.
    The caller computes it once per session; omitted, the hardware line is
    simply absent rather than guessed at.
    """
    out: list[str] = []
    add = out.append

    add("=" * WIDTH)
    add("  HOW THIS ANSWER WAS MADE")
    add(f"  {t.timestamp}   {t.dataset}   everything below ran on this device")
    add("=" * WIDTH)

    # ---------------------------------------------------------------- 1
    out += _step(1, "THE QUESTION ARRIVES")
    add(_wrap(f'"{t.query}"'))

    # ---------------------------------------------------------------- 2
    out += _step(2, "IT BECOMES A VECTOR")
    embed = next((s for s in t.stages if s.name == "embed"), None)
    if embed is not None:
        add(_wrap(f"{embed.detail} -- {embed.ms:.1f} ms"))
    add(
        _wrap(
            "The question and every passage in the corpus are turned into points in "
            "the same 384-dimensional space, so 'similar in meaning' becomes "
            "'close together' and retrieval becomes a geometry problem."
        )
    )

    # ---------------------------------------------------------------- 3
    out += _step(3, "THE DEVICE IS READ")
    r = t.resources
    add(f"        source       {r.source}: {r.source_detail}")
    add(f"        measured on  {r.measured_on}")
    if r.source == "hypothetical":
        add("        THIS IS NOT A MEASUREMENT. It was typed, to ask what the")
        add("        controller would do in a state like this.")
    add(
        f"        temperature  {_num(r.temperature_c, 'C')}"
        f"   (rate {_num(r.temperature_slope_c_per_s, 'C/s', 3, sign=True)})"
    )
    add(
        f"        battery      {_num(r.battery_pct, '%', 0)}"
        f"   {'charging' if r.charging else 'on battery'}"
    )
    add(
        f"        cpu          {_num(100 * r.cpu_util, '%', 0) if math.isfinite(r.cpu_util) else 'not measured'}"
    )
    add(f"        free RAM     {_num(r.mem_available_mb, 'MB', 0)}")
    absent = r.absent_fields()
    if absent:
        add("")
        add(
            _wrap(
                f"Not available on this platform: {', '.join(absent)}. These are "
                f"absent, not zero, and every policy in this project declines to act "
                f"on a reading it did not get."
            )
        )

    # ---------------------------------------------------------------- 4
    out += _step(4, "THE CONTROLLER DECIDES HOW HARD TO LOOK")
    c = t.controller
    add(f"        policy       {c.policy}")
    if c.budget_ms is not None:
        add(f"        budget       {c.budget_ms:.0f} ms")
    add(f"        decided in   {c.overhead_us:.1f} us")
    add("")
    add(_wrap(c.reason, indent="           ", first="        -> "))
    if c.derated and c.baseline_nprobe is not None:
        add("")
        add(
            _wrap(
                f"Without the device signal it would have opened {c.baseline_nprobe} "
                f"clusters. It opened {c.nprobe}. That is the adaptation: the same "
                f"question, answered with less work, because the handset was in a "
                f"state where more work costs more than it returns."
            )
        )
    if math.isfinite(c.level_recall):
        add("")
        add(f"        at nprobe={c.nprobe}, this LEVEL measured:")
        add(f"          recall {c.level_recall:.4f}   p95 {c.level_p95_ms:.1f} ms")
        add(_wrap(f"measured under: {c.table_measured_under}", "            "))
        add(
            _wrap(
                "That is the level's number over the evaluation set, not this "
                "question's: a typed question has no ground truth to be scored "
                "against, so none is reported."
            )
        )

    # ---------------------------------------------------------------- 5
    out += _step(5, "THE INDEX IS SEARCHED")
    search = next((s for s in t.stages if s.name == "search"), None)
    add(
        f"        {_plural(t.clusters_probed, 'cluster')} opened, "
        f"{t.vectors_scanned:,} vectors compared" + (f", {search.ms:.1f} ms" if search else "")
    )
    add(
        _wrap(
            "Only the chosen clusters are read from disk. The rest of the corpus is "
            "never touched, which is what makes the effort level a real cost saving "
            "rather than a label."
        )
    )
    # The other half of the provenance story. Every timing on this page is this
    # machine's and says so; without this line a reader would reasonably conclude
    # the retrieval code has only ever run on a laptop, which is not true.
    if evidence is not None and evidence.any:
        add("")
        add(_wrap(f"This same retrieval code on real hardware: {evidence.headline()}"))
        add(
            _wrap(
                "Those are separate recorded runs, not this query. This query's "
                "milliseconds are this machine's."
            )
        )

    # ---------------------------------------------------------------- 6
    out += _step(6, "WHAT CAME BACK")
    for ch in t.chunks:
        where = ch.title or f"chunk #{ch.chunk_id}"
        add(f"        [{ch.rank}]  {ch.score:.3f}   {where}")
        add(
            f"             chunk #{ch.chunk_id}"
            + (f", document {ch.doc_id}" if ch.doc_id else "")
            + f", {ch.store}"
        )
    if math.isfinite(t.best_score):
        add("")
        add(f"        best score {t.best_score:.3f}")

    # ---------------------------------------------------------------- 6b
    if t.cache_hit:
        # Not `c`: that is the ControllerDecision bound in step 4.
        reuse = t.cache or {}
        out += _step(7, "THIS QUESTION WAS ALREADY ANSWERED, IN OTHER WORDS")
        add(f"        matched   {reuse.get('matched_question', '')!r}")
        add(
            f"        similarity {reuse.get('similarity', float('nan')):.3f} "
            f"(threshold {reuse.get('threshold', float('nan')):.3f})"
        )
        add(
            f"        avoided    {reuse.get('saved_ms', float('nan')):.0f} ms of work that the "
            f"first run paid for"
        )
        add(
            _wrap(
                "The answer below was not recomputed. The passages, the controller "
                "decision and the generation all come from the earlier run; only the "
                "device reading above belongs to this query."
            )
        )

    if t.guard is not None:
        out += _step(7 if not t.cache_hit else 8, "IS THE CORPUS ACTUALLY ABLE TO ANSWER THIS?")
        verdict = t.guard
        add(
            f"        best passage {verdict.get('best_score', float('nan')):.3f}   "
            f"threshold {verdict.get('threshold', float('nan')):.3f}   "
            f"margin {verdict.get('margin', float('nan')):+.3f}"
        )
        add(_wrap(str(verdict.get("reason", "")), "        -> ", first="        -> "))
        if t.refused:
            add("")
            add(
                _wrap(
                    "The generator was never run. Refusing on the retrieval scores costs "
                    "nothing; refusing after generating costs the most expensive stage and "
                    "then discards it."
                )
            )

    # ---------------------------------------------------------------- 7
    out += _step(9 if t.guard is not None else 7, "THE ANSWER IS WRITTEN FROM THOSE PASSAGES")
    gen = next((s for s in t.stages if s.name == "generate"), None)
    add(
        f"        {t.generator}"
        + (f", {gen.ms:.0f} ms, {t.prompt_tokens} prompt tokens" if gen else "")
    )
    if t.truncated:
        add("        WARNING: the prompt was truncated; some retrieved context")
        add("        never reached the model.")
    add("")
    add(_wrap(t.answer, "        "))
    add("")
    if t.answer_declined:
        add(
            _wrap(
                "The model declined. The passages retrieved do not contain the "
                "answer, and it said so rather than inventing one.",
                "        ",
            )
        )
    elif t.cited_chunk_ids:
        cited = ", ".join(f"#{i}" for i in t.cited_chunk_ids)
        add(f"        CITES chunk {cited}, word for word.")
    else:
        add(
            _wrap(
                "No passage contains this answer word for word, so the model "
                "rephrased or combined. Read the passages above before quoting it.",
                "        ",
            )
        )

    # ---------------------------------------------------------------- 8
    out += _step(8, "WHAT IT COST")
    for s in t.stages:
        share = s.ms / t.total_ms if t.total_ms > 0 else 0.0
        bar = "#" * max(0, min(24, round(share * 24)))
        add(f"        {s.name:<9} {s.ms:8.1f} ms  {bar:<24} {100 * share:4.1f}%")
    add(f"        {'TOTAL':<9} {t.total_ms:8.1f} ms")
    add("")
    e = t.energy
    if e.label == "estimate":
        add(f"        energy    {e.mj:.1f} mJ   ESTIMATE, not a measurement")
        add(_wrap(f"method: {e.method}", "            "))
        add(_wrap(f"coefficient: {e.coefficient_source}", "            "))
        add(_wrap(f"covers {e.covers}; excludes {e.excludes}", "            "))
    else:
        add(f"        energy    not measured -- {e.method}")

    # ---------------------------------------------------------------- 9
    out += _step(9, "WHAT LEFT THE DEVICE")
    add(
        f"        {_plural(t.network_bytes_sent, 'byte')} sent, "
        f"{_plural(t.network_connections, 'connection')} opened"
    )
    if t.sent_nothing:
        add(
            _wrap(
                "Nothing. The question, the corpus, the retrieval and the model all "
                "stayed on this device. Counted by watching this process's own "
                "sockets across the query, not asserted.",
                "        ",
            )
        )
    else:
        add(
            _wrap(
                "SOMETHING LEFT THE DEVICE. This query was not fully local, and the "
                "privacy claim does not hold for it.",
                "        ",
            )
        )

    add("")
    add("=" * WIDTH)
    return "\n".join(out)

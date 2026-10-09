"""14 -- the demonstration dashboard.

Four tabs: ask a question and watch how it was answered, drive the device into a
named condition and watch the controller react, compare the on-device path with
a cloud API, and read the benchmarks.

**Every number on this page is read from a file that a run produced.** Nothing
is typed into a template. Where a result has not been produced the panel says so
and names the script that would produce it, rather than showing a placeholder
that looks like a measurement. The provenance of each figure -- measured here,
replayed from the handset, computed, or estimated -- travels with it.

Run it with:

    streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"

st.set_page_config(
    page_title="SMART-ANN",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Colours are defined once and reused by every chart so the page reads as one
# system. They are chosen to hold up on both light and dark backgrounds, which
# Streamlit switches between without telling the app which is active.
#
# No plotly `template` is set anywhere on this page, deliberately. Pinning
# `plotly_white` puts dark axis text on a dark background the moment the viewer
# is in dark mode; leaving it unset lets Streamlit apply its own theme, which
# follows the active mode. Backgrounds are transparent for the same reason.
IDEAS = {
    "device": "#1f7a4d",
    "cloud": "#b3382c",
    "warn": "#c2410c",
    "cool": "#0369a1",
    "muted": "#64748b",
    "accent": "#7c3aed",
}

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px;}
      div[data-testid="stMetricValue"] {font-size: 1.5rem;}
      .badge {
        display:inline-block; padding:.28rem .7rem; border-radius:999px;
        font-size:.78rem; font-weight:600; margin:.15rem .35rem .15rem 0;
        border:1px solid currentColor;
      }
      .ok    {color:#1f7a4d;}
      .warn  {color:#c2410c;}
      .info  {color:#0369a1;}
      .muted {color:#64748b;}
      .step {
        border-left:3px solid #7c3aed; padding:.35rem 0 .35rem .8rem; margin:.35rem 0;
      }
      .prov {font-size:.78rem; opacity:.75; font-style:italic;}
      h1,h2,h3 {letter-spacing:-.01em;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Loading results
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict[str, Any] | None:
    p = RESULTS / name
    if not p.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
        return data
    except json.JSONDecodeError:
        return None


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame | None:
    p = RESULTS / name
    if not p.exists():
        return None
    return pd.read_csv(p)


def missing(what: str, script: str) -> None:
    """Say what is absent and how to produce it. Never draw a placeholder."""
    st.info(f"**{what} has not been produced yet.**  \nRun `{script}` and reload.")


def badge(text: str, kind: str = "ok") -> str:
    return f'<span class="badge {kind}">{text}</span>'


def provenance(text: str) -> None:
    st.markdown(f'<div class="prov">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# The pipeline, loaded once
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading index, embedder and generator ...")
def get_pipeline() -> Any:
    from smartann.pipeline.ask import RagPipeline

    table = RESULTS / "demo_profile_table.json"
    p = RagPipeline(
        dataset="demo_phone_care",
        k=5,
        profile_table_path=table if table.exists() else None,
        use_kb=False,
    )
    p.load()
    return p


@st.cache_resource(show_spinner=False)
def get_guard() -> Any:
    from smartann.pipeline.guard import GroundingGuard

    p = RESULTS / "grounding_threshold.json"
    if not p.exists():
        return None
    try:
        return GroundingGuard.load(p, dataset="demo_phone_care")
    except ValueError:
        return None


@st.cache_resource(show_spinner=False)
def get_cache() -> Any:
    from smartann.pipeline.cache import SemanticCache

    p = RESULTS / "cache_threshold.json"
    threshold = 0.95
    if p.exists():
        threshold = float(json.loads(p.read_text(encoding="utf-8"))["threshold"])
    return SemanticCache(threshold=threshold)


@st.cache_data(show_spinner=False)
def get_device_evidence() -> Any:
    """What has actually run on the handset. Cached: the scan reads six JSONL
    files and the answer does not change while the page is open."""
    from smartann.telemetry.provenance import device_evidence

    return device_evidence(RESULTS / "device", RESULTS / "device_profile_table.json")


@st.cache_data(show_spinner=False)
def get_stress_modes() -> dict[str, Any]:
    from smartann.telemetry.stress import load_modes

    return load_modes(RESULTS / "device")


PRESETS = [
    "Why does my phone get hot even when it is not charging?",
    "My phone heats up while playing games. Why?",
    "Is it dangerous if my phone battery is swollen?",
    "Why does my phone get hot when the signal is weak?",
    "What should I do when my phone is too hot right now?",
    "Who wrote the play Hamlet?",
]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### SMART-ANN")
    st.caption(
        "A resource-aware adaptive ANN controller for on-device RAG. "
        "Everything on this page runs on this machine."
    )

    proof = load_json("offline_proof.json")
    if proof:
        if proof.get("passed"):
            st.markdown(badge("Works offline — proved", "ok"), unsafe_allow_html=True)
            st.caption(f"Phase 12.5 proof passed {proof['ran_at']}")
        else:
            st.markdown(badge("Offline proof FAILED", "warn"), unsafe_allow_html=True)
    else:
        st.markdown(badge("Offline: not proved", "muted"), unsafe_allow_html=True)

    evidence = get_device_evidence()
    if evidence.any:
        st.markdown(badge("Measured on real hardware", "ok"), unsafe_allow_html=True)
        st.caption(evidence.headline())
        with st.expander("Which stages ran on the phone"):
            from smartann.telemetry.provenance import PIPELINE_STAGES

            for stage in PIPELINE_STAGES:
                mark = "✅" if stage.on_device else "💻"
                where = "handset" if stage.on_device else "host only"
                st.markdown(f"{mark} **{stage.name}** — {where}")
                st.caption(stage.evidence)
            st.caption(
                "Counts are read from results/device/*_queries.jsonl. Every timing "
                "shown elsewhere on this page is this machine's and says so."
            )
    else:
        st.markdown(badge("No handset runs on disk", "muted"), unsafe_allow_html=True)

    st.divider()
    st.markdown("**Device condition**")
    modes = get_stress_modes()
    mode_name = st.radio(
        "Stress mode",
        options=["(none — laptop sensors)", *modes],
        label_visibility="collapsed",
    )
    active_mode = modes.get(mode_name)
    if active_mode is not None:
        kind = "ok" if active_mode.is_measured else "warn"
        label = "MEASURED" if active_mode.is_measured else "HYPOTHETICAL"
        st.markdown(badge(label, kind), unsafe_allow_html=True)
        st.caption(active_mode.detail)

    st.divider()
    use_cache = st.toggle(
        "Semantic cache",
        value=True,
        help="Reuse answers to questions already asked in other words. Bypassed while a "
        "stress mode is pinned, so the controller's reaction is always this query's.",
    )
    use_guard = st.toggle(
        "Grounding guard",
        value=True,
        help="Refuse rather than guess when the corpus cannot answer.",
    )

    guard = get_guard()
    if use_guard and guard is not None:
        st.caption(f"Refuse below similarity {guard.threshold:.3f}")
        provenance(guard.calibrated_from)
    elif use_guard:
        st.caption("No calibrated threshold on disk; the guard stays off.")


tab_ask, tab_device, tab_cloud, tab_bench = st.tabs(
    ["Ask", "Device & Controller", "On-device vs Cloud", "Benchmarks"]
)


# ---------------------------------------------------------------------------
# Tab 1 -- Ask
# ---------------------------------------------------------------------------
with tab_ask:
    st.markdown("## Ask the knowledge base")
    st.caption(
        "71 authored documents about why a phone gets hot, indexed on this machine. "
        "The question, the corpus, the retrieval and the model all stay here."
    )

    if "question" not in st.session_state:
        st.session_state.question = PRESETS[0]

    cols = st.columns(len(PRESETS))
    for col, preset in zip(cols, PRESETS, strict=True):
        with col:
            if st.button(
                preset if len(preset) < 34 else preset[:31] + "…",
                width="stretch",
                help=preset,
            ):
                st.session_state.question = preset

    question = st.text_input("Your question", key="question")
    go_button = st.button("Answer it", type="primary")

    # Load the generator when the PAGE opens, not when the first question is
    # asked. It costs about 21 seconds, and paying that on the first question
    # means a live demonstration stands in silence in front of an audience.
    # Paid here it happens while the page is still being introduced, behind a
    # spinner that says what it is doing.
    #
    # Wrapped, because a missing index must break this tab and not the whole
    # page: the benchmark and comparison tabs read files and need no model.
    try:
        get_pipeline()
    except Exception as exc:
        st.error(
            f"The query pipeline could not load, so this tab cannot answer: {exc}\n\n"
            f"The other tabs read recorded results and still work."
        )
        st.stop()

    if go_button or st.session_state.get("answered_once"):
        st.session_state.answered_once = True
        from smartann.pipeline.ask import SCR_OFF
        from smartann.pipeline.querytrace import NetworkWitness, doc_slugs, from_ask_result

        pipeline = get_pipeline()
        device = active_mode.state if active_mode is not None else None
        source = active_mode.label if active_mode is not None else ""

        with st.spinner("retrieving and generating ..."):
            with NetworkWitness() as witness:
                result = pipeline.ask(
                    question,
                    nprobe=8,
                    scr=SCR_OFF,
                    device=device,
                    device_source=source,
                    device_is_measured=(
                        active_mode.is_measured if active_mode is not None else False
                    ),
                    guard=guard if use_guard else None,
                    # The cache is bypassed whenever a stress mode is pinned. A
                    # hit returns the earlier run's controller decision, and the
                    # whole point of a stress mode is to watch the controller
                    # react to THIS device state -- showing the new temperature
                    # beside the old decision would be the one misleading thing
                    # this page could do.
                    cache=get_cache() if (use_cache and active_mode is None) else None,
                )
            from smartann.config import resolve_paths

            slugs = doc_slugs(
                Path(resolve_paths().data_root) / "processed" / "demo_phone_care" / "chunks.sqlite"
            )
            trace = from_ask_result(
                result,
                network_bytes_sent=witness.bytes_sent,
                network_connections=witness.connections,
                doc_ids=slugs,
            )

        badges = []
        if trace.sent_nothing:
            badges.append(badge("0 bytes sent off-device", "ok"))
        else:
            badges.append(badge(f"{trace.network_bytes_sent} bytes LEFT the device", "warn"))
        if proof and proof.get("passed"):
            badges.append(badge("Works offline", "ok"))
        if trace.cache_hit:
            badges.append(badge("Cache hit", "info"))
        if trace.refused:
            badges.append(badge("Refused — not in the knowledge base", "warn"))
        if active_mode is not None and not active_mode.is_measured:
            badges.append(badge("Device state is hypothetical", "warn"))
        st.markdown(" ".join(badges), unsafe_allow_html=True)

        left, right = st.columns([3, 2])
        with left:
            st.markdown("### Answer")
            if trace.refused:
                st.warning(result.answer)
            else:
                st.success(result.answer)

            if trace.guard is not None:
                g = trace.guard
                st.markdown("**Confidence**")
                best, thr = g.get("best_score", float("nan")), g.get("threshold", float("nan"))
                st.progress(
                    max(0.0, min(1.0, float(best))),
                    text=f"best passage {best:.3f} against a refusal threshold of {thr:.3f}",
                )
                provenance(
                    "The threshold is the midpoint of a measured gap: over 20 labelled "
                    "queries, no answerable one scored below it and no unanswerable one above."
                )

            st.markdown("### Sources")
            for ch in trace.chunks:
                with st.expander(
                    f"[{ch.rank}]  {ch.score:.3f}  —  {ch.title or f'chunk #{ch.chunk_id}'}"
                ):
                    st.write(
                        next(
                            (c.reduced_text for c in result.chunks if c.chunk_id == ch.chunk_id),
                            "",
                        )
                    )
                    st.caption(
                        f"chunk #{ch.chunk_id} · document `{ch.doc_id or 'n/a'}` · {ch.store}"
                    )

        with right:
            st.markdown("### How this answer was made")
            steps = [
                ("Question", f"“{trace.query}”"),
                (
                    "Embedding",
                    next((s.detail for s in trace.stages if s.name == "embed"), "—"),
                ),
                (
                    "Device read",
                    f"{trace.resources.source} — {trace.resources.measured_on}",
                ),
                ("Controller", trace.controller.reason),
                (
                    "Search",
                    f"{trace.clusters_probed} clusters opened, "
                    f"{trace.vectors_scanned:,} vectors compared",
                ),
                ("Passages", f"top {len(trace.chunks)}, best {trace.best_score:.3f}"),
                (
                    "Generator",
                    "not run — refused before generating"
                    if trace.refused
                    else ("reused from cache" if trace.cache_hit else f"{trace.generator}"),
                ),
            ]
            if trace.cache_hit:
                st.caption(
                    "Reused from the cache: the controller, search and generator rows "
                    "below describe the earlier run. Only the device reading is this "
                    "query's."
                )
            for name, detail in steps:
                st.markdown(
                    f'<div class="step"><b>{name}</b><br>'
                    f'<span style="font-size:.86rem">{detail}</span></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("### What it cost")
            stage_df = pd.DataFrame([{"stage": s.name, "ms": s.ms} for s in trace.stages])
            fig = go.Figure(
                go.Bar(
                    x=stage_df["ms"],
                    y=stage_df["stage"],
                    orientation="h",
                    marker_color=IDEAS["accent"],
                    hovertemplate="%{y}: %{x:.1f} ms<extra></extra>",
                )
            )
            fig.update_layout(
                height=40 * len(stage_df) + 90,
                margin={"l": 0, "r": 0, "t": 10, "b": 10},
                xaxis_title="ms (measured on this machine)",
                yaxis={"autorange": "reversed"},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, width="stretch")

            c1, c2 = st.columns(2)
            c1.metric("Total", f"{trace.total_ms:.0f} ms")
            if trace.energy.label == "estimate":
                c2.metric("Energy (estimate)", f"{trace.energy.mj:.1f} mJ")
            else:
                c2.metric("Energy", "not measured")
            provenance(trace.energy.method)

        with st.expander("The full trace, as saved to traces/"):
            st.json(trace.as_dict())


# ---------------------------------------------------------------------------
# Tab 2 -- Device & Controller
# ---------------------------------------------------------------------------
with tab_device:
    st.markdown("## Device state and what the controller does with it")

    if active_mode is None:
        st.caption(
            "No stress mode selected. Pick one in the sidebar to pin the device to a "
            "named condition; the gauges below then show that state."
        )
        from smartann.telemetry import unavailable_state

        try:
            from smartann.telemetry.windows import WindowsMonitor

            state = WindowsMonitor().sample()
            state_label = "live sensors on this laptop"
            measured = True
        except Exception:
            state = unavailable_state()
            state_label = "nothing readable on this platform"
            measured = False
    else:
        state = active_mode.state
        state_label = active_mode.label
        measured = active_mode.is_measured

    st.markdown(
        badge("MEASURED" if measured else "HYPOTHETICAL", "ok" if measured else "warn")
        + f'<span class="prov"> &nbsp;{state_label}</span>',
        unsafe_allow_html=True,
    )

    def gauge(title: str, value: float, lo: float, hi: float, unit: str, colour: str) -> go.Figure:
        if not math.isfinite(value):
            fig = go.Figure()
            fig.add_annotation(
                text=f"<b>{title}</b><br>not measured<br><span style='font-size:.8em'>"
                f"absent, not zero</span>",
                showarrow=False,
                font={"size": 13, "color": IDEAS["muted"]},
            )
            fig.update_layout(
                height=190,
                margin={"l": 10, "r": 10, "t": 10, "b": 10},
                xaxis={"visible": False},
                yaxis={"visible": False},
                paper_bgcolor="rgba(0,0,0,0)",
            )
            return fig
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=value,
                number={"suffix": f" {unit}"},
                title={"text": title, "font": {"size": 14}},
                gauge={
                    "axis": {"range": [lo, hi]},
                    "bar": {"color": colour},
                    "bgcolor": "rgba(0,0,0,0)",
                },
            )
        )
        fig.update_layout(
            height=190,
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    g1, g2, g3, g4 = st.columns(4)
    g1.plotly_chart(
        gauge("Battery", state.battery_pct, 0, 100, "%", IDEAS["device"]), width="stretch"
    )
    g2.plotly_chart(
        gauge("Temperature", state.temperature_c, 20, 90, "°C", IDEAS["warn"]),
        width="stretch",
    )
    g3.plotly_chart(gauge("CPU", 100 * state.cpu_util, 0, 100, "%", IDEAS["cool"]), width="stretch")
    g4.plotly_chart(
        gauge("Free RAM", state.mem_available_mb, 0, 32768, "MB", IDEAS["muted"]),
        width="stretch",
    )

    st.caption(
        "A gauge that reads *not measured* is a platform that cannot report that sensor. "
        "It is not a zero, and every policy here declines to act on it."
    )

    st.divider()
    st.markdown("### Retrieval depth against temperature, over a recorded handset trace")
    stress = load_csv("phase12_fixed_vs_adaptive.csv")
    if stress is None:
        missing("The fixed-vs-adaptive run", "python scripts/phase12_stress_test.py")
    else:
        fig = go.Figure()
        colours = {
            "fixed": IDEAS["cloud"],
            "adaptive-hottest": IDEAS["device"],
            "adaptive-skin": IDEAS["cool"],
        }
        for arm, colour in colours.items():
            a = stress[stress["arm"] == arm]
            if a.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=a["t_trace_s"],
                    y=a["nprobe"],
                    name=arm,
                    line={"color": colour, "shape": "hv"},
                    hovertemplate=f"{arm}<br>t=%{{x:.0f}}s<br>nprobe=%{{y}}<extra></extra>",
                )
            )
        hot = stress[stress["arm"] == "adaptive-hottest"]
        if not hot.empty:
            fig.add_trace(
                go.Scatter(
                    x=hot["t_trace_s"],
                    y=hot["temperature_c"],
                    name="hottest zone (°C)",
                    yaxis="y2",
                    line={"color": IDEAS["warn"], "width": 1.3, "dash": "dot"},
                )
            )
        fig.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="time along the recorded trace (s)",
            yaxis_title="retrieval depth (nprobe)",
            yaxis2={"title": "temperature (°C)", "overlaying": "y", "side": "right"},
            legend={"orientation": "h", "y": 1.12},
            margin={"l": 0, "r": 0, "t": 40, "b": 0},
        )
        st.plotly_chart(fig, width="stretch")
        provenance(
            "Device state replayed from the measured C5 handset trace. The two adaptive "
            "arms differ only in which thermometer they read."
        )

        summary = load_json("phase12_fixed_vs_adaptive.json")
        if summary:
            rows = []
            for arm, s in summary["arms"].items():
                rows.append(
                    {
                        "arm": arm,
                        "median nprobe": s["nprobe_p50"],
                        "p50 ms": round(s["p50_ms"], 2),
                        "p95 ms": round(s["p95_ms"], 2),
                        "recall@10": round(s["recall_at_10"], 4),
                        "energy mJ/query (est)": round(s["energy_mj_per_query_estimate"], 2),
                        "over budget %": round(s["over_budget_pct"], 1),
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            provenance(
                "Latency measured on this laptop; recall exact against brute force; "
                "energy an estimate from the handset's measured mJ/cluster."
            )


# ---------------------------------------------------------------------------
# Tab 3 -- Cloud comparison
# ---------------------------------------------------------------------------
with tab_cloud:
    st.markdown("## On-device RAG against a cloud API")
    comp = load_json("comparison.json")
    comp_csv = load_csv("comparison.csv")
    if comp is None or comp_csv is None:
        missing("The comparison", "python scripts/phase13_comparison.py")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Bytes sent on-device",
            f"{comp['device_bytes_total']}",
            help="Measured: this process's own socket sends, across every query.",
        )
        c2.metric(
            "Bytes the cloud arm would send",
            f"{comp['cloud_payload_bytes']['total']:,.0f}",
            help="Computed from the exact request body; nothing was transmitted.",
        )
        c3.metric(
            "Cloud latency",
            "measured" if comp["cloud_latency_ms"] else "not measured",
            help="Measured only when OPENAI_API_KEY is set.",
        )

        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=["on-device"], y=[0], marker_color=IDEAS["device"], name="on-device")
        )
        fig.add_trace(
            go.Bar(
                x=["cloud API"],
                y=[comp["cloud_payload_bytes"]["mean"]],
                marker_color=IDEAS["cloud"],
                name="cloud API",
            )
        )
        fig.update_layout(
            height=320,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="user data leaving the device (bytes per query)",
            margin={"l": 0, "r": 0, "t": 30, "b": 0},
            title="Privacy: the question and the retrieved passages would both be sent",
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown("### The comparison, dimension by dimension")
        st.dataframe(comp_csv, width="stretch", hide_index=True)
        provenance(
            "Every cell carries its own provenance column. Where a number was not "
            "measured the cell says so rather than estimating."
        )


# ---------------------------------------------------------------------------
# Tab 4 -- Benchmarks
# ---------------------------------------------------------------------------
with tab_bench:
    st.markdown("## Benchmarks")

    st.markdown("### Why the per-cluster graphs were dropped")
    regime = load_json("regime_summary.json")
    if regime is None:
        missing("The index regime comparison", "scripts/experiment_matrix.py")
    else:
        fig = go.Figure()
        palette = [IDEAS["device"], IDEAS["cloud"], IDEAS["accent"]]
        for (name, series), colour in zip(regime["series"].items(), palette, strict=False):
            fig.add_trace(
                go.Scatter(
                    x=series["ms"],
                    y=series["recall"],
                    mode="markers+lines",
                    name=name,
                    marker={"size": 10, "color": colour},
                    line={"color": colour},
                    hovertemplate=f"{name}<br>%{{x:.1f}} ms<br>recall %{{y:.4f}}<extra></extra>",
                )
            )
        fig.update_layout(
            height=430,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="latency (ms)",
            yaxis_title="recall",
            legend={"orientation": "h", "y": 1.12},
            margin={"l": 0, "r": 0, "t": 40, "b": 0},
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"Measured under MobileRAG's own build condition: {regime['condition']}, "
            f"N/Nc = {regime['n_over_nc']:.0f}. At matched recall the flat mmap'd clusters "
            f"are faster at every point, which is why this project's index has no "
            f"per-cluster graphs."
        )
        provenance(f"source: {regime['source']}")

    st.divider()
    st.markdown("### Fixed depth against the controller")
    png = RESULTS / "phase12_fixed_vs_adaptive.png"
    if png.exists():
        st.image(str(png), width="stretch")
    else:
        missing("The fixed-vs-adaptive charts", "python scripts/phase12_stress_test.py")

    st.divider()
    st.markdown("### Retrieval quality on the demonstration corpus")
    ev = load_json("demo_eval.json")
    table = load_json("demo_profile_table.json")
    if ev is None or table is None:
        missing("The demo evaluation", "python scripts/phase11_demo_eval.py")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "recall@5 (paraphrases)", f"{ev['paraphrases']['recall_at_5_frozen_primary']:.3f}"
        )
        c2.metric("relevant@5", f"{ev['paraphrases']['relevant_at_5']:.3f}")
        c3.metric("distractors in top 5", ev["paraphrases"]["distractors_in_top5"])

        entries = pd.DataFrame(table["entries"])
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=entries["p95_ms"],
                y=entries["recall"],
                mode="markers+lines+text",
                text=[f"n={n}" for n in entries["nprobe"]],
                textposition="bottom right",
                marker={"size": 9, "color": IDEAS["device"]},
                line={"color": IDEAS["device"]},
                name="effort level",
            )
        )
        fig.update_layout(
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="measured p95 latency (ms), this laptop",
            yaxis_title="recall@10 against exact search",
            margin={"l": 0, "r": 0, "t": 30, "b": 0},
            title="The ladder the controller chooses from",
        )
        st.plotly_chart(fig, width="stretch")
        provenance(table["measured_under"])

    st.divider()
    st.markdown("### The controller's stability check")
    check = load_json("controller_check.json")
    if check is None:
        missing("The controller check", "python scripts/phase11_controller_check.py")
    else:
        rows = [
            {
                "control period (s)": a["control_period_s"],
                "median nprobe": a["nprobe_p50"],
                "at floor %": round(100 * a["floor_fraction"], 1),
                "swing (% of range)": round(100 * a["swing_fraction_of_range"], 0),
                "reversals/min": round(a["reversals_per_min"], 2),
                "p95 ms": round(a["p95_ms"], 1),
                "stable": "yes" if a["stable_amended"] else "no",
            }
            for a in check["arms"]
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(
            "Trace-replay on measured handset latency. The unpaced arm reproduces the "
            "ADR-0032 ratchet; pacing to the window's fill time removes it."
        )
        img = RESULTS / "controller_check.png"
        if img.exists():
            st.image(str(img), width="stretch")

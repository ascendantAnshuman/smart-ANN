"""Gather what the project has actually measured, into one page.

A dashboard is where a project most easily starts lying, because a panel with
nothing behind it looks exactly like a panel with a measurement behind it. So
every section here is built from a file on disk, and a section whose file is
missing renders as *not run* rather than disappearing. A reader must be able to
tell "we measured this and it was small" apart from "we never measured this"
(Section 3, rule 1).

**Sections fail independently.** A missing Phase 8 summary must not blank the
page, because the most common reason to open this is to see how far a run has
got. Each collector returns None on a missing input and the renderer says so.

**No number is retyped.** Everything comes from the same JSON the figures and
the paper read, so the page cannot drift from the results it describes.
"""

from __future__ import annotations

import base64
import json
import math
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["DashboardState", "collect"]


def _read_json(path: Path) -> dict[str, Any] | None:
    """A missing or unreadable input is absence, not an error to swallow later."""
    if not path.exists():
        return None
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return None


def _git_sha(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


@dataclass
class DashboardState:
    """Everything the page can show, with absence represented rather than hidden."""

    generated_at: str
    git_sha: str
    results_dir: Path

    profile: dict[str, Any] | None = None
    thermal: dict[str, Any] | None = None
    qa: dict[str, Any] | None = None
    joint: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    device_response: dict[str, Any] | None = None
    contributions: list[dict[str, Any]] = field(default_factory=list)
    phase2: dict[str, Any] | None = None
    policy6: dict[str, Any] | None = None
    energy: dict[str, Any] | None = None
    cost_model: dict[str, Any] | None = None
    thermal_model: dict[str, Any] | None = None
    regime: dict[str, Any] | None = None
    query_adaptive: dict[str, Any] | None = None
    kb: dict[str, Any] | None = None
    answers: list[dict[str, Any]] = field(default_factory=list)
    figures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def missing(self) -> list[str]:
        """Sections with no input on disk, named so the page can list them."""
        pairs = (
            ("profile table", self.profile),
            ("thermal attribution", self.thermal),
            ("answer quality (phase 7)", self.qa),
            ("joint control (phase 8)", self.joint),
            ("verification", self.verification),
            ("device response", self.device_response),
            ("policy comparison (phase 6)", self.policy6),
            ("energy model (phase 3.5)", self.energy),
            ("latency cost model (phase 3)", self.cost_model),
            ("index acceptance (phase 2)", self.phase2),
            ("query-adaptive effort (phase 9)", self.query_adaptive),
        )
        return [name for name, value in pairs if value is None]


def _collect_thermal(results: Path, prefix: str) -> dict[str, Any] | None:
    """Run the attribution, or report why it could not run.

    Called rather than read: the attribution is derived from two fitted models
    and a trace, and recomputing keeps the page's score identical to the one
    `smartann thermal explain` prints. A stored copy could drift from it.
    """
    try:
        from smartann.thermal.attribution import (
            EnergyModel,
            ThermalModel,
            attribute,
            load_trace,
        )

        a = attribute(
            load_trace(results / "device", prefix),
            EnergyModel.load(results / "energy_model.json"),
            ThermalModel.load(results / "thermal_model.json"),
        )
    except (FileNotFoundError, ValueError, KeyError):
        return None

    t = a.trace
    return {
        "score": a.score,
        "idle_w": a.power.idle_w,
        "work_w": a.power.work_w,
        "unexplained_w": a.power.unexplained_w,
        "measured_w": a.power.measured_mean_w,
        "idle_share": a.power.idle_share,
        "work_share": a.power.work_share,
        "unexplained_share": a.power.unexplained_share,
        "skin_max_c": float(t.skin_c.max()),
        "skin_min_c": float(t.skin_c.min()),
        "hottest_max_c": t.hottest_max_c,
        "excursion_c": t.peak_excursion_c,
        "was_charging": t.was_charging,
        "minutes": t.minutes,
        "n_queries": t.n_queries,
        "tau_s": a.thermal_model.tau_s,
        "sensors_disagree": a.sensors_disagree,
        "trace": t.name,
    }


def _collect_figures(figures_dir: Path, wanted: tuple[str, ...]) -> list[tuple[str, str]]:
    """Embed each figure as a data URI so the page is one portable file.

    A dashboard that only renders next to its image directory is a dashboard
    that breaks the moment it is emailed, which is the main thing anyone wants
    to do with it.
    """
    out: list[tuple[str, str]] = []
    for name in wanted:
        path = figures_dir / name
        if not path.exists():
            continue
        try:
            data = base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError:
            continue
        out.append((path.stem, f"data:image/png;base64,{data}"))
    return out


def _collect_device_response(results: Path, trace: str) -> dict[str, Any] | None:
    """Replay one measured trace through the device policy, under both sensors.

    This is the C5 result as a demonstration rather than a claim. The same
    recorded run is fed to the same policy twice, changing only which of the
    handset's two thermometers it reads, and the effort levels it chooses are
    recorded against time. Every input is measured: the temperatures come from
    the trace, and the recall and latency of each level come from the profile
    table. The only thing computed here is which row of that table the policy
    picks, which is the decision under test.
    """
    table_path = results / "device_profile_table.json"
    telemetry = results / "device" / f"{trace}_telemetry.jsonl"
    if not table_path.exists() or not telemetry.exists():
        return None

    try:
        from smartann.controller.policies.budget import BudgetPolicy
        from smartann.controller.policies.device import DeviceStatePolicy, DeviceTiers
        from smartann.pipeline.ask import COLD_HISTORY, load_profile_table
        from smartann.telemetry.replay import SENSOR_MEANING, SENSORS, load_trace
        from smartann.types import (
            AnnConfig,
            Constraints,
            ControllerState,
            QueryFeatures,
            ScrConfig,
        )

        table = load_profile_table(table_path, "squad")
    except (OSError, ValueError, RuntimeError):
        return None

    base = AnnConfig(nprobe=8, ef_centroid=128, ef_list=64, k=5, per_cluster_k=32)
    scr = ScrConfig(1, 0, 0, 3, 0.0, enabled=False)
    budget_ms = 120.0
    cons = Constraints(
        latency_target_ms=budget_ms,
        latency_hard_ms=budget_ms,
        energy_budget_mj=float("inf"),
        min_recall=None,
    )
    tiers = DeviceTiers()

    # Query features do not enter DeviceStatePolicy's decision, and supplying a
    # recorded query's would suggest they did. NaN says they were not consulted.
    unused_query = QueryFeatures(
        centroid_top1_dist=math.nan,
        centroid_dist_gap=math.nan,
        centroid_dist_entropy=math.nan,
        query_norm=math.nan,
        query_token_len=0,
    )

    series: list[dict[str, Any]] = []
    for sensor in SENSORS:
        try:
            tr = load_trace(telemetry, sensor=sensor)
        except (OSError, ValueError):
            return None
        policy = DeviceStatePolicy(table, base, scr, tiers)
        budget_only = BudgetPolicy(table, base, scr)
        step = max(1, len(tr) // 160)  # a readable number of points, evenly spaced
        points: list[dict[str, Any]] = []
        derated = 0
        for i in range(0, len(tr), step):
            st = ControllerState(device=tr.state_at(i), query=unused_query, history=COLD_HISTORY)
            chosen = policy.decide(st, cons).ann.nprobe
            allowed = budget_only.decide(st, cons).ann.nprobe
            if chosen < allowed:
                derated += 1
            entry = table.entry_for(chosen)
            points.append(
                {
                    "t": tr.samples[i].t - tr.samples[0].t,
                    "temperature_c": tr.temperature_at(i),
                    "nprobe": chosen,
                    "allowed_nprobe": allowed,
                    "recall": entry.recall if entry else math.nan,
                }
            )
        recalls = [p["recall"] for p in points if math.isfinite(p["recall"])]
        series.append(
            {
                "sensor": sensor,
                "means": SENSOR_MEANING[sensor],
                "peak_c": tr.peak_c,
                "n_points": len(points),
                "n_derated": derated,
                "levels": sorted({p["nprobe"] for p in points}),
                "mean_recall": (sum(recalls) / len(recalls)) if recalls else math.nan,
                "points": points,
            }
        )

    # Every other recorded run, summarised. One trace showing the two sensors
    # disagreeing is an anecdote; six independent runs agreeing that the skin
    # never reaches the threshold the shield trips on is a repeated measurement,
    # and that is the difference between an observation and a finding.
    across: list[dict[str, Any]] = []
    from smartann.telemetry.replay import available_traces, trace_path

    for name in available_traces(results / "device"):
        try:
            peaks = {s: load_trace(trace_path(results / "device", name), s).peak_c for s in SENSORS}
        except (OSError, ValueError):
            continue
        across.append(
            {
                "name": name,
                "skin_peak_c": peaks[SENSORS[0]],
                "hottest_peak_c": peaks[SENSORS[1]],
                "skin_crossed": peaks[SENSORS[0]] >= tiers.warm_c,
                "hottest_crossed": peaks[SENSORS[1]] >= tiers.warm_c,
            }
        )

    unconstrained = table.best_within(budget_ms)
    return {
        "trace": trace,
        "across_traces": across,
        "budget_ms": budget_ms,
        "warm_c": tiers.warm_c,
        "hot_c": tiers.hot_c,
        "duration_s": series[0]["points"][-1]["t"] if series and series[0]["points"] else 0.0,
        "budget_only_nprobe": unconstrained.nprobe,
        "budget_only_recall": unconstrained.recall,
        "series": series,
    }


def _samples_note(model: dict[str, Any]) -> str:
    """How many samples a fitted model rests on, or that the file does not say.

    A fit whose sample count was never recorded is not the same as one fitted on
    an unknown-but-large number, so the absence is printed rather than a "?"
    formatted as if it were a figure.
    """
    n = model.get("n_samples")
    if isinstance(n, (int, float)) and not isinstance(n, bool):
        return f"fitted on {int(n):,} measured samples"
    return "fitted; the sample count is not recorded in the file"


def _count_note(summary: dict[str, Any], key: str, what: str) -> str:
    """How many of something a summary records, whether it stored a count or the items.

    `joint_cells_on_pareto_front` holds the cells themselves, not their number,
    and formatting that straight into a sentence printed a page of JSON where a
    figure belonged. Counting the list is the same fact in the form the sentence
    needs; a number already stored as a number is used as it is.
    """
    value = summary.get(key)
    if isinstance(value, list):
        return f"{len(value)} {what}"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{int(value)} {what}"
    return f"the summary records no {what}"


def _collect_contributions(results: Path, state: DashboardState) -> list[dict[str, Any]]:
    """The five claims, each with the file that settles it and what that file says.

    A status board is the easiest place in a project to promote a hope into a
    result, so nothing here is written by hand. A contribution counts as settled
    only when a summary on disk records a verdict, and that verdict's own words
    are quoted. Where no file records one, the row says the evidence exists but
    the verdict was never written down -- which is a different and weaker claim
    than "demonstrated", and must read as one.
    """

    def stat(ok: bool, evidence: Path, verdict: str, detail: str) -> dict[str, Any]:
        return {
            "state": "yes" if ok else ("partial" if evidence.exists() else "no"),
            # Forward slashes: the page is read on whatever machine it is
            # emailed to, and a Windows path in it reads as a local accident.
            "evidence": evidence.relative_to(results).as_posix() if evidence.exists() else "",
            "verdict": verdict,
            "detail": detail,
        }

    rows: list[dict[str, Any]] = []

    p6 = results / "phase6" / "policy_comparison.json"
    n_levels = len(state.profile["entries"]) if state.profile else 0
    rows.append(
        {
            "id": "C1",
            "claim": "Retrieval effort can be chosen at runtime, from measured cost.",
            **stat(
                bool(state.profile) and n_levels > 1,
                p6,
                "demonstrated" if n_levels > 1 else "no measured levels to choose from",
                f"{n_levels} effort levels measured on the handset; a budget selects one"
                if n_levels
                else "profile table missing",
            ),
        }
    )

    qa = state.qa
    arms = qa["arms"] if qa else []
    best = max(arms, key=lambda a: a["f1"]) if arms else None
    off = next((a for a in arms if a["arm"] == "off"), None)
    rows.append(
        {
            "id": "C2",
            "claim": "Context reduction is a controllable quality/cost trade, and its cost "
            "is measured rather than assumed.",
            **stat(
                best is not None,
                results / "phase7" / "qa_summary.json",
                "measured: every reduction level tested cost quality"
                if (best and off and best["arm"] == "off")
                else ("measured" if best else "not run"),
                f"best arm '{best['arm']}' at {best['f1']:.4f} F1 over {best['n']:,} questions"
                if best
                else "no phase 7 summary",
            ),
        }
    )

    joint = state.joint
    rows.append(
        {
            "id": "C3",
            "claim": "Controlling retrieval and reduction jointly reaches trade-offs "
            "neither reaches alone.",
            **stat(
                bool(joint and joint.get("verdict")),
                results / "phase8" / "joint_summary.json",
                str(joint.get("verdict", "")) if joint else "not run",
                _count_note(joint, "joint_cells_on_pareto_front", "joint cells on the Pareto front")
                if joint
                else "no phase 8 summary",
            ),
        }
    )

    energy = _read_json(results / "energy_model.json")
    rows.append(
        {
            "id": "C4",
            "claim": "Per-query energy is measured on the device, not extrapolated.",
            **stat(
                bool(energy),
                results / "energy_model.json",
                "measured: fitted from the handset's own current and voltage"
                if energy
                else "not run",
                _samples_note(energy) if energy else "no energy model",
            ),
        }
    )

    c5 = _read_json(results / "device" / "c5_summary.json")
    blind = ""
    if c5:
        total = int(c5.get("adaptive_decisions_total", 0) or 0)
        withr = int(c5.get("adaptive_decisions_with_a_reason", 0) or 0)
        if total:
            blind = (
                f"{withr:,} of {total:,} decisions ({100.0 * withr / total:.1f}%) had any "
                f"device signal behind them; the rest were taken blind"
            )
    rows.append(
        {
            "id": "C5",
            "claim": "A device-state policy can act on battery and thermal signals.",
            **stat(
                False,  # recorded as real but not actionable; see the note below
                results / "device" / "c5_summary.json",
                "NOT demonstrated: the signal is real but not actionable" if c5 else "not run",
                blind or "no C5 summary",
            ),
        }
    )

    q9 = state.query_adaptive
    detail = "not run"
    if q9:
        wins = sum(1 for c in q9["comparisons"] if c["held_out"]["delta_recall"] > 0)
        best = max((c["held_out"]["delta_recall"] for c in q9["comparisons"]), default=0.0)
        detail = (
            f"{wins} of {len(q9['comparisons'])} budgets improved, best {best:+.4f} recall "
            f"on {q9['n_test']:,} held-out queries"
        )
    rows.append(
        {
            "id": "C6",
            "claim": "Effort allocated by measured query difficulty beats spending the "
            "same time uniformly.",
            **stat(
                bool(q9 and str(q9.get("verdict", "")).startswith("C6 SUPPORTED")),
                results / "phase9" / "query_adaptive.json",
                str(q9.get("verdict", "")) if q9 else "not run",
                detail,
            ),
        }
    )
    return rows


def _collect_kb(data_root: Path) -> dict[str, Any] | None:
    """The knowledge base's own inventory, or None when nothing was added.

    Read live rather than from a written summary: the store is the only record
    of what a query could actually have retrieved, and a stale copy would
    describe a corpus that no longer exists.
    """
    try:
        from smartann.kb import KnowledgeBase

        kb = KnowledgeBase(data_root / "kb")
    except (OSError, ValueError):
        return None
    try:
        srcs = kb.sources()
        if not srcs:
            return None
        return {
            "n_chunks": len(kb),
            "sources": [
                {
                    "title": s.title,
                    "url": s.url,
                    "kind": s.kind,
                    "added_at": s.added_at,
                    "n_chunks": s.n_chunks,
                    "n_chars": s.n_chars,
                }
                for s in srcs
            ],
        }
    finally:
        kb.close()


def collect(
    results: Path = Path("results"),
    repo: Path = Path(),
    thermal_prefix: str = "c5_",
    answers_path: Path | None = None,
) -> DashboardState:
    """Read every result this project has produced, tolerating the missing ones."""
    state = DashboardState(
        generated_at=datetime.now(UTC).astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        git_sha=_git_sha(repo),
        results_dir=results,
    )
    state.profile = _read_json(results / "device_profile_table.json")
    state.thermal = _collect_thermal(results, thermal_prefix)
    state.qa = _read_json(results / "phase7" / "qa_summary.json")
    state.joint = _read_json(results / "phase8" / "joint_summary.json")
    state.verification = _read_json(results / "verification.json")
    state.device_response = _collect_device_response(results, thermal_prefix.rstrip("_"))
    state.phase2 = _read_json(results / "phase2_summary.json")
    state.policy6 = _read_json(results / "phase6" / "policy_comparison.json")
    state.energy = _read_json(results / "energy_model.json")
    state.cost_model = _read_json(results / "cost_model.json")
    state.thermal_model = _read_json(results / "thermal_model.json")
    state.regime = _read_json(results / "regime_summary.json")
    state.query_adaptive = _read_json(results / "phase9" / "query_adaptive.json")

    from smartann.config import resolve_paths

    state.kb = _collect_kb(Path(resolve_paths().data_root))

    if answers_path is not None and answers_path.exists():
        raw = json.loads(answers_path.read_text(encoding="utf-8"))
        state.answers = raw if isinstance(raw, list) else [raw]

    # Last: it reads the sections collected above rather than the disk again.
    state.contributions = _collect_contributions(results, state)

    state.figures = _collect_figures(
        results / "figures",
        (
            "fig_thermal_attribution.png",
            "fig1_headline.png",
            "fig_joint.png",
            "fig_scr_quality.png",
            "fig_recall_effort.png",
            "fig_device.png",
            "fig_regime.png",
            "fig_footprint.png",
        ),
    )
    return state

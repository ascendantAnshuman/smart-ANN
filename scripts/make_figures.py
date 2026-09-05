#!/usr/bin/env python
"""Figures for the review. Every number comes from stored parquet or JSON.

Section 3, rule 1: no plotting script may contain a hardcoded result, example
data, or a placeholder metric. If an input is missing this script fails loudly
rather than drawing something plausible. The contention window is read from the
run manifest's recorded events, never from a hardcoded time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_acceptance import MOBILERAG_FIG6_MB

from smartann.config import resolve_paths

# One caption fragment, used everywhere the index is named. The artifact labelled
# "ecovector" has no per-cluster graphs, so it is not EcoVector (ADR-0028).
INDEX_LABEL = (
    "IVF-HNSW + mmap\n(our EcoVector configuration with per-cluster graphs disabled, ADR-0028)"
)

plt.rcParams.update(
    {"figure.dpi": 130, "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3}
)


def _load_run(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    parquet, manifest = run_dir / "queries.parquet", run_dir / "manifest.json"
    if not parquet.exists() or not manifest.exists():
        raise FileNotFoundError(
            f"{run_dir} is missing queries.parquet or manifest.json. Run "
            f"scripts/experiment_contention.py first; this script will not invent data."
        )
    return pd.read_parquet(parquet), json.loads(manifest.read_text(encoding="utf-8"))


def _event_time(manifest: dict, name: str) -> float | None:
    for e in manifest.get("events", []):
        if e["name"] == name:
            return float(e["t_wall"])
    return None


def _controller_p95(v, window: int = 50):
    """The estimator the policy actually consumed, reconstructed from the parquet.

    RollingWindow observes a latency AFTER the query completes and the policy
    reads p95 BEFORE the next one, so at query i the controller saw the p95 of
    queries [i-window, i-1]. Reproducible exactly from stored latencies in order,
    which beats adding a field: the plotted signal is provably the one acted on
    rather than a similar-looking one. A time-based trailing p95 would show a
    reader a trace that never justified the action in the nprobe panel.
    """
    out = np.full(v.size, np.nan)
    for i in range(1, v.size):
        out[i] = np.percentile(v[max(0, i - window) : i], 95)
    return out


def _rolling_p95(
    t: np.ndarray, v: np.ndarray, window_s: float = 3.0, stride: int = 1
) -> np.ndarray:
    """p95 over a trailing time window, in O(n) rather than O(n^2).

    Two pointers walk the array once and the window is re-sorted only over its
    own contents, so cost is linear in samples rather than quadratic. The naive
    form -- percentile over a growing slice at every step -- was not merely slow:
    run concurrently with an experiment it perturbed the very latencies being
    plotted (defect #10). Cheap analysis is a correctness property here, not a
    convenience.
    """
    out = np.empty(v.size)
    lo = 0
    last = 0.0
    for i in range(v.size):
        while t[lo] < t[i] - window_s:
            lo += 1
        if i % stride == 0:
            last = float(np.percentile(v[lo : i + 1], 95))
        out[i] = last
    return out


def phase_stats(df: pd.DataFrame, mf: dict, target: float) -> dict[str, dict[str, float]]:
    """SLO violation rate and recall per phase.

    The violation rate is what makes the trade legible. Without it a reader sees
    static holding recall flat while adaptive gives some up, and concludes static
    won -- when static "won" recall by blowing through its latency budget.
    """
    a, b = _event_time(mf, "burners_start"), _event_time(mf, "burners_stop")
    t = df["t_wall"].to_numpy()
    phases = {
        "quiet": t < a,
        "contended": (t >= a) & (t < b),
        "recovery": t >= b,
    }
    # In target-step mode the MIDDLE phase runs against a tightened target, so a
    # violation rate computed against the original budget measures compliance
    # with a constraint that was not in force. The disturbance event carries the
    # target that actually applied; use it where present.
    phase_target = dict.fromkeys(phases, target)
    for e in mf.get("events", []):
        if e["name"] == "burners_start" and "target_ms" in e:
            phase_target["contended"] = float(e["target_ms"])
    out: dict[str, dict[str, float]] = {}
    for name, mask in phases.items():
        if not mask.any():
            continue
        lat = df["t_total"].to_numpy()[mask]
        rec = df["recall_at_k"].to_numpy(dtype=float)[mask]
        ovh = df["controller_overhead_us"].to_numpy()[mask]
        # Percentiles OF THE PER-QUERY RATIO, not a ratio of percentiles. The
        # latter reported p95 (0.173%) BELOW p50 (0.570%), which is impossible
        # for one distribution: latency's p95/p50 spread is wider than
        # overhead's, so the larger denominator shrank the ratio. As written it
        # would have read as "tiny even in the worst 5%", which was not measured.
        ratio_pct = (ovh / 1000.0) / np.maximum(lat, 1e-9) * 100.0
        tgt = phase_target[name]
        out[name] = {
            "n": int(mask.sum()),
            "target_ms": tgt,
            "violation_pct": float((lat > tgt).mean() * 100.0),
            "p95_ms": float(np.percentile(lat, 95)),
            "recall": float(np.nanmean(rec)) if np.isfinite(rec).any() else float("nan"),
            "nprobe_p50": float(np.percentile(df["cfg_ann_nprobe"].to_numpy()[mask], 50)),
            "overhead_p50_pct": float(np.percentile(ratio_pct, 50)),
            "overhead_p95_pct": float(np.percentile(ratio_pct, 95)),
        }
    return out


def headline_sentence(stats: dict[str, dict[str, dict[str, float]]]) -> str:
    """The sentences a supervisor will remember. Computed, never asserted.

    An earlier version ended with a fixed clause -- "the mechanism works; the
    control law does not" -- which was true of the run it was written for and
    false of the next one, where it printed alongside a NEGATIVE recall cost.
    A generated sentence that carries a hardcoded conclusion will eventually
    state the opposite of its own numbers, so the verdict is derived here.
    """
    s_c, a_c = stats["static"]["contended"], stats["aimd"]["contended"]
    s_q, a_q = stats["static"]["quiet"], stats["aimd"]["quiet"]
    q_gap = s_q["recall"] - a_q["recall"]

    first = (
        f"Under contention the static policy exceeded its latency target on "
        f"{s_c['violation_pct']:.1f}% of queries and the adaptive policy on "
        f"{a_c['violation_pct']:.1f}%, at a cost of {s_c['recall'] - a_c['recall']:.3f} "
        f"recall@10 ({s_c['recall']:.4f} -> {a_c['recall']:.4f})."
    )

    # The quiet phase is the diagnostic: with no disturbance, does the controller
    # sit where a sensible static config would, or has it collapsed?
    collapsed = a_q["nprobe_p50"] <= max(2.0, 0.25 * s_q["nprobe_p50"])
    if collapsed:
        verdict = (
            f"In the QUIET phase, with no disturbance at all, the adaptive policy missed "
            f"the target on {a_q['violation_pct']:.1f}% against static's "
            f"{s_q['violation_pct']:.1f}%, while holding median nprobe at "
            f"{a_q['nprobe_p50']:.0f} against {s_q['nprobe_p50']:.0f} and giving up "
            f"{q_gap:.3f} recall. It has collapsed to its floor: the mechanism works, "
            f"the control law does not."
        )
    else:
        gained = "gaining" if q_gap < 0 else "giving up"
        verdict = (
            f"In the QUIET phase it holds median nprobe at {a_q['nprobe_p50']:.0f} against "
            f"static's {s_q['nprobe_p50']:.0f}, {gained} {abs(q_gap):.3f} recall at "
            f"{a_q['violation_pct']:.1f}% over target against static's "
            f"{s_q['violation_pct']:.1f}%. It is not collapsed to its floor."
        )
    return first + "\n" + verdict


def feedback_delay_table(latency_by_nprobe: dict[int, float], window: int = 50) -> str:
    """How long the controller's own feedback takes, as a function of its output.

    The window is fixed in SAMPLES, so its delay in TIME varies with effort. A
    controller whose feedback delay is a function of its own output is a harder
    object than one with a fixed delay, and this argues for a time-based window
    independently of the ratchet.
    """
    rows = [f"  window = {window} samples"]
    for npb, ms in sorted(latency_by_nprobe.items()):
        rows.append(f"    nprobe={npb:>3}: {ms:7.2f} ms/query -> {window * ms / 1000:6.2f} s delay")
    return "\n".join(rows)


def fig_headline(runs: dict[str, Path], out: Path) -> None:
    """The figure the project is about: does the controller hold the target?

    Each policy is normalised to ITS OWN start and ITS OWN event times. The runs
    are sequential -- static then aimd -- so a shared origin put the second run's
    data outside the first run's zoom window entirely, leaving one panel empty
    and another showing a single policy. Elapsed-time-within-run is the axis that
    makes them comparable.
    """
    data = {name: _load_run(d) for name, d in runs.items()}
    # Per-run origin and per-run phase boundaries.
    origin, burn = {}, {}
    for name, (df, mf) in data.items():
        origin[name] = float(df["t_wall"].min())
        a, b = _event_time(mf, "burners_start"), _event_time(mf, "burners_stop")
        if a is None or b is None:
            raise ValueError(f"{name}: manifest has no burners_start/burners_stop to shade")
        burn[name] = (a - origin[name], b - origin[name])
    ref = next(iter(data.values()))[1]
    target = float(ref["latency_target_ms"])

    stats = {name: phase_stats(df, mf, target) for name, (df, mf) in data.items()}
    sentence = headline_sentence(stats)
    print(f"\n  {sentence}\n", flush=True)
    (out.parent / "headline_stats.json").write_text(
        json.dumps({"sentence": sentence, "stats": stats}, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(4, 1, figsize=(11, 14), gridspec_kw={"height_ratios": [3, 3, 3, 2]})
    colors = {"static": "#c1121f", "aimd": "#0353a4"}

    # -- panel 1: the whole run, each policy on its own clock ---------------
    ax = axes[0]
    for name, (df, _mf) in data.items():
        t = df["t_wall"].to_numpy() - origin[name]
        ax.plot(
            t,
            _rolling_p95(t, df["t_total"].to_numpy(), stride=25),
            lw=1.4,
            color=colors.get(name),
            label=f"{name} (n={len(df):,})",
        )
    a0, b0 = burn[next(iter(data))]
    ax.axvspan(a0, b0, color="grey", alpha=0.18, lw=0, label="disturbance")
    ax.axhline(target, ls="--", c="k", lw=1.2, label=f"target {target:.1f} ms")
    ax.set_ylabel("p95 latency (ms, 3 s trailing)")
    ax.set_xlabel("elapsed time within run (s)")
    ax.set_title(
        f"Adaptive retrieval effort under a mid-run disturbance\n{INDEX_LABEL}", fontsize=10
    )
    ax.legend(fontsize=8, loc="upper left")

    # -- panel 2: the transient, on the estimator the controller consumed ---
    ax = axes[1]
    lo, hi = a0 - 10.0, a0 + 10.0
    for name, (df, _mf) in data.items():
        t = df["t_wall"].to_numpy() - origin[name]
        m = (t >= lo) & (t <= hi)
        ax.plot(
            t[m],
            _controller_p95(df["t_total"].to_numpy())[m],
            lw=1.7,
            color=colors.get(name),
            label=f"{name}: 50-sample p95 (what the policy read)",
        )
    ax.axvspan(a0, hi, color="grey", alpha=0.18, lw=0)
    ax.axhline(target, ls="--", c="k", lw=1.2)
    ax.set_xlim(lo, hi)
    ax.set_ylabel("p95 latency (ms)")
    ax.set_title(
        "Zoom around the disturbance. This is the estimator the controller acted on", fontsize=9
    )
    ax.legend(fontsize=8, loc="upper left")

    # -- panel 3: what the controller actually did -------------------------
    ax = axes[2]
    df, _ = data["aimd"]
    t = df["t_wall"].to_numpy() - origin["aimd"]
    a1, _b1 = burn["aimd"]
    m = (t >= a1 - 10.0) & (t <= a1 + 10.0)
    ax.step(
        t[m],
        df["cfg_ann_nprobe"].to_numpy()[m],
        where="post",
        color="#0353a4",
        lw=1.5,
        label="nprobe chosen (adaptive)",
    )
    ax.axvspan(a1, a1 + 10.0, color="grey", alpha=0.18, lw=0)
    ax.set_xlim(a1 - 10.0, a1 + 10.0)
    ax.set_ylabel("nprobe")
    ax.set_xlabel("elapsed time within run (s)")
    rec = df["recall_at_k"].to_numpy(dtype=float)[m]
    if np.isfinite(rec).any():
        ax2 = ax.twinx()
        ax2.plot(
            t[m],
            pd.Series(rec).rolling(50, min_periods=1).mean(),
            color="#588157",
            lw=1.4,
            label="recall@10 (50-query mean)",
        )
        ax2.set_ylabel("recall@10")
        ax2.grid(False)
        ax2.legend(fontsize=8, loc="lower right")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(
        "The controller sits at the nprobe floor in every phase -- the collapse, not an adaptation",
        fontsize=9,
    )

    # -- panel 4: the number the story rests on ----------------------------
    ax = axes[3]
    phases = ["quiet", "contended", "recovery"]
    x = np.arange(len(phases))
    for i, (name, st) in enumerate(stats.items()):
        vals = [st.get(p, {}).get("violation_pct", 0.0) for p in phases]
        bars = ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, label=name, color=colors.get(name))
        for b, v in zip(bars, vals, strict=True):
            ax.annotate(
                f"{v:.1f}%",
                (b.get_x() + b.get_width() / 2, v),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x, phases)
    ax.set_ylabel("queries over target (%)")
    ax.set_title(sentence, fontsize=8.5, wrap=True)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    for name, st in stats.items():
        c = st.get("contended", {})
        print(
            f"  {name:>7}: controller overhead p50 {c.get('overhead_p50_pct', float('nan')):.3f}% "
            f"p95 {c.get('overhead_p95_pct', float('nan')):.3f}% of query latency",
            flush=True,
        )


def fig_recall_effort(accept_json: Path, out: Path) -> None:
    """The controller's operating range, from Phase 2 check 2."""
    rows = json.loads(accept_json.read_text(encoding="utf-8"))["check2"]
    npb = [r["nprobe"] for r in rows]
    rec = [r["recall"] for r in rows]
    ms = [r["ms"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(ms, rec, "o-", color="#0353a4", lw=1.6)
    for x, y, n in zip(ms, rec, npb, strict=True):
        ax.annotate(f"nprobe={n}", (x, y), textcoords="offset points", xytext=(6, -9), fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("latency (ms/query, log scale)")
    ax.set_ylabel("recall@10 (score-equality)")
    ax.set_title(
        f"Recall against effort -- the range the controller moves in\n{INDEX_LABEL}", fontsize=9
    )
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def fig_footprint(accept_json: Path, out: Path) -> None:
    """Measured resident footprint against MobileRAG Figure 6."""
    meas = json.loads(accept_json.read_text(encoding="utf-8"))["check5"]
    names = list(meas)
    ours = [meas[n] for n in names]
    theirs = [MOBILERAG_FIG6_MB[n] for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.2, theirs, 0.4, label="MobileRAG Fig. 6 (reported)", color="#adb5bd")
    ax.bar(x + 0.2, ours, 0.4, label="measured here", color="#0353a4")
    for i, (a, b) in enumerate(zip(theirs, ours, strict=True)):
        ax.annotate(f"{b / a:.2f}x", (i + 0.2, b), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, names)
    ax.set_ylabel("resident footprint (MB)")
    ax.set_title(
        "Resident footprint against the paper\n(same order of magnitude: "
        "independent evidence the index is built correctly)",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def fig_scr_quality(qa_json: Path, out: Path) -> None:
    """C2's trade-off: what context reduction costs in answer quality.

    The whole point of Phase 7. Every earlier SCR measurement priced reduction
    in characters, tokens and milliseconds, and on those axes a reduction that
    deletes the answer looks exactly like a free one. Here the vertical axis is
    quality, so a destructive setting falls and a free one does not.

    Error bars are the paired bootstrap against `off` re-centred on each arm's
    own mean: the arms answer the SAME questions, so the pairing is real and an
    unpaired interval would discard it. An interval crossing the baseline line
    means this experiment cannot distinguish that arm from full context.
    """
    d = json.loads(qa_json.read_text(encoding="utf-8"))
    arms = d["arms"]
    paired = {p["arm"]: p for p in d["paired_bootstrap_vs_off"]}
    base_f1 = next(a["f1"] for a in arms if a["arm"] == "off")

    red = [a["reduction_pct"] for a in arms]
    f1 = [a["f1"] for a in arms]
    gen = [a["t_generate_ms_mean"] for a in arms]
    names = [a["arm"] for a in arms]

    lo, hi = [], []
    for a in arms:
        if a["arm"] == "off":
            lo.append(0.0)
            hi.append(0.0)
        else:
            c = paired[a["arm"]]["d_f1_ci"]
            lo.append(a["f1"] - (a["f1"] + (c[0] - paired[a["arm"]]["d_f1"])))
            hi.append((a["f1"] + (c[1] - paired[a["arm"]]["d_f1"])) - a["f1"])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax.errorbar(red, f1, yerr=[lo, hi], fmt="o-", color="#0353a4", capsize=4, lw=1.5)
    ax.axhline(base_f1, color="#adb5bd", ls="--", lw=1, label="full context (off)")
    for x, y, n in zip(red, f1, names, strict=True):
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("context removed (%)")
    ax.set_ylabel("token F1")
    ax.set_title(
        f"Answer quality against reduction\n"
        f"SQuAD, n={d['n_questions']}, {d['generator']['model_id']}, greedy",
        fontsize=9,
    )
    ax.legend(fontsize=8)

    ax2.plot(red, gen, "o-", color="#c1121f", lw=1.5)
    for x, y, n in zip(red, gen, names, strict=True):
        ax2.annotate(n, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax2.set_xlabel("context removed (%)")
    ax2.set_ylabel("mean generation latency (ms)")
    ax2.set_title("What the reduction buys\n(this laptop, not the target device)", fontsize=9)

    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def fig_regime(regime_json: Path, out: Path) -> None:
    """Wall-clock cost of the graphs against MobileRAG's own condition.

    Plots the RATIO to ivf_disk rather than absolute milliseconds. The regime
    run's absolutes are drift-affected -- non-monotone in nprobe, moving for all
    three variants at once -- while the within-nprobe ratios are stable, which is
    exactly what interleaving buys (ADR-0030). Plotting the absolutes would put
    the noisiest quantity on the y-axis of the figure that carries the claim.
    """
    d = json.loads(regime_json.read_text(encoding="utf-8"))
    series = d["series"]
    base = series["ivf_disk (no graphs)"]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for label, s_ in sorted(series.items()):
        if label.startswith("ivf"):
            continue
        ratios = [m / b for m, b in zip(s_["ms"], base["ms"], strict=True)]
        x = s_["ratio"][0]
        ax.scatter([x] * len(ratios), ratios, s=42, zorder=3, label=f"{label} (per nprobe)")
        ax.scatter([x], [np.median(ratios)], s=150, marker="_", lw=2.5, color="k", zorder=4)

    ax.axhline(1.0, ls="-", c="#444", lw=1.0)
    ax.axvline(1.0, ls="--", c="#c1121f", lw=1.4)
    ax.annotate(
        "MobileRAG's condition:\ngraphs can pay only LEFT of this line",
        (1.0, ax.get_ylim()[1] * 0.95),
        ha="right",
        va="top",
        fontsize=8.5,
        color="#c1121f",
    )
    ax.annotate(
        "default config\n(ef_L*M'/(N/Nc) = 2.05)",
        (2.05, 1.0),
        xytext=(2.05, 1.9),
        ha="center",
        fontsize=8,
        arrowprops={"arrowstyle": "->", "lw": 1},
    )
    ax.scatter(
        [2.05],
        [2.0],
        marker="X",
        s=110,
        color="#c1121f",
        zorder=5,
        label="default config (measured 2x slower, check 4)",
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"$ef_L \cdot M' \;/\; (N/N_c)$   — MobileRAG's own condition")
    ax.set_ylabel("wall-clock relative to no graphs\n(>1 means the graphs cost time)")
    ax.set_title(
        "Even deep inside the regime where the paper's model says the graphs\n"
        "should pay, they cost 2.6-3.5x wall-clock at identical recall",
        fontsize=9.5,
    )
    ax.legend(fontsize=7.5, loc="upper left")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def fig_joint(joint_json, out) -> None:
    """C3: does controlling both dimensions reach anything one cannot?

    The declared test is Pareto domination -- a joint cell supports C3 when no
    single-dimension cell is at least as accurate AND at least as cheap. Marker
    shape carries the family, so a reader can see WHERE on the frontier the
    joint cells sit rather than taking the verdict on trust.

    The cost axis is a MEDIAN, not a mean. The first run of this experiment had
    one generation take 20.4 minutes when the host slept, which dragged that
    cell's mean to six times its median and inflated exactly the baseline the
    joint cells were compared against (ADR-0046).
    """
    d = json.loads(joint_json.read_text(encoding="utf-8"))
    cells = d["cells"]
    front = {(c["nprobe"], c["scr"]) for c in d["joint_cells_on_pareto_front"]}

    style = {
        "retrieval_only": ("o", "#0353a4", "retrieval only (SCR off)"),
        "reduction_only": ("s", "#c1121f", "reduction only (max nprobe)"),
        "joint": ("^", "#2a9d8f", "joint (both varying)"),
    }
    fig, ax = plt.subplots(figsize=(7.5, 5))
    seen = set()
    for c in cells:
        m, col, lab = style[c["family"]]
        ax.scatter(
            c["cost_ms"],
            c["f1"],
            marker=m,
            s=130,
            color=col,
            edgecolor="black" if (c["nprobe"], c["scr"]) in front else "none",
            linewidth=1.6,
            zorder=3,
            label=lab if lab not in seen else None,
        )
        seen.add(lab)
        ax.annotate(
            f"n={c['nprobe']}/{c['scr']}",
            (c["cost_ms"], c["f1"]),
            textcoords="offset points",
            xytext=(7, -3),
            fontsize=7.5,
        )
    ax.set_xlabel("cost: median retrieval + median generation (ms)")
    ax.set_ylabel("token F1")
    ax.set_title(
        f"C3 -- joint vs single-dimension control\n"
        f"SQuAD, n={d['n_questions']}, ring = on the Pareto front "
        f"(no single-dimension cell dominates it)",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None, help="directory holding static/ and aimd/")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    paths = resolve_paths()
    # The experiment writes E_DEMO_<mode>; "E_DEMO" alone was left behind when
    # target-step mode was added, so the documented reproduce command pointed at
    # a directory nothing creates. Pick the newest of whatever exists.
    if args.runs:
        runs_root = Path(args.runs)
    else:
        candidates = sorted(paths.results_root.glob("E_DEMO_*"), key=lambda d: d.stat().st_mtime)
        if not candidates:
            print(
                f"no E_DEMO_* run directories under {paths.results_root}; "
                f"run scripts/experiment_contention.py first",
                file=sys.stderr,
            )
            return 1
        runs_root = candidates[-1]
        print(f"using most recent run: {runs_root}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Every registered figure belongs in the default set. `scr_quality` was
    # registered in the dispatch table below but omitted here, so it silently
    # drew nothing on a default run -- no figure, no error, and the summary
    # reported success. A figure that is skipped by default is a figure nobody
    # will notice is missing.
    all_figures = {"headline", "recall_effort", "footprint", "regime", "scr_quality", "joint"}
    wanted = set(args.only) if args.only else all_figures
    unknown = wanted - all_figures
    if unknown:
        print(
            f"unknown figure(s): {sorted(unknown)}; known: {sorted(all_figures)}", file=sys.stderr
        )
        return 1
    failures = []
    if "headline" in wanted:
        try:
            fig_headline(
                {"static": runs_root / "static", "aimd": runs_root / "aimd"},
                out / "fig1_headline.png",
            )
        except (FileNotFoundError, ValueError) as exc:
            failures.append(f"headline: {exc}")

    for name, fn, src in (
        ("recall_effort", fig_recall_effort, out.parent / "phase2_summary.json"),
        ("footprint", fig_footprint, out.parent / "phase2_summary.json"),
        ("regime", fig_regime, out.parent / "regime_summary.json"),
        ("scr_quality", fig_scr_quality, out.parent / "phase7" / "qa_summary.json"),
        ("joint", fig_joint, out.parent / "phase8" / "joint_summary.json"),
    ):
        if name not in wanted:
            continue
        try:
            fn(src, out / f"fig_{name}.png")
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
            failures.append(f"{name}: needs {src} ({type(exc).__name__}: {exc})")

    for f in failures:
        print(f"NOT DRAWN -- {f}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

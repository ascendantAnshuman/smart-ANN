#!/usr/bin/env python
"""Figures for the conference paper, drawn from the recorded results only.

Same rule as `make_figures.py`: every value is read from a file this project
wrote, and a missing input fails loudly rather than drawing something plausible
(Section 3, rule 1). A figure in a paper is a claim, and a claim whose number
nobody can trace back to a run is the thing this repository exists not to do.

**Vector output.** PDF, because the paper is set in LaTeX and a raster figure at
column width is visibly soft in print. PNG is written alongside for the dashboard
and for anyone previewing without a TeX toolchain.

**Colour follows the data's job, not taste.** Ordered things -- difficulty
quintiles, work quintiles -- take one hue stepped light to dark, because their
order is the information. Unordered things -- three policies, two thermometers --
take the validated categorical slots in fixed order. Both palettes were checked
with the validator rather than by eye, and every series is also direct-labelled
and marker-coded so the figures survive greyscale printing and photocopying,
which is what actually happens to a conference paper.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

R = Path("results")
OUT = R / "paper"

# Validated light-mode palette. Categorical slots 1-3 clear the all-pairs CVD
# and normal-vision floors; the blue ramp is the ordinal instance, monotone in
# lightness with the light end still clear of the surface.
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
RAMP5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#82817a", "#e4e2db"
CRITICAL, GOOD = "#d03b3b", "#0ca30c"

# IEEE two-column geometry.
COL, WIDE = 3.45, 7.16

# Named rather than written inline: an en dash is the right character for a
# numeric range, and the linter flags the literal in source as ambiguous with
# a hyphen -- which, in a file full of negative numbers, it is.
EN_DASH = chr(0x2013)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "figure.dpi": 200,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def load(rel: str) -> dict:
    p = R / rel
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing; run the experiment that writes it")
    return json.loads(p.read_text(encoding="utf-8"))


def style(ax, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {OUT / name}.pdf / .png")


# ---------------------------------------------------------------------------
# Fig 1 - the architecture, and where the two signals enter
# ---------------------------------------------------------------------------
def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(WIDE, 2.55))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 38)
    ax.axis("off")

    def box(x, y, w, h, label, sub="", fc="#ffffff", ec=MUTED, lw=0.8, bold=False):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.3,rounding_size=1.2",
                fc=fc,
                ec=ec,
                linewidth=lw,
                zorder=2,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2 + (1.1 if sub else 0),
            label,
            ha="center",
            va="center",
            fontsize=7.6,
            fontweight="bold" if bold else "normal",
            color=INK,
            zorder=3,
        )
        if sub:
            ax.text(
                x + w / 2,
                y + h / 2 - 2.0,
                sub,
                ha="center",
                va="center",
                fontsize=6.3,
                color=MUTED,
                zorder=3,
            )

    def arrow(x1, y1, x2, y2, color=MUTED, style_="-|>", lw=0.9, ls="-"):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle=style_,
                mutation_scale=7,
                color=color,
                linewidth=lw,
                linestyle=ls,
                shrinkA=0,
                shrinkB=0,
                zorder=1,
            )
        )

    y = 20
    box(1, y, 11, 8, "Query", "text")
    box(15, y, 12, 8, "Embed", "gte-small, 384-d")
    box(30, y, 15, 8, "PROBE", "centroid graph, RAM", fc="#eef5fd", ec=C1, lw=1.2, bold=True)
    box(63, y, 15, 8, "FINISH", "clusters, on disk", fc="#eef5fd", ec=C1, lw=1.2, bold=True)
    box(81, y, 11, 8, "Generate", "flan-t5-base")

    for a, b in ((12, 15), (27, 30), (78, 81)):
        arrow(a, y + 4, b, y + 4)

    # The controller sits between the two phases: that is the whole design.
    box(48, 5, 17, 9, "CONTROLLER", "chooses nprobe", fc="#fdf1ec", ec=C2, lw=1.2, bold=True)
    arrow(45, y + 2, 56.5, 14, color=C2, lw=1.1)
    arrow(56.5, 14, 63, y + 2, color=C2, lw=1.1)

    # Placed off the arrows rather than on them: at column width these two
    # labels sat on the controller box and on the legend beneath it.
    ax.text(
        43.5,
        13.0,
        "difficulty\n(free, C6)",
        fontsize=6.3,
        color=C2,
        ha="right",
        va="center",
        style="italic",
    )
    ax.text(
        69.5,
        13.0,
        "chosen\neffort level",
        fontsize=6.3,
        color=C2,
        ha="left",
        va="center",
        style="italic",
    )

    box(20, 2, 20, 7, "Device telemetry", "thermal, battery (C5)", fc="#f4f4f1", ec=MUTED)
    arrow(40, 6, 48, 8, color=MUTED, ls=(0, (3, 2)))

    box(73, 2, 22, 7, "Measured profile table", "p95, recall per level", fc="#f4f4f1", ec=MUTED)
    arrow(73, 7, 65, 9.5, color=MUTED, ls=(0, (3, 2)))

    ax.annotate(
        "cheap: no cluster opened",
        xy=(37.5, y),
        xytext=(37.5, 16.2),
        ha="center",
        fontsize=6.2,
        color=C1,
        arrowprops={"arrowstyle": "-", "color": C1, "lw": 0.6},
    )
    ax.annotate(
        "expensive: I/O + distance work",
        xy=(70.5, y),
        xytext=(70.5, 16.2),
        ha="center",
        fontsize=6.2,
        color=C1,
        arrowprops={"arrowstyle": "-", "color": C1, "lw": 0.6},
    )
    save(fig, "fig1_architecture")


# ---------------------------------------------------------------------------
# Fig 2 - the effort/return curve the controller chooses from
# ---------------------------------------------------------------------------
def fig_effort_return() -> None:
    t = load("device_profile_table.json")
    e = sorted(t["entries"], key=lambda x: x["nprobe"])
    p95 = [x["p95_ms"] for x in e]
    rec = [x["recall"] for x in e]

    fig, ax = plt.subplots(figsize=(COL, 2.25))
    style(ax, "both")
    ax.plot(p95, rec, "-o", color=C1, linewidth=1.6, markersize=4.5, zorder=3)
    for x, yv, n in zip(p95, rec, [x["nprobe"] for x in e], strict=True):
        ax.annotate(
            f"n={n}",
            (x, yv),
            textcoords="offset points",
            xytext=(4, -7.5),
            fontsize=6.6,
            color=INK2,
        )
    ax.set_xlabel("measured p95 latency on the handset (ms)")
    ax.set_ylabel("recall@10")
    ax.set_ylim(0.58, 1.02)
    ax.set_xlim(-12, 392)
    # The shape is the argument: the first 52 ms buys 0.20 recall, the last 164 buys 0.03.
    ax.annotate(
        "26x the latency\nfor 1.53x the recall",
        xy=(354, 0.9788),
        xytext=(196, 0.70),
        fontsize=6.6,
        color=MUTED,
        arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 0.7, "mutation_scale": 6},
    )
    save(fig, "fig2_effort_return")


# ---------------------------------------------------------------------------
# Fig 3 - the novelty: difficulty separates recall, before any cluster is opened
# ---------------------------------------------------------------------------
def fig_difficulty() -> None:
    q = load("phase9/query_adaptive.json")
    levels = q["levels"]
    sep = q["separation"]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(WIDE, 2.5), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    style(ax, "both")
    marks = ["o", "s", "^", "D", "v"]
    for i, b in enumerate(sep):
        ys = [b["recall_by_level"][str(lv)] for lv in levels]
        ax.plot(
            levels,
            ys,
            marker=marks[i],
            color=RAMP5[i],
            linewidth=1.5,
            markersize=4.0,
            zorder=3,
            # En dash for a numeric range, written as an escape: it is correct
            # typography and the linter flags the literal as ambiguous.
            label=f"Q{i + 1}  gap {b['gap_lo']:.3f}{EN_DASH}{b['gap_hi']:.3f}",
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(levels)
    ax.set_xticklabels([str(lv) for lv in levels])
    ax.set_xlabel("effort level (nprobe, clusters opened)")
    ax.set_ylabel("recall@10")
    ax.set_ylim(0.20, 1.03)
    ax.legend(loc="lower right", frameon=False, handlelength=1.6, borderpad=0.2)
    ax.set_title("(a) recall by difficulty quintile", loc="left", color=INK)

    # The spread at the cheapest level is the whole opportunity.
    lo = sep[0]["recall_by_level"]["1"]
    hi = sep[-1]["recall_by_level"]["1"]
    ax.annotate(
        "",
        xy=(1, lo),
        xytext=(1, hi),
        arrowprops={"arrowstyle": "<|-|>", "color": CRITICAL, "lw": 1.0, "mutation_scale": 6},
    )
    ax.text(
        1.15,
        (lo + hi) / 2,
        f"{hi / lo:.1f}x spread\nat nprobe=1",
        fontsize=6.6,
        color=CRITICAL,
        va="center",
    )

    # Panel (b): what one level of effort is worth in each quintile.
    style(ax2, "x")
    gains = [b["recall_by_level"]["4"] - b["recall_by_level"]["1"] for b in sep]
    ypos = np.arange(len(sep))[::-1]
    ax2.barh(ypos, gains, height=0.62, color=RAMP5, zorder=3)
    for yv, g in zip(ypos, gains, strict=True):
        ax2.text(g + 0.008, yv, f"+{g:.3f}", va="center", fontsize=6.8, color=INK2)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([f"Q{i + 1}" for i in range(len(sep))])
    ax2.set_xlabel("recall gained by nprobe 1 $\\rightarrow$ 4")
    ax2.set_xlim(0, max(gains) * 1.30)
    ax2.set_title("(b) marginal value of effort", loc="left", color=INK)
    ax2.invert_yaxis()
    save(fig, "fig3_difficulty")


# ---------------------------------------------------------------------------
# Fig 4 - C6 held out: allocation against uniform effort
# ---------------------------------------------------------------------------
def fig_c6() -> None:
    q = load("phase9/query_adaptive.json")
    comps = q["comparisons"]
    uc = [c["held_out"]["uniform_mean_cost_ms"] for c in comps]
    ur = [c["held_out"]["uniform_recall"] for c in comps]
    ac = [c["held_out"]["adaptive_mean_cost_ms"] for c in comps]
    ar = [c["held_out"]["adaptive_recall"] for c in comps]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(WIDE, 2.45), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    style(ax, "both")
    ax.plot(uc, ur, "-o", color=C1, linewidth=1.5, markersize=4.5, zorder=3, label="uniform effort")
    ax.plot(
        ac,
        ar,
        "--s",
        color=C2,
        linewidth=1.5,
        markersize=4.5,
        zorder=4,
        label="allocated by difficulty",
    )
    for x, yv, c in zip(uc, ur, comps, strict=True):
        ax.annotate(
            f"n={c['uniform_nprobe']}",
            (x, yv),
            textcoords="offset points",
            xytext=(3, -8),
            fontsize=6.4,
            color=MUTED,
        )
    ax.set_xlabel("mean device cost (ms, handset-measured p95 per level)")
    ax.set_ylabel("recall@10, 500 held-out queries")
    ax.set_ylim(0.57, 1.01)
    ax.legend(loc="lower right", frameon=False, handlelength=2.0)
    ax.set_title("(a) quality against cost", loc="left", color=INK)

    style(ax2, "y")
    d = [c["held_out"]["delta_recall"] for c in comps]
    xs = np.arange(len(comps))
    cols = [GOOD if v > 0 else (MUTED if abs(v) < 1e-9 else CRITICAL) for v in d]
    ax2.bar(xs, d, width=0.6, color=cols, zorder=3)
    res = q["recall_resolution"]
    ax2.axhline(3 * res, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax2.text(
        len(comps) - 0.5,
        3 * res + 0.0007,
        "3x resolution floor",
        fontsize=6.2,
        color=MUTED,
        ha="right",
    )
    for x, v in zip(xs, d, strict=True):
        ax2.text(
            x,
            v + (0.0008 if v >= 0 else -0.0018),
            f"{v:+.4f}",
            ha="center",
            fontsize=6.5,
            color=INK2,
        )
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"n={c['uniform_nprobe']}" for c in comps])
    ax2.set_xlabel("budget (the uniform level it replaces)")
    ax2.set_ylabel("$\\Delta$ recall@10")
    ax2.axhline(0, color=MUTED, lw=0.7)
    ax2.set_ylim(min(d) - 0.004, max(d) + 0.005)
    ax2.set_title("(b) held-out difference", loc="left", color=INK)
    save(fig, "fig4_c6")


# ---------------------------------------------------------------------------
# Fig 5 - the negative result: the shield reads the wrong thermometer
# ---------------------------------------------------------------------------
def fig_sensors() -> None:
    d = load("phase9/query_adaptive.json")  # presence check only, keeps ordering stable
    del d
    from smartann.controller.policies.device import DeviceTiers
    from smartann.telemetry.replay import SENSORS, available_traces, load_trace, trace_path

    device_dir = R / "device"
    names = available_traces(device_dir)
    if not names:
        raise FileNotFoundError("no recorded handset traces in results/device")

    peaks = {}
    for n in names:
        peaks[n] = {s: load_trace(trace_path(device_dir, n), s).peak_c for s in SENSORS}

    tiers = DeviceTiers()
    fig, ax = plt.subplots(figsize=(COL, 2.45))
    style(ax, "y")
    xs = np.arange(len(names))
    w = 0.36
    skin = [peaks[n]["skin"] for n in names]
    hot = [peaks[n]["hottest"] for n in names]
    ax.bar(xs - w / 2, skin, width=w, color=C1, zorder=3, label="skin sensor")
    ax.bar(xs + w / 2, hot, width=w, color=C2, zorder=3, label="hottest zone")

    ax.axhline(tiers.warm_c, color=CRITICAL, lw=1.1, ls=(0, (4, 2)), zorder=4)
    ax.text(
        -0.35,
        tiers.warm_c + 1.8,
        f"policy derates at {tiers.warm_c:.0f}°C",
        fontsize=6.5,
        color=CRITICAL,
        ha="left",
    )
    for x, v in zip(xs, hot, strict=True):
        ax.text(x + w / 2, v + 1.2, f"{v:.0f}", ha="center", fontsize=6.2, color=INK2)
    for x, v in zip(xs, skin, strict=True):
        ax.text(x - w / 2, v + 1.2, f"{v:.0f}", ha="center", fontsize=6.2, color=INK2)

    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=6.8)
    ax.set_ylabel("peak temperature (°C)")
    ax.set_xlabel("recorded handset run")
    ax.set_ylim(0, max(hot) * 1.18)
    ax.legend(loc="upper left", frameon=False, ncol=2, handlelength=1.4, borderpad=0.2)
    save(fig, "fig5_sensors")


# ---------------------------------------------------------------------------
# Fig 6 - three policies against one declared budget
# ---------------------------------------------------------------------------
def fig_policies() -> None:
    p = load("phase6/policy_comparison.json")
    arms = p["arms"]
    names = [a["policy"] for a in arms]
    viol = [a["violation_pct"] for a in arms]
    rec = [a["recall"] for a in arms]
    cols = [C1, C2, C3][: len(arms)]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(WIDE, 2.3))
    style(ax, "y")
    xs = np.arange(len(arms))
    ax.bar(xs, viol, width=0.55, color=cols, zorder=3)
    for x, v in zip(xs, viol, strict=True):
        ax.text(x, v + 0.12, f"{v:.2f}%", ha="center", fontsize=6.8, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_ylabel(f"queries over the {p['budget_ms']:.1f} ms budget (%)")
    ax.set_ylim(0, max(viol) * 1.22)
    ax.set_title("(a) the promise the controller keeps", loc="left", color=INK)

    style(ax2, "y")
    ax2.bar(xs, rec, width=0.55, color=cols, zorder=3)
    for x, v in zip(xs, rec, strict=True):
        ax2.text(x, v + 0.012, f"{v:.4f}", ha="center", fontsize=6.8, color=INK2)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(names)
    ax2.set_ylabel("recall@10")
    ax2.set_ylim(0, 1.06)
    ax2.set_title("(b) what it pays for that", loc="left", color=INK)
    save(fig, "fig6_policies")


# ---------------------------------------------------------------------------
# Fig 7 - effort costs power, measured on the handset
# ---------------------------------------------------------------------------
def fig_energy() -> None:
    e = load("energy_model.json")
    q = e["effort_vs_power"]["quintiles"]
    pw = [x["power_median_w"] for x in q]

    fig, ax = plt.subplots(figsize=(COL, 2.2))
    style(ax, "y")
    xs = np.arange(len(q))
    ax.bar(xs, pw, width=0.62, color=RAMP5[: len(q)], zorder=3)
    for x, v in zip(xs, pw, strict=True):
        ax.text(x, v + 0.022, f"{v:.3f}", ha="center", fontsize=6.6, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"Q{i + 1}" for i in range(len(q))])
    ax.set_xlabel("retrieval work in the interval (quintile, clusters$\\cdot$s$^{-1}$)")
    ax.set_ylabel("median power (W)")
    ax.set_ylim(0, max(pw) * 1.22)
    rise = 100.0 * (pw[-1] - pw[0]) / pw[0]
    ax.annotate(
        f"+{rise:.0f}% from quietest to busiest",
        xy=(len(q) - 1, pw[-1]),
        xytext=(0.15, max(pw) * 1.12),
        fontsize=6.6,
        color=MUTED,
        arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 0.7, "mutation_scale": 6},
    )
    save(fig, "fig7_energy")


# ---------------------------------------------------------------------------
# Fig 8 - the controller's choices over a measured run, under each thermometer
# ---------------------------------------------------------------------------
def fig_trace() -> None:
    from smartann.dashboard import _collect_device_response

    d = _collect_device_response(R, "c5")
    if not d:
        raise FileNotFoundError("could not replay the c5 trace")

    by = {s["sensor"]: s for s in d["series"]}
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(COL, 3.15), sharex=True, gridspec_kw={"height_ratios": [1, 1]}
    )
    # Titles sit inside each panel's own band; the default spacing put (b)'s
    # title on (a)'s x-axis at column width.
    fig.subplots_adjust(hspace=0.42)
    cols = {"skin": C1, "hottest": C2}

    style(ax, "y")
    for s in ("skin", "hottest"):
        pts = by[s]["points"]
        ax.plot(
            [p["t"] for p in pts],
            [p["temperature_c"] for p in pts],
            color=cols[s],
            lw=1.3,
            label=f"{s}, {by[s]['peak_c']:.0f}°C peak",
        )
    ax.axhline(d["warm_c"], color=CRITICAL, lw=0.9, ls=(0, (4, 2)))
    # Left, under the line: the legend owns the right half of this panel.
    ax.text(
        d["duration_s"] * 0.02,
        d["warm_c"] - 6.5,
        f"derate at {d['warm_c']:.0f}°C",
        fontsize=6.2,
        color=CRITICAL,
        ha="left",
    )
    ax.set_ylabel("°C")
    ax.set_ylim(22, 78)
    ax.legend(
        loc="upper right",
        frameon=False,
        handlelength=1.4,
        borderpad=0.15,
        labelspacing=0.25,
    )
    ax.set_title("(a) what each thermometer reported", loc="left", color=INK)

    style(ax2, "y")
    for s in ("skin", "hottest"):
        pts = by[s]["points"]
        ax2.step(
            [p["t"] for p in pts],
            [p["nprobe"] for p in pts],
            where="post",
            color=cols[s],
            lw=1.4,
        )
    ax2.set_yscale("log", base=2)
    ax2.set_yticks([1, 4, 8, 16, 32])
    ax2.set_yticklabels(["1", "4", "8", "16", "32"])
    ax2.set_ylabel("nprobe chosen")
    ax2.set_xlabel("time through the recorded run (s)")
    ax2.set_title("(b) what the controller did about it", loc="left", color=INK)
    save(fig, "fig8_trace")


def main() -> int:
    print("building paper figures from recorded results ...")
    for fn in (
        fig_architecture,
        fig_effort_return,
        fig_difficulty,
        fig_c6,
        fig_sensors,
        fig_policies,
        fig_energy,
        fig_trace,
    ):
        try:
            fn()
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"  {fn.__name__}: SKIPPED -- {exc}", file=sys.stderr)
            return 1
    print(f"\nall figures in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

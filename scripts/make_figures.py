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


def _rolling_p95(t: np.ndarray, v: np.ndarray, window_s: float = 3.0) -> np.ndarray:
    """p95 over a trailing time window, so the x-axis stays wall-clock."""
    out = np.empty(v.size)
    for i in range(v.size):
        lo = np.searchsorted(t, t[i] - window_s, side="left")
        out[i] = np.percentile(v[lo : i + 1], 95)
    return out


def fig_headline(runs: dict[str, Path], out: Path) -> None:
    """The figure the project is about: does the controller hold the target?"""
    data = {name: _load_run(d) for name, d in runs.items()}
    ref = next(iter(data.values()))[1]
    t0 = min(df["t_wall"].min() for df, _ in data.values())
    burn_a = _event_time(ref, "burners_start")
    burn_b = _event_time(ref, "burners_stop")
    if burn_a is None or burn_b is None:
        raise ValueError("manifest has no burners_start/burners_stop events to shade")
    target = float(ref["latency_target_ms"])

    fig, axes = plt.subplots(3, 1, figsize=(11, 11))
    colors = {"static": "#c1121f", "aimd": "#0353a4"}

    # -- panel 1: the whole run -------------------------------------------
    ax = axes[0]
    for name, (df, _mf) in data.items():
        t = df["t_wall"].to_numpy() - t0
        ax.plot(
            t,
            _rolling_p95(t, df["t_total"].to_numpy()),
            lw=1.4,
            color=colors.get(name),
            label=f"{name} (n={len(df):,})",
        )
    ax.axhline(target, ls="--", c="k", lw=1.2, label=f"target {target:.1f} ms")
    for a in axes[:2]:
        a.axvspan(burn_a - t0, burn_b - t0, color="grey", alpha=0.18, lw=0)
    ax.set_ylabel("p95 latency (ms, 3 s trailing)")
    ax.set_title(f"Adaptive retrieval effort under CPU contention\n{INDEX_LABEL}", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")

    # -- panel 2: the transient, which is invisible at full scale ----------
    ax = axes[1]
    lo, hi = burn_a - t0 - 10.0, burn_a - t0 + 10.0
    for name, (df, _mf) in data.items():
        t = df["t_wall"].to_numpy() - t0
        p95 = _rolling_p95(t, df["t_total"].to_numpy())
        m = (t >= lo) & (t <= hi)
        ax.plot(t[m], p95[m], lw=1.6, color=colors.get(name), label=name)
    ax.axhline(target, ls="--", c="k", lw=1.2)
    ax.set_xlim(lo, hi)
    ax.set_ylabel("p95 latency (ms)")
    ax.set_title("Zoom: +/- 10 s around contention onset -- the control action itself", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")

    # -- panel 3: what the controller actually did -------------------------
    ax = axes[2]
    df, _ = data["aimd"]
    t = df["t_wall"].to_numpy() - t0
    m = (t >= lo) & (t <= hi)
    ax.step(
        t[m],
        df["cfg_ann_nprobe"].to_numpy()[m],
        where="post",
        color="#0353a4",
        lw=1.6,
        label="nprobe (adaptive)",
    )
    ax.set_ylabel("nprobe")
    ax.set_xlabel("elapsed time (s)")
    ax.set_xlim(lo, hi)
    ax.axvspan(burn_a - t0, min(hi, burn_b - t0), color="grey", alpha=0.18, lw=0)

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
        "The trade the controller makes: effort down, recall down, target held", fontsize=9
    )

    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


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


def fig_regime(regime_json: Path, out: Path) -> None:
    """Recall and wall-clock against the paper's own condition, ef_L*M' / (N/Nc)."""
    d = json.loads(regime_json.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for label, series in d["series"].items():
        ax.plot(series["ratio"], series["ms"], "o-", lw=1.6, label=label)
    ax.axvline(1.0, ls="--", c="k", lw=1.2)
    ax.annotate(
        "graphs can pay  <--|-->  graphs cannot pay",
        (1.0, ax.get_ylim()[1]),
        ha="center",
        va="top",
        fontsize=8,
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"$ef_L \cdot M' \; / \; (N/N_c)$   (MobileRAG's own condition)")
    ax.set_ylabel("latency (ms/query)")
    ax.set_title("The regime condition, and where our configurations sit", fontsize=9)
    ax.legend(fontsize=8)
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
    runs_root = Path(args.runs) if args.runs else paths.results_root / "E_DEMO"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    wanted = set(args.only) if args.only else {"headline", "recall_effort", "footprint", "regime"}
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

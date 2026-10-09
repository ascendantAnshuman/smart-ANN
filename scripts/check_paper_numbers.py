#!/usr/bin/env python
"""Every number the paper states, checked against the file it came from.

A paper is the one artifact in this project that a reader cannot re-run. Its
numbers are therefore the easiest place for a typo to become a false claim, and
the hardest place to notice one. This script asserts each claim in the paper
against the recorded result it is drawn from, so a figure that drifts -- because
an experiment was re-run, or because a digit was mistyped -- fails here rather
than in review.

It checks values, not prose. A claim this cannot express (`the curve is
asymmetric`) is out of scope; a claim that reduces to arithmetic on recorded
numbers is in scope and is listed below.

There are two manuscripts drawing on the same results -- `paper/main.tex`
(conference) and `paper/journal.tex` (journal) -- so the file to check is an
argument, defaulting to the conference one:

    python scripts/check_paper_numbers.py                     # main.tex
    python scripts/check_paper_numbers.py paper/journal.tex   # journal.tex

The value checks are identical for both, because both are drawn from the same
recorded runs. Only the presence checks read the file.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

R = Path("results")
TEX = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("paper/main.tex")


def load(rel: str) -> dict:
    return json.loads((R / rel).read_text(encoding="utf-8"))


def main() -> int:
    # LaTeX groups thousands so the separator does not break kerning: main.tex
    # writes 19{,}719 and journal.tex writes 19\,719. Normalise both to a plain
    # comma, or every grouped number reads as absent.
    tex = TEX.read_text(encoding="utf-8").replace("{,}", ",").replace("\\,", ",")
    table = load("device_profile_table.json")
    lv = {e["nprobe"]: e for e in table["entries"]}
    p9 = load("phase9/query_adaptive.json")
    p6 = load("phase6/policy_comparison.json")
    p7 = load("phase7/qa_summary.json")
    en = load("energy_model.json")
    c5 = load("device/c5_summary.json")
    held = {c["uniform_nprobe"]: c["held_out"] for c in p9["comparisons"]}
    sep = p9["separation"]
    arms6 = {a["policy"]: a for a in p6["arms"]}
    arms7 = {a["arm"]: a for a in p7["arms"]}

    fails: list[str] = []
    checks = 0

    def eq(label: str, claimed: float, actual: float, tol: float = 5e-3) -> None:
        nonlocal checks
        checks += 1
        if not math.isclose(claimed, actual, rel_tol=tol, abs_tol=tol):
            fails.append(f"{label}: paper says {claimed}, results say {actual:.6g}")

    def present(label: str, text: str) -> None:
        nonlocal checks
        checks += 1
        if text not in tex:
            fails.append(f"{label}: '{text}' is not in the paper")

    # --- the profile table, transcribed into the paper -------------------
    for n in (1, 4, 8, 16, 32):
        present(f"profile p95 n={n}", f"{lv[n]['p95_ms']:.2f}")
        present(f"profile recall n={n}", f"{lv[n]['recall']:.4f}")

    # --- the effort curve's shape ----------------------------------------
    eq("first step latency", 38.0, lv[4]["p95_ms"] - lv[1]["p95_ms"], tol=1.0)
    eq("first step recall", 0.20, lv[4]["recall"] - lv[1]["recall"], tol=0.01)
    eq("last step latency", 164.0, lv[32]["p95_ms"] - lv[16]["p95_ms"], tol=1.0)
    eq("last step recall", 0.03, lv[32]["recall"] - lv[16]["recall"], tol=0.01)

    # --- the difficulty separation, the paper's central observation -------
    lo, hi = sep[0]["recall_by_level"]["1"], sep[-1]["recall_by_level"]["1"]
    eq("hardest quintile at nprobe=1", 0.278, lo)
    eq("easiest quintile at nprobe=1", 0.978, hi)
    eq("separation ratio", 3.5, hi / lo, tol=0.05)
    g_hard = sep[0]["recall_by_level"]["4"] - sep[0]["recall_by_level"]["1"]
    g_easy = sep[-1]["recall_by_level"]["4"] - sep[-1]["recall_by_level"]["1"]
    eq("hardest marginal gain", 0.403, g_hard)
    eq("easiest marginal gain", 0.009, g_easy)
    eq("marginal value ratio", 45.0, g_hard / g_easy, tol=1.0)

    # --- the held-out result ---------------------------------------------
    for n in (1, 4, 8, 16, 32):
        present(f"held-out uniform n={n}", f"{held[n]['uniform_recall']:.4f}")
        present(f"held-out adaptive n={n}", f"{held[n]['adaptive_recall']:.4f}")
    eq("best delta", 0.0162, held[8]["delta_recall"], tol=1e-4)
    eq("resolution floor", 0.0002, p9["recall_resolution"], tol=1e-9)
    eq("best delta in floors", 81.0, held[8]["delta_recall"] / p9["recall_resolution"], tol=1.0)
    eq("smallest win in floors", 45.0, held[16]["delta_recall"] / p9["recall_resolution"], tol=1.0)

    over8 = held[8]["adaptive_mean_cost_ms"] / held[8]["uniform_mean_cost_ms"] - 1
    over16 = held[16]["adaptive_mean_cost_ms"] / held[16]["uniform_mean_cost_ms"] - 1
    under4 = 1 - held[4]["adaptive_mean_cost_ms"] / held[4]["uniform_mean_cost_ms"]
    under32 = 1 - held[32]["adaptive_mean_cost_ms"] / held[32]["uniform_mean_cost_ms"]
    eq("overspend at n=8", 0.028, over8, tol=1e-3)
    eq("overspend at n=16", 0.034, over16, tol=1e-3)
    eq("underspend at n=4", 0.046, under4, tol=1e-3)
    eq("underspend at n=32", 0.088, under32, tol=1e-3)

    # The exchange-rate argument, which is the paper's defence of the result.
    alloc_rate = (held[8]["adaptive_recall"] / held[8]["uniform_recall"] - 1) / over8
    climb_rate = (held[16]["uniform_recall"] / held[8]["uniform_recall"] - 1) / (
        lv[16]["p95_ms"] / lv[8]["p95_ms"] - 1
    )
    eq("allocation exchange rate", 0.64, alloc_rate, tol=0.01)
    eq("uniform-climb exchange rate", 0.067, climb_rate, tol=0.005)
    eq("climb latency cost", 0.87, lv[16]["p95_ms"] / lv[8]["p95_ms"] - 1, tol=0.01)

    # --- policies ---------------------------------------------------------
    eq("static violations", 5.98, arms6["static"]["violation_pct"], tol=0.01)
    eq("budget violations", 0.17, arms6["budget"]["violation_pct"], tol=0.01)
    eq(
        "violation reduction",
        35.0,
        arms6["static"]["violation_pct"] / arms6["budget"]["violation_pct"],
        tol=1.0,
    )
    eq("recall paid", 0.068, arms6["static"]["recall"] - arms6["budget"]["recall"], tol=1e-3)
    eq("policy queries", 26699.0, sum(a["n"] for a in p6["arms"]), tol=1.0)
    eq("controller overhead", 0.167, max(a["overhead_p95_pct"] for a in p6["arms"]), tol=1e-3)
    eq("declared budget", 14.01, p6["budget_ms"], tol=0.01)

    # --- context reduction ------------------------------------------------
    for arm in ("off", "keep1", "keep2", "keep1_ext"):
        present(f"scr F1 {arm}", f"{arms7[arm]['f1']:.4f}")
    if max(arms7, key=lambda a: arms7[a]["f1"]) != "off":
        fails.append("the paper says reduction OFF is best; the results disagree")
    checks += 1

    # --- energy -----------------------------------------------------------
    m = en["measured"]
    eq("energy per query", 130.99, m["mj_per_query_allin"], tol=0.01)
    eq("mean power", 1.006, m["mean_power_w"], tol=1e-3)
    eq("total energy", 1266.6, m["total_energy_j"], tol=0.1)
    eq("mj per cluster", 3.21, en["attribution"]["mj_per_cluster"], tol=0.01)
    q = en["effort_vs_power"]["quintiles"]
    eq("power rise", 97.6, 100 * (q[-1]["power_median_w"] / q[0]["power_median_w"] - 1), tol=0.2)
    eq("energy r2", 0.16, en["attribution"]["r2"], tol=0.01)

    # --- C5, the negative result -----------------------------------------
    a5 = c5["arms"]
    eq("c5 static p95", 84.68, a5["static"]["p95_ms"], tol=0.01)
    eq("c5 adaptive p95", 43.29, a5["adaptive"]["p95_ms"], tol=0.01)
    eq("c5 static violations", 1.11, a5["static"]["violation_pct"], tol=0.01)
    eq("c5 adaptive violations", 0.01, a5["adaptive"]["violation_pct"], tol=0.005)
    eq("c5 recall cost", 0.10, a5["static"]["recall"] - a5["adaptive"]["recall"], tol=0.005)
    eq(
        "c5 blind fraction",
        2.7,
        100 * c5["adaptive_decisions_with_a_reason"] / c5["adaptive_decisions_total"],
        tol=0.1,
    )
    present("c5 decisions with a reason", f"{c5['adaptive_decisions_with_a_reason']}")
    present("c5 total decisions", f"{c5['adaptive_decisions_total']:,}")

    # --- sensor divergence ------------------------------------------------
    from smartann.telemetry.replay import SENSORS, available_traces, load_trace, trace_path

    peaks = {
        n: {s: load_trace(trace_path(R / "device", n), s).peak_c for s in SENSORS}
        for n in available_traces(R / "device")
    }
    checks += 1
    if not peaks:
        fails.append("no recorded handset traces to check the sensor claim against")
    else:
        skin = [p["skin"] for p in peaks.values()]
        hot = [p["hottest"] for p in peaks.values()]
        gaps = [h - s for h, s in zip(hot, skin, strict=True)]
        eq("n traces", 6.0, float(len(peaks)), tol=0.1)
        eq("skin peak min", 34.0, min(skin), tol=0.1)
        eq("skin peak max", 37.5, max(skin), tol=0.1)
        eq("hottest peak min", 54.1, min(hot), tol=0.1)
        eq("hottest peak max", 78.8, max(hot), tol=0.1)
        eq("divergence min", 20.1, min(gaps), tol=0.1)
        eq("divergence max", 41.3, max(gaps), tol=0.1)
        checks += 1
        if any(s >= 42.0 for s in skin):
            fails.append("the paper says skin never reaches 42 C; a trace disagrees")
        checks += 1
        if not all(h >= 42.0 for h in hot):
            fails.append("the paper says the hottest zone exceeds 42 C in all runs; one does not")

    # --- figures the paper includes must exist ---------------------------
    for m_ in re.finditer(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex):
        checks += 1
        if not (R / "paper" / m_.group(1)).exists():
            fails.append(f"figure {m_.group(1)} is cited but not built")

    print(f"{checks} checks against the recorded results")
    for f in fails:
        print(f"  FAIL  {f}")
    if fails:
        print(f"\n{len(fails)} of {checks} failed")
        return 1
    print("every checked number in the paper matches the file it came from")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

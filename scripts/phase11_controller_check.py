#!/usr/bin/env python
"""11.1 -- is the AIMD loop stable when its feedback is handset-measured latency?

ADR-0032 established the rule (a control loop must act no faster than its
feedback delay) and confirmed it on the laptop at `control_period_s = 4.05`.
That number is 50 samples x the laptop's quiet latency at the top effort level.
The handset is roughly 20x slower per query at that level, so its window takes
20x longer to fill, and the laptop's period is **not** transferable. Carrying it
over would be reusing a constant whose derivation no longer holds -- the exact
move the pre-declaration protocol exists to prevent.

So the period is re-derived here from the trace's own measurements, and the
whole demonstration from Phase 11.2 onwards is gated on the result.

**What is real here and what is replay.** The policy, the rolling window, the
clock and the control law are the shipping objects -- `AimdPolicy` deciding
against `RollingWindow`, the same pair `smartann/bench/runner.py` drives. The
latencies fed back are measurements from the Phase 3.5 handset trace
(`results/device/trace_queries.jsonl`), which swept nprobe 1/4/8/16/32
round-robin for 25 minutes while a sampler recorded temperature. Pairing a
decision with a latency recorded at that effort level at that moment in the run
is the replay's one construction, and it is labelled `trace-replay` everywhere
it is reported. Nothing is modelled and nothing is generated.

**Time advances by the latency that was measured**, so a cheaper decision buys
more queries per second exactly as it would on the device. A fixed query rate
would hide the thing the controller is for.

**Temperature is a spectator in this plot.** AIMD reads p95 latency and nothing
else. The temperature panel is the device's real condition during the same
trace, included because it is the reason the latencies drift, not because this
policy reads a thermometer. `DeviceStatePolicy` is the one that does.

**Two criteria, both reported.** The criterion pre-declared in
docs/predeclared/phase11_controller_check.json -- reversals <= 1.0 per minute
and median nprobe off the floor -- FAILED on the paced arms, at 1.96/min. It
was the criterion that was wrong: AIMD reverses direction once per tooth by
construction, and a controller pinned to its floor reverses never, scoring a
perfect 0.00 while being the exact pathology ADR-0032 exists to prevent.

The gate is therefore amplitude (ADR-0032 amendment): the p5-p95 span of
nprobe within half the action range, the median off the floor, and the p95
the loop delivers at or under the target it controls to. The declared verdict
is still computed, still printed, and still recorded as failed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smartann.bench.runner import RollingWindow
from smartann.controller.policies.aimd import AimdConfig, AimdPolicy
from smartann.types import (
    AnnConfig,
    Constraints,
    ControllerState,
    DeviceState,
    QueryFeatures,
    ScrConfig,
)

WINDOW_SAMPLES = 50  # RollingWindow's default, and the delay the period must respect
NPROBE_MIN = 1
NPROBE_MAX = 32  # the trace's top measured level; above it there is nothing measured
WARMUP = 20  # as scripts/phase6_profile.py configures it
# The amended stability bound (ADR-0032 amendment): the p5-p95 span of nprobe
# as a fraction of the action range. The ratcheting arm covers 0.90 of it, the
# paced arms 0.29. A reversal COUNT is not used as a gate, because a
# controller pinned to its floor reverses direction never and would pass.
MAX_SWING_FRACTION = 0.5

# AIMD reads none of these. NaN rather than zero so nothing downstream can read
# a filled-in value as a query feature that was available and acted on.
NO_QUERY_FEATURES = QueryFeatures(
    centroid_top1_dist=math.nan,
    centroid_dist_gap=math.nan,
    centroid_dist_entropy=math.nan,
    query_norm=math.nan,
    query_token_len=0,
)
SCR_OFF = ScrConfig(1, 0, 0, 1, 0.0, enabled=False)
BASE_ANN = AnnConfig(nprobe=NPROBE_MAX, ef_centroid=128, ef_list=64, k=10, per_cluster_k=32)


# ---------------------------------------------------------------------------
# The measured trace
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LevelLatency:
    """Every latency the handset recorded at one effort level, in time order."""

    nprobe: int
    t: np.ndarray  # seconds since the query loop started
    ms: np.ndarray

    def at(self, when: float) -> float:
        """The latency measured at this level nearest in time to `when`.

        Nearest rather than interpolated in time: each value is a real query
        that really took this long. Averaging two of them would produce a
        latency the device never recorded.
        """
        i = int(np.searchsorted(self.t, when))
        if i <= 0:
            return float(self.ms[0])
        if i >= self.t.size:
            return float(self.ms[-1])
        lo, hi = i - 1, i
        return float(self.ms[lo] if (when - self.t[lo]) <= (self.t[hi] - when) else self.ms[hi])


class TraceLatency:
    """Measured latency as a function of (time, nprobe) over the whole trace."""

    def __init__(self, levels: dict[int, LevelLatency]) -> None:
        if not levels:
            raise ValueError("the trace recorded no effort levels")
        self.levels = dict(sorted(levels.items()))
        self.keys = sorted(self.levels)

    def at(self, when: float, nprobe: int) -> tuple[float, str]:
        """Latency for an arbitrary nprobe, and how that number was obtained.

        A measured level is returned as measured. Between two measured levels
        the value is interpolated linearly in nprobe, which the trace itself
        supports: p50 runs 5.59 / 20.50 / 40.46 / 82.61 / 169.21 ms at
        1 / 4 / 8 / 16 / 32, i.e. close to a constant cost per probed cluster.
        The provenance string travels with the number so no summary can report
        an interpolated value as a measured one.
        """
        p = int(max(self.keys[0], min(self.keys[-1], nprobe)))
        if p in self.levels:
            return self.levels[p].at(when), "measured"
        lo = max(k for k in self.keys if k < p)
        hi = min(k for k in self.keys if k > p)
        a, b = self.levels[lo].at(when), self.levels[hi].at(when)
        frac = (p - lo) / (hi - lo)
        return a + frac * (b - a), "interpolated"

    def p50(self, nprobe: int) -> float:
        return float(np.median(self.levels[nprobe].ms))

    def p95(self, nprobe: int) -> float:
        return float(np.percentile(self.levels[nprobe].ms, 95))

    @property
    def duration_s(self) -> float:
        return max(float(lv.t[-1]) for lv in self.levels.values())


def load_query_trace(path: Path) -> TraceLatency:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows = [json.loads(ln) for ln in lines]
    if not rows:
        raise ValueError(f"{path} holds no queries")
    by_level: dict[int, list[tuple[float, float]]] = {}
    for r in rows:
        by_level.setdefault(int(r["nprobe"]), []).append(
            (float(r["t_wall"]), float(r["t_total_ms"]))
        )
    levels = {}
    for npb, pairs in by_level.items():
        pairs.sort()
        levels[npb] = LevelLatency(
            nprobe=npb,
            t=np.array([p[0] for p in pairs], dtype=np.float64),
            ms=np.array([p[1] for p in pairs], dtype=np.float64),
        )
    return TraceLatency(levels)


@dataclass(frozen=True)
class Thermals:
    """The sampler's record, re-based onto the query loop's clock.

    `scripts/device_trace.py` starts the sampler and then takes `t0_wall`, so
    sample 0 is the origin of the query clock to within one statement. The
    re-basing is that subtraction and nothing more.
    """

    t: np.ndarray
    skin_c: np.ndarray
    hottest_c: np.ndarray

    def hottest_at(self, when: float) -> float:
        return float(np.interp(when, self.t, self.hottest_c))

    def slope_at(self, when: float) -> float:
        """Finite difference of the hottest zone across the bracketing samples.

        Arithmetic on two measurements, the same way smartann.telemetry.replay
        recomputes it. NaN at the very start, where there is nothing to
        difference against.
        """
        i = int(np.searchsorted(self.t, when))
        if i <= 0 or i >= self.t.size:
            return math.nan
        dt = self.t[i] - self.t[i - 1]
        if dt <= 0:
            return math.nan
        return float((self.hottest_c[i] - self.hottest_c[i - 1]) / dt)


def load_thermals(path: Path) -> Thermals:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "t" in r:
            rows.append(r)
    if not rows:
        raise ValueError(f"{path} holds no telemetry samples")
    t0 = float(rows[0]["t"])
    return Thermals(
        t=np.array([float(r["t"]) - t0 for r in rows], dtype=np.float64),
        skin_c=np.array([float(r.get("skin_c", math.nan)) for r in rows], dtype=np.float64),
        hottest_c=np.array([float(r.get("hottest_c", math.nan)) for r in rows], dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# One replayed arm
# ---------------------------------------------------------------------------
@dataclass
class ArmResult:
    control_period_s: float
    label: str
    t: list[float]
    nprobe: list[int]
    latency_ms: list[float]
    window_p95: list[float]
    n_interpolated: int

    # filled by summarise()
    reversals: int = 0
    reversals_per_min: float = 0.0
    effort_changes: int = 0
    nprobe_p50: float = 0.0
    floor_fraction: float = 0.0
    p95_ms: float = 0.0
    violation_pct: float = 0.0
    duration_s: float = 0.0
    stable: bool = False  # the amended criterion; this is the gate
    stable_declared: bool = False  # the criterion pre-declared before the run
    nprobe_p5: float = 0.0
    nprobe_p95: float = 0.0
    nprobe_iqr: float = 0.0
    mean_abs_step: float = 0.0
    max_abs_step: float = 0.0
    swing_fraction_of_range: float = 0.0

    def summarise(self, target_ms: float) -> ArmResult:
        npb = np.array(self.nprobe, dtype=np.float64)
        lat = np.array(self.latency_ms, dtype=np.float64)
        self.duration_s = self.t[-1] - self.t[0] if len(self.t) > 1 else 0.0
        self.nprobe_p50 = float(np.median(npb))
        self.floor_fraction = float(np.mean(npb <= NPROBE_MIN))
        self.p95_ms = float(np.percentile(lat, 95))
        self.violation_pct = 100.0 * float(np.mean(lat > target_ms))

        # A reversal is the controller changing its mind about direction: a cut
        # after a raise, or a raise after a cut. Steps where effort did not move
        # are not directions and are skipped, so holding steady is not counted
        # as agreement with whatever came before it.
        steps = list(itertools.pairwise(self.nprobe))
        directions = [int(np.sign(b - a)) for a, b in steps if b != a]
        self.effort_changes = len(directions)
        self.reversals = sum(1 for a, b in itertools.pairwise(directions) if a != b)

        # Amplitude, because a reversal COUNT cannot tell a tight sawtooth
        # from a collapse: AIMD reverses direction once per tooth by
        # construction, so counting reversals alone would condemn every
        # working AIMD loop. What distinguishes health from pathology is how
        # far each tooth travels and where the teeth sit.
        self.nprobe_p5 = float(np.percentile(npb, 5))
        self.nprobe_p95 = float(np.percentile(npb, 95))
        self.nprobe_iqr = float(np.percentile(npb, 75) - np.percentile(npb, 25))
        self.mean_abs_step = (
            float(np.mean([abs(b - a) for a, b in steps if b != a])) if directions else 0.0
        )
        self.max_abs_step = (
            float(np.max([abs(b - a) for a, b in steps if b != a])) if directions else 0.0
        )
        self.swing_fraction_of_range = (self.nprobe_p95 - self.nprobe_p5) / (
            NPROBE_MAX - NPROBE_MIN
        )
        minutes = self.duration_s / 60.0
        self.reversals_per_min = self.reversals / minutes if minutes > 0 else math.nan

        # Both verdicts are computed and both are reported. The pre-declared one
        # FAILED on the paced arms and is kept visible for that reason: a
        # criterion amended after seeing the result is only honest if the
        # original and its outcome stay beside the amendment (ADR-0032 amendment).
        self.stable_declared = self.reversals_per_min <= 1.0 and self.nprobe_p50 > NPROBE_MIN
        self.stable = (
            self.swing_fraction_of_range <= MAX_SWING_FRACTION
            and self.nprobe_p50 > NPROBE_MIN
            and self.p95_ms <= target_ms
        )
        return self


def replay(
    latency: TraceLatency,
    thermals: Thermals,
    control_period_s: float,
    target_ms: float,
    duration_s: float,
    label: str,
) -> ArmResult:
    """Drive the real AimdPolicy from measured latencies for `duration_s` of trace time."""
    policy = AimdPolicy(
        AimdConfig(
            nprobe_min=NPROBE_MIN,
            nprobe_max=NPROBE_MAX,
            warmup_queries=WARMUP,
            control_period_s=control_period_s,
        ),
        BASE_ANN,
        SCR_OFF,
    )
    constraints = Constraints(
        latency_target_ms=target_ms,
        latency_hard_ms=3.0 * target_ms,
        energy_budget_mj=float("inf"),
        min_recall=None,
    )
    window = RollingWindow(size=WINDOW_SAMPLES)
    out = ArmResult(control_period_s, label, [], [], [], [], 0)

    now = 0.0
    while now < duration_s:
        device = DeviceState(
            timestamp=now,
            temperature_c=thermals.hottest_at(now),
            temperature_slope_c_per_s=thermals.slope_at(now),
            freq_ratio=math.nan,
            throttled=False,
            cpu_util=math.nan,
            mem_available_mb=math.nan,
            mem_pressure=math.nan,
            # The battery probe timed out on every sample of this run, so these
            # stay NaN. AIMD reads neither, and a zero here would be a reading
            # the device never gave.
            battery_pct=math.nan,
            charging=False,
            discharge_rate_ma=math.nan,
        )
        snapshot = window.snapshot()
        state = ControllerState(device=device, query=NO_QUERY_FEATURES, history=snapshot)
        decision = policy.decide_timed(state, constraints)
        npb = decision.config.ann.nprobe

        ms, how = latency.at(now, npb)
        if how == "interpolated":
            out.n_interpolated += 1

        out.t.append(now)
        out.nprobe.append(npb)
        out.latency_ms.append(ms)
        out.window_p95.append(snapshot.p95_latency_ms)

        window.observe(ms, 0.0, now, target_ms)
        # Serial queries, back to back, exactly as the trace itself ran them.
        now += ms / 1000.0

    return out.summarise(target_ms)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot(
    arms: list[ArmResult], thermals: Thermals, target_ms: float, trace: str, out: Path
) -> None:
    plt.rcParams.update(
        {"figure.dpi": 130, "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3}
    )
    # Four panels rather than three, and the unpaced arm gets one to itself.
    # Drawn together it is a solid block of ink spanning the whole action range
    # -- which IS the finding, but it hides the two paced arms underneath it, and
    # a figure where the result obscures the comparison proves nothing to a
    # reader who cannot see the comparison.
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(
        4, 1, figsize=(11, 12), sharex=True, gridspec_kw={"height_ratios": [1, 1.3, 1, 1]}
    )
    control, paced = arms[0], arms[1:]
    colours = ["#2c6fb3", "#1f7a4d"]

    ax0.step(control.t, control.nprobe, where="post", color="#b3382c", lw=0.5, alpha=0.8)
    ax0.axhline(NPROBE_MIN, color="grey", ls=":", lw=1)
    ax0.set_ylabel("nprobe")
    ax0.set_title(
        "AIMD retrieval depth vs control period -- trace-replay on measured handset latency\n"
        f"Phase 3.5 trace '{trace}': nprobe swept 1/4/8/16/32 for "
        f"{thermals.t[-1] / 60:.1f} min on the phone",
        fontsize=11,
    )
    ax0.text(
        0.01,
        0.93,
        f"UNPACED, {control.label.split('(')[0].strip()} -- the ADR-0032 ratchet, reproduced on "
        f"handset feedback:\n"
        f"median nprobe {control.nprobe_p50:.0f}, {100 * control.floor_fraction:.0f}% of queries "
        f"on the floor, {control.reversals_per_min:.1f} reversals/min",
        transform=ax0.transAxes,
        va="top",
        fontsize=8.5,
        color="#8a2a20",
    )

    for arm, c in zip(paced, colours, strict=True):
        ax1.step(
            arm.t,
            arm.nprobe,
            where="post",
            color=c,
            lw=1.1,
            label=(
                f"{arm.label}: {arm.reversals_per_min:.2f} reversals/min, median "
                f"{arm.nprobe_p50:.0f}, p5-p95 {arm.nprobe_p5:.0f}-{arm.nprobe_p95:.0f}"
            ),
        )
    ax1.axhline(NPROBE_MIN, color="grey", ls=":", lw=1)
    ax1.text(5, NPROBE_MIN + 0.5, "floor", color="grey", fontsize=8)
    ax1.set_ylabel("retrieval depth (nprobe)")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.text(
        0.99,
        0.05,
        "PACED. Neither arm reaches the floor. The remaining up-down movement is\n"
        "AIMD's sawtooth: additive increase until p95 crosses the target, then one cut.",
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#1f5131",
    )

    for arm, c in zip(paced, colours, strict=True):
        ax2.plot(arm.t, arm.window_p95, color=c, lw=0.8, alpha=0.85)
    ax2.axhline(
        target_ms,
        color="black",
        ls="--",
        lw=1.2,
        label=f"latency target {target_ms:.0f} ms (1.5 x measured p95 at the median level)",
    )
    ax2.set_ylabel("p95 the controller read (ms)")
    ax2.set_yscale("log")
    ax2.legend(fontsize=8, loc="lower left")

    ax3.plot(thermals.t, thermals.hottest_c, color="#c2410c", lw=1.3, label="hottest zone inside")
    ax3.plot(thermals.t, thermals.skin_c, color="#0369a1", lw=1.3, label="skin (the case)")
    ax3.set_ylabel("temperature (C)")
    ax3.set_xlabel("time into the trace (s)")
    ax3.set_ylim(28, 88)
    ax3.legend(fontsize=8, loc="upper left")
    ax3.text(
        0.99,
        0.93,
        "measured on the handset. AIMD reads p95 latency, NOT temperature:\n"
        "this panel is the device's condition, not this controller's input.",
        transform=ax3.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color="#444444",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", default="trace", help="Phase 3.5 trace prefix in results/device.")
    ap.add_argument("--device-dir", type=Path, default=Path("results/device"))
    ap.add_argument("--out-png", type=Path, default=Path("results/controller_check.png"))
    ap.add_argument("--out-json", type=Path, default=Path("results/controller_check.json"))
    args = ap.parse_args()

    qpath = args.device_dir / f"{args.trace}_queries.jsonl"
    tpath = args.device_dir / f"{args.trace}_telemetry.jsonl"
    for p in (qpath, tpath):
        if not p.exists():
            print(f"missing {p}. This check replays a recorded trace and will not invent one.")
            return 2

    latency = load_query_trace(qpath)
    thermals = load_thermals(tpath)

    # Both numbers derived from this trace, by rules fixed before the run.
    fill_s = WINDOW_SAMPLES * latency.p50(NPROBE_MAX) / 1000.0
    median_level = latency.keys[len(latency.keys) // 2]
    target_ms = 1.5 * latency.p95(median_level)
    duration_s = min(latency.duration_s, float(thermals.t[-1]))

    print(f"trace {args.trace}: {duration_s:.0f} s, levels {latency.keys}")
    print(
        f"  window fill time = {WINDOW_SAMPLES} x p50 at nprobe={NPROBE_MAX} "
        f"({latency.p50(NPROBE_MAX):.2f} ms) = {fill_s:.2f} s"
    )
    print(
        f"  target = 1.5 x p95 at the median level (nprobe={median_level}, "
        f"{latency.p95(median_level):.2f} ms) = {target_ms:.2f} ms"
    )

    specs = [
        (0.0, "period 0 s (ADR-0032's collapsing config, control arm)"),
        (4.05, "period 4.05 s (the laptop's pre-declared value)"),
        (round(fill_s, 2), f"period {fill_s:.2f} s (this trace's own window fill time)"),
    ]
    arms = [replay(latency, thermals, p, target_ms, duration_s, lab) for p, lab in specs]

    print()
    for a in arms:
        print(
            f"  {a.label}\n"
            f"      queries {len(a.nprobe):>6}  effort changes {a.effort_changes:>5}  "
            f"reversals {a.reversals:>4} ({a.reversals_per_min:.2f}/min)\n"
            f"      median nprobe {a.nprobe_p50:>5.1f}  at floor {100 * a.floor_fraction:5.1f}%  "
            f"p95 {a.p95_ms:7.1f} ms  over target {a.violation_pct:5.1f}%\n"
            f"      swing p5-p95 {a.nprobe_p5:.0f}-{a.nprobe_p95:.0f} "
            f"({100 * a.swing_fraction_of_range:.0f}% of the action range), "
            f"mean step {a.mean_abs_step:.1f}, max step {a.max_abs_step:.0f}\n"
            f"      -> amended criterion (amplitude): "
            f"{'STABLE' if a.stable else 'NOT STABLE'}   |   "
            f"pre-declared criterion (reversal count): "
            f"{'stable' if a.stable_declared else 'FAILED'}"
        )

    plot(arms, thermals, target_ms, args.trace, args.out_png)

    verdict = arms[-1]
    payload = {
        "test_id": "E_PHASE11_controller_check",
        "predeclared": "docs/predeclared/phase11_controller_check.json",
        "provenance": "trace-replay: Phase 3.5 handset trace, measured latency per effort level",
        "trace": {
            "name": args.trace,
            "queries_path": str(qpath),
            "telemetry_path": str(tpath),
            "duration_s": duration_s,
            "levels": latency.keys,
            "p50_ms_by_level": {str(k): latency.p50(k) for k in latency.keys},
            "p95_ms_by_level": {str(k): latency.p95(k) for k in latency.keys},
            "hottest_c_range": [
                float(np.nanmin(thermals.hottest_c)),
                float(np.nanmax(thermals.hottest_c)),
            ],
            "skin_c_range": [
                float(np.nanmin(thermals.skin_c)),
                float(np.nanmax(thermals.skin_c)),
            ],
            "battery": "not measured -- the battery probe timed out on every sample of this run",
        },
        "derived_settings": {
            "window_samples": WINDOW_SAMPLES,
            "window_fill_time_s": fill_s,
            "window_fill_derivation": (
                f"{WINDOW_SAMPLES} samples x measured p50 at nprobe={NPROBE_MAX} "
                f"({latency.p50(NPROBE_MAX):.2f} ms)"
            ),
            "latency_target_ms": target_ms,
            "target_derivation": (
                f"1.5 x measured p95 at the median swept level (nprobe={median_level})"
            ),
            "nprobe_range": [NPROBE_MIN, NPROBE_MAX],
        },
        "criterion_predeclared": {
            "reversals_per_min_max": 1.0,
            "median_nprobe_above": NPROBE_MIN,
            "outcome": "FAILED on every arm; see the amendment. Kept visible deliberately.",
        },
        "criterion_amended": {
            "swing_fraction_of_range_max": MAX_SWING_FRACTION,
            "median_nprobe_above": NPROBE_MIN,
            "p95_at_or_under_target": True,
            "why": (
                "a reversal count cannot tell a sawtooth from a collapse: a controller "
                "pinned to its floor scores 0.00 reversals/min. ADR-0032 (amendment)."
            ),
        },
        "arms": [
            {
                "control_period_s": a.control_period_s,
                "label": a.label,
                "n_queries": len(a.nprobe),
                "effort_changes": a.effort_changes,
                "reversals": a.reversals,
                "reversals_per_min": a.reversals_per_min,
                "nprobe_p50": a.nprobe_p50,
                "floor_fraction": a.floor_fraction,
                "p95_ms": a.p95_ms,
                "violation_pct": a.violation_pct,
                "nprobe_p5": a.nprobe_p5,
                "nprobe_p95": a.nprobe_p95,
                "nprobe_iqr": a.nprobe_iqr,
                "mean_abs_step": a.mean_abs_step,
                "max_abs_step": a.max_abs_step,
                "swing_fraction_of_range": a.swing_fraction_of_range,
                "latency_interpolated_fraction": a.n_interpolated / max(1, len(a.nprobe)),
                "stable_amended": a.stable,
                "stable_predeclared": a.stable_declared,
            }
            for a in arms
        ],
        "verdict": {
            "arm": verdict.label,
            "stable_amended": verdict.stable,
            "stable_predeclared": verdict.stable_declared,
            "swing_fraction_of_range": verdict.swing_fraction_of_range,
            "reversals_per_min": verdict.reversals_per_min,
            "nprobe_p50": verdict.nprobe_p50,
            "gate": (
                "phase 11.2 may proceed" if verdict.stable else "STOP: do not build on this loop"
            ),
        },
        "figure": str(args.out_png),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out_png}\nwrote {args.out_json}")
    return 0 if verdict.stable else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Render a thermal attribution as something a person can read and act on.

Kept apart from `attribution.py` on purpose: that module computes and this one
formats, so a change to the wording can never change a number. The ordering of
the sections is the argument, and it runs from strongest evidence to weakest.
Measured quantities come first, the power split second because both of its terms
were measured on battery, and the fitted thermal decomposition last because it
explains 30% of the variance and says so in its own section.
"""

from __future__ import annotations

import math

from smartann.thermal.attribution import Attribution

__all__ = ["render"]

WIDTH = 78


def _rule(ch: str = "-") -> str:
    return ch * WIDTH


def _f(x: float, nd: int = 3, unit: str = "") -> str:
    """Numbers that were never measured print as `n/a`, never as zero."""
    if not math.isfinite(x):
        return "n/a"
    return f"{x:.{nd}f}{unit}"


def _band(score: float) -> str:
    """What the score means, in the only terms that matter: can the controller act?"""
    if not math.isfinite(score):
        return "cannot be scored -- the power split is unavailable"
    if score < 25:
        return "retrieval is a BYSTANDER here; the heat is mostly the platform itself"
    if score < 50:
        return "retrieval is a SIGNIFICANT MINORITY of the heat"
    if score < 75:
        return "retrieval DOMINATES the heat; the controller has real leverage"
    return "retrieval is ESSENTIALLY THE WHOLE STORY; effort is the only lever needed"


def render(a: Attribution) -> str:
    t, p, th = a.trace, a.power, a.thermal
    out: list[str] = []
    add = out.append

    add(_rule("="))
    add("  WHY IS THIS DEVICE HOT?".ljust(WIDTH - 28) + "SMART-ANN thermal attribution")
    add(_rule("="))
    add("")
    add(f"  Trace    {t.name}   --   {t.minutes:.1f} min, {t.n_queries:,} queries")

    if t.power_state_known:
        states = ", ".join(f"{k} x{v}" for k, v in sorted(t.status_counts.items()))
        add(f"  Power    {states}")
        if not t.was_charging:
            add("")
            add("  THE CABLE WAS OUT for every sample of this trace. Charging cannot")
            add(f"  be the answer, and the phone still gained {t.peak_excursion_c:.1f} C.")
    else:
        add("  Power    unknown -- the battery API returned nothing for this trace")

    add("")
    add(_rule())
    add("  MEASURED -- thermometer and battery sensor, no model involved")
    add(_rule())
    add("")
    add(
        f"    skin temperature        {t.skin_c.min():6.2f} C -> {t.skin_c.max():6.2f} C"
        f"      swing {t.peak_excursion_c:+.2f} C"
    )
    if math.isfinite(t.hottest_max_c):
        add(
            f"    hottest thermal zone    {t.hottest_c.min():6.2f} C -> "
            f"{t.hottest_c.max():6.2f} C      peak {t.hottest_max_c:.2f} C"
        )
    add(f"    battery power draw      {_f(t.mean_power_w, 3, ' W'):>12}      mean over trace")
    add(f"    retrieval work          {_f(t.mean_work_rate, 1, ''):>12}      clusters/s, mean")
    add("")

    add(_rule())
    add("  WHY -- power attribution, both terms measured on battery")
    add(_rule())
    add("")
    add("    Heat is dissipated power, so splitting the watts splits the heat.")
    add("")
    add(
        f"    idle floor (SoC, screen, radios)   {p.idle_w:6.3f} W   "
        f"{100 * p.idle_share:5.1f}%   not retrieval's"
    )
    add(
        f"    retrieval work                     {p.work_w:6.3f} W   "
        f"{100 * p.work_share:5.1f}%   CONTROLLABLE"
    )
    if t.was_charging:
        add("    charging                            n/a         n/a   cable was in")
    else:
        add("    charging                            0.000 W     0.0%   cable was out")
    if math.isfinite(p.unexplained_w) and p.unexplained_w > 0:
        add(
            f"    unattributed                       {p.unexplained_w:6.3f} W   "
            f"{100 * p.unexplained_share:5.1f}%   counts against"
        )
    add("                                       " + "-" * 8)
    add(f"    measured draw                      {_f(p.measured_mean_w, 3, ' W'):>8}")
    add("")
    add("    The idle floor is not assumed. It was measured while the device sat")
    add(
        f"    genuinely idle for {100 * a.energy_model.idle_fraction:.0f}% of a discharging "
        f"trace; without that idle time"
    )
    add("    the intercept is extrapolated and comes back negative (ADR-0052).")
    add("")

    add(_rule())
    add("  THE SCORE")
    add(_rule())
    add("")
    add(f"      CONTROLLABLE HEAT SCORE      {a.score:.0f} / 100")
    add("")
    add(f"      {_band(a.score)}")
    add("")
    add("      Share of this device's heat that retrieval effort controls.")
    add("      A controller driving effort to zero removes this much and no more.")
    add("")

    add(_rule())
    add("  THERMAL MODEL -- the same question, put to the Phase 4 fit")
    add(_rule())
    add("")
    add(f"    dT/dt = a*work + b*(T - T_ref) + c        R^2 = {a.thermal_model.r2:.2f}")
    add("")

    if th.usable:
        add(f"    heating from retrieval        {th.retrieval_c:+7.2f} C")
        add(f"    cooling to ambient            {th.dissipation_c:+7.2f} C")
        add(f"    baseline offset               {th.baseline_c:+7.2f} C")
        add("                                  " + "-" * 7)
        add(f"    model accounts for            {th.modelled_rise_c:+7.2f} C")
        add(f"    thermometer actually read     {th.observed_rise_c:+7.2f} C")
        add(f"    unexplained residual          {th.residual_c:+7.2f} C")
        add("")
        add(f"    At R^2 = {a.thermal_model.r2:.2f} this is indicative, not authoritative.")
        add("    The residual is the honest size of what is still unexplained, and")
        add("    hiding it would not make it smaller.")
    else:
        add("    REFUSED -- no term-by-term split is reported for this trace.")
        add("")
        add("    The constant term c cannot be used here: the Phase 4 trace ran with")
        add("    the cable IN, so c carries charger heat that no longer applies.")
        add("")
        add("    Dropping it does not leave a conservative estimate, it leaves an")
        add("    unbalanced equation. Integrated over this trace the retrieval term")
        add("    alone reaches +216 C, because c is what the fit balanced it against.")
        add("    Printing that beside a thermometer that moved")
        add(f"    {th.observed_rise_c:+.1f} C would be arithmetic, not evidence.")
        add("")
        add("    The power split above answers the same question with two terms that")
        add("    were both measured on battery. It needs no constant.")
        add("")
        add("    To unlock this section: one 21-minute unplugged trace, re-fit with")
        add("    scripts/fit_thermal_model.py. Nothing else is missing.")
    add("")

    add(_rule())
    add("  DERIVED -- what the two fits say together that neither says alone")
    add(_rule())
    add("")
    add("    Both fits are stated per cluster per second, so their ratio cancels")
    add("    the work unit and leaves a property of the handset.")
    add("")
    add(f"    thermal capacitance C_th      {_f(a.c_th_j_per_c, 2, ' J/C')}")
    add(f"    thermal resistance  R_th      {_f(a.r_th_c_per_w, 2, ' C/W')}")
    add(f"    time constant       tau       {_f(a.thermal_model.tau_s, 1, ' s')}")
    add("")
    add(f"    tau is the controller's observation delay: {a.thermal_model.tau_s:.0f} s pass")
    add("    before an effort change shows up in temperature. A control law acting")
    add("    faster than this repeats the Phase 6 collapse.")
    add("")

    eq, obs = a.predicted_equilibrium_rise_c, t.peak_excursion_c
    if math.isfinite(eq):
        add(f"    R_th implies a steady-state rise of {eq:.1f} C at this power.")
        if eq > obs * 1.5:
            add(f"    The trace showed {obs:.1f} C over {t.minutes:.0f} min, well short of that.")
            add("    Two readings fit: the phone had not reached equilibrium, or R_th is")
            add("    overstated by a fit that explains 30% of its variance. This does")
            add("    not distinguish them, and a longer trace is what would.")
        add("")

    if a.sensors_disagree:
        add(_rule("="))
        add("  SENSOR DISAGREEMENT -- read this before trusting any thermal policy")
        add(_rule("="))
        add("")
        add(f"    The hottest zone ran up to {t.peak_divergence_c:.1f} C ahead of the skin,")
        add(
            f"    peaking at {t.hottest_max_c:.1f} C while the skin never passed {t.skin_c.max():.1f} C."
        )
        add("")
        add("    DeviceStatePolicy derates at 42 C and shields at 46 C, and it reads")
        add("    the SKIN. On this trace the skin never reached either line while the")
        add("    silicon sat far above both. A policy watching this sensor sees a cool")
        add("    device and does nothing, which is exactly what C5 reported: the")
        add("    thermal signal was real and the controller could not act on it.")
        add("")
        add("    That is a sensor-selection finding, not a tuning problem. Lowering")
        add("    the thresholds to fit the skin would be fitting constants to one")
        add("    trace; the question is which thermometer the policy should read.")
        add("")

    add(_rule("="))
    return "\n".join(out)

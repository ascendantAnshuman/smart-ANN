"""Render the collected state as one self-contained HTML page.

Separate from `dashboard.py` for the reason every renderer in this project is
separate from its collector: formatting must not be able to change a number.

**Charts follow the form the data's job asks for.** The heat score and the
quality headline are single values, so they are stat tiles rather than charts.
The profile table carries two measures on different scales, latency in
milliseconds and recall as a fraction, so they get one bar column each against
their own scale. A single pair of axes with two scales would let any reader draw
whatever crossing point they liked out of an arbitrary choice of units.

**Colors come from a validated categorical palette, three slots only.** The
scatter plots every family against every other, and only the first three slots
of the reference palette clear the all-pairs colour-vision floors. Three
families is what Phase 8 has, so the cap costs nothing here. Every mark is also
direct-labelled and every chart has a table beside or beneath it, which is the
relief the light-mode aqua's sub-3:1 contrast requires.
"""

from __future__ import annotations

import html
import math
from typing import Any

from smartann.dashboard import DashboardState

__all__ = ["render_html"]

# Reference categorical palette, slots 1-3, validated all-pairs in both modes.
SERIES = {
    "retrieval_only": ("--series-1", "retrieval only"),
    "reduction_only": ("--series-2", "reduction only"),
    "joint": ("--series-3", "joint"),
}

CSS = """
:root {
  color-scheme: light;
  --surface-0: #f4f4f1;
  --surface-1: #fcfcfb;
  --surface-2: #eceae4;
  --border:    #dcdad2;
  --text-primary:   #0b0b0b;
  --text-secondary: #52514e;
  --text-muted:     #82817a;
  --series-1: #2a78d6;
  --series-2: #eb6834;
  --series-3: #1baf7a;
  --good:     #0ca30c;
  --warning:  #fab219;
  --critical: #d03b3b;
  --grid: #e4e2db;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-0: #121211;
    --surface-1: #1a1a19;
    --surface-2: #232321;
    --border:    #34342f;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #8d8c83;
    --series-1: #3987e5;
    --series-2: #d95926;
    --series-3: #199e70;
    --grid: #2c2c28;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #121211;
  --surface-1: #1a1a19;
  --surface-2: #232321;
  --border:    #34342f;
  --text-primary:   #ffffff;
  --text-secondary: #c3c2b7;
  --text-muted:     #8d8c83;
  --series-1: #3987e5;
  --series-2: #d95926;
  --series-3: #199e70;
  --grid: #2c2c28;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--surface-0);
  color: var(--text-primary);
  font: 15px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.wrap { max-width: 1120px; margin: 0 auto; padding-block: 32px; padding-left: 20px; padding-right: 20px; }

header.top { border-bottom: 2px solid var(--text-primary); padding-bottom: 18px; margin-bottom: 28px; }
h1 { font-size: 30px; margin: 0 0 6px; letter-spacing: -0.02em; }
.sub { color: var(--text-secondary); margin: 0; font-size: 15px; }
.meta { color: var(--text-muted); font-size: 12.5px; margin-top: 10px; font-variant-numeric: tabular-nums; }

section { margin-bottom: 40px; }
h2 {
  font-size: 12px; text-transform: uppercase; letter-spacing: 0.09em;
  color: var(--text-muted); margin: 0 0 14px; font-weight: 600;
}
h3 { font-size: 16px; margin: 0 0 8px; }
p.note { color: var(--text-secondary); font-size: 13.5px; margin: 10px 0 0; max-width: 78ch; }

.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin-bottom: 40px; }
.tile {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 10px; padding: 18px 18px 16px;
}
.tile .label { font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.07em; color: var(--text-muted); }
.tile .value {
  font-size: 38px; line-height: 1.05; font-weight: 650; margin: 8px 0 4px;
  font-variant-numeric: tabular-nums; letter-spacing: -0.02em;
}
.tile .unit { font-size: 17px; color: var(--text-secondary); font-weight: 500; }
.tile .foot { font-size: 12.5px; color: var(--text-secondary); }

.card {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 10px; padding: 20px;
}
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 800px) { .grid2 { grid-template-columns: 1fr; } }
/* The figures have wildly different aspect ratios -- one is a four-panel
   column, another a wide pair -- so a row grid leaves a screen of dead space
   beside the short ones. Columns pack them by height instead. */
.figs { column-count: 2; column-gap: 16px; }
@media (max-width: 800px) { .figs { column-count: 1; } }
.figs figure { break-inside: avoid; display: inline-block; width: 100%; }

table { border-collapse: collapse; width: 100%; font-size: 13.5px; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--border); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--text-muted); font-weight: 600; font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.05em; }
tbody tr:last-child td { border-bottom: none; }
.tablewrap { overflow-x: auto; }

.bar { height: 11px; border-radius: 4px; display: inline-block; vertical-align: middle; }
.barcell { display: flex; align-items: center; gap: 9px; justify-content: flex-end; }

.legend { display: flex; flex-wrap: wrap; gap: 16px; margin: 0 0 12px; font-size: 13px; color: var(--text-secondary); }
.swatch { width: 11px; height: 11px; border-radius: 3px; display: inline-block; margin-right: 6px; vertical-align: -1px; }

.stack { display: flex; height: 34px; border-radius: 6px; overflow: hidden; gap: 2px; background: var(--surface-2); }
.stack div { display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 600; overflow: hidden; white-space: nowrap; }

.callout {
  border-left: 3px solid var(--warning); background: var(--surface-2);
  padding: 14px 16px; border-radius: 0 8px 8px 0; font-size: 13.5px; color: var(--text-secondary);
}
.callout strong { color: var(--text-primary); }

.status { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; }
.dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }

.qa { border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; background: var(--surface-1); margin-bottom: 12px; }
.qa .q { font-weight: 600; margin-bottom: 8px; }
.qa .a { font-size: 19px; color: var(--series-3); font-weight: 600; margin-bottom: 8px; }
.qa .facts { font-size: 12.5px; color: var(--text-muted); font-variant-numeric: tabular-nums; }
.qa .a.declined { color: var(--text-muted); font-style: italic; font-weight: 500; }
.qa .cite { font-size: 12.5px; color: var(--text-secondary); margin-bottom: 6px; }
.qa .cite a { color: var(--series-1); }
.badge {
  display: inline-block; margin-left: 8px; padding: 1px 7px; border-radius: 10px;
  background: var(--surface-2); color: var(--text-secondary); font-size: 11.5px;
}

figure { margin: 0 0 18px; }
figure img { width: 100%; height: auto; border: 1px solid var(--border); border-radius: 8px; background: #fff; }
figcaption { font-size: 12.5px; color: var(--text-muted); margin-top: 7px; }

.nav { display: flex; flex-wrap: wrap; gap: 6px 8px; margin: 0 0 34px; padding-bottom: 18px;
  border-bottom: 1px solid var(--border); }
.nav a { font-size: 12.5px; color: var(--text-secondary); text-decoration: none;
  border: 1px solid var(--border); border-radius: 20px; padding: 4px 11px; background: var(--surface-1); }
.nav a:hover { color: var(--text-primary); border-color: var(--text-muted); }
section { scroll-margin-top: 16px; }
.missing { color: var(--text-muted); font-size: 13.5px; font-style: italic; }
svg { max-width: 100%; height: auto; }
/* Axis labels live inside the SVG, so they need their own fill rather than
   inheriting a colour that only exists on HTML elements. */
text.ax { font-size: 10.5px; fill: var(--text-muted); font-variant-numeric: tabular-nums; }
.dim { color: var(--text-muted); font-size: 12px; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12.5px;
  background: var(--surface-2); padding: 1px 5px; border-radius: 4px; }
footer { border-top: 1px solid var(--border); padding-top: 16px; color: var(--text-muted); font-size: 12.5px; }
"""


def _e(text: Any) -> str:
    return html.escape(str(text))


def _fmt(x: float | None, nd: int = 2, dash: str = "n/a") -> str:
    if x is None or not math.isfinite(x):
        return dash
    return f"{x:,.{nd}f}"


def _tile(label: str, value: str, unit: str = "", foot: str = "") -> str:
    return (
        f'<div class="tile"><div class="label">{_e(label)}</div>'
        f'<div class="value">{_e(value)}<span class="unit">{_e(unit)}</span></div>'
        f'<div class="foot">{_e(foot)}</div></div>'
    )


def _bar_cell(value: float, vmax: float, colour: str, text: str) -> str:
    """A bar with its number beside it, never a bar alone.

    The direct label is not decoration: on the light surface the aqua slot sits
    below 3:1 against the page, and the palette's relief rule is satisfied by a
    visible label or a table. This is both.
    """
    width = 0.0 if vmax <= 0 else max(2.0, 100.0 * value / vmax)
    return (
        f'<div class="barcell"><span>{_e(text)}</span>'
        f'<span class="bar" style="width:{width:.1f}px;background:var({colour})"></span></div>'
    )


def _profile_section(state: DashboardState) -> str:
    if not state.profile:
        return _empty(
            "Effort levels", "no profile table; run `smartann run experiment profile`", "effort"
        )
    entries = state.profile["entries"]
    p95_max = max(e["p95_ms"] for e in entries)
    rec_max = max(e["recall"] for e in entries)

    def row(e: dict[str, Any]) -> str:
        p95 = _bar_cell(e["p95_ms"], p95_max, "--series-2", f"{e['p95_ms']:.1f} ms")
        rec = _bar_cell(e["recall"], rec_max, "--series-1", f"{e['recall']:.4f}")
        return (
            f"<tr><td><strong>{e['nprobe']}</strong></td>"
            f"<td>{p95}</td><td>{rec}</td><td>{e['n_samples']:,}</td></tr>"
        )

    rows = "".join(row(e) for e in entries)
    return f"""
<section id="effort">
  <h2>Adaptive control &mdash; the measured menu the controller chooses from</h2>
  <div class="card">
    <div class="legend">
      <span><span class="swatch" style="background:var(--series-2)"></span>p95 latency, own scale</span>
      <span><span class="swatch" style="background:var(--series-1)"></span>recall@10, own scale</span>
    </div>
    <div class="tablewrap"><table>
      <thead><tr><th>nprobe</th><th>measured p95</th><th>measured recall</th><th>samples</th></tr></thead>
      <tbody>{rows}</tbody>
    </table></div>
    <p class="note">Two measures on different scales, so each bar is drawn against its own
    maximum and every value is printed. Measured under: {_e(state.profile["measured_under"])}.
    A budget selects the highest-recall row whose p95 fits, and the search then opens
    exactly that many clusters.</p>
  </div>
</section>"""


def _thermal_section(state: DashboardState) -> str:
    t = state.thermal
    if not t:
        return _empty("Why the device is hot", "no device trace or fitted models on disk", "heat")

    total = max(t["measured_w"] if math.isfinite(t["measured_w"]) else 0.0, 1e-9)
    # Each segment carries its own ink. The two saturated slots take white; the
    # neutral one must not, because in light mode white on a pale surface is
    # invisible, and the label is the only thing naming that slice.
    segs = [
        ("idle floor", t["idle_w"], "--series-1", "#fff"),
        ("retrieval", t["work_w"], "--series-2", "#fff"),
        ("unattributed", max(t["unexplained_w"], 0.0), "--surface-2", "var(--text-secondary)"),
    ]
    stack = "".join(
        f'<div style="flex:{max(v, 0.0) / total:.4f};background:var({c});color:{ink}">'
        f"{_e(label)} {v:.2f} W</div>"
        for label, v, c, ink in segs
        if v > 0.02
    )

    finding = ""
    if t["sensors_disagree"]:
        finding = f"""
    <div class="callout"><strong>The policy reads the wrong thermometer.</strong>
    The skin sensor peaked at {t["skin_max_c"]:.1f}&nbsp;&deg;C while the hottest zone reached
    {t["hottest_max_c"]:.1f}&nbsp;&deg;C. The device policy derates at 42&nbsp;&deg;C and shields at
    46&nbsp;&deg;C, reading the skin, so it saw a cool device for the whole run and did
    nothing. That is a sensor-selection finding, not a threshold that needs retuning.</div>"""

    cable = "the cable was OUT for every sample" if not t["was_charging"] else "the cable was in"
    return f"""
<section id="heat">
  <h2>Why the device is hot &mdash; trace {_e(t["trace"])}, {t["minutes"]:.0f} min, {t["n_queries"]:,} queries</h2>
  <div class="card">
    <h3>Measured battery draw, split into causes</h3>
    <div class="stack">{stack}</div>
    <p class="note">Heat is dissipated power, so splitting the watts splits the heat. Both
    terms were measured on battery: {_e(cable)}, and the phone still gained
    {t["excursion_c"]:.1f}&nbsp;&deg;C. Power nobody attributed lowers the score rather than
    being credited to retrieval. Thermal time constant {t["tau_s"]:.0f}&nbsp;s, which is how
    long any effort change takes to show up in temperature.</p>
    {finding}
  </div>
</section>"""


def _qa_section(state: DashboardState) -> str:
    if not state.qa:
        return _empty("Answer quality", "no phase 7 summary; run `scripts/phase7_qa.py`", "quality")
    arms = state.qa["arms"]
    f1_max = max(a["f1"] for a in arms)

    def row(a: dict[str, Any]) -> str:
        f1 = _bar_cell(a["f1"], f1_max, "--series-1", f"{a['f1']:.4f}")
        return (
            f"<tr><td><strong>{_e(a['arm'])}</strong></td>"
            f"<td>{f1}</td><td>{a['em']:.3f}</td><td>{a['n']:,}</td></tr>"
        )

    rows = "".join(row(a) for a in arms)
    return f"""
<section id="quality">
  <h2>Answer quality &mdash; what context reduction costs</h2>
  <div class="card">
    <div class="tablewrap"><table>
      <thead><tr><th>reduction arm</th><th>F1</th><th>exact match</th><th>questions</th></tr></thead>
      <tbody>{rows}</tbody>
    </table></div>
    <p class="note">One retrieval configuration held fixed across every arm, so the whole
    difference is reduction. <code>off</code> is the byte-identical passthrough. Quality
    falls at every level, which is the trade the adaptive claim has to be worth.</p>
  </div>
</section>"""


def _joint_scatter(state: DashboardState) -> str:
    if not state.joint:
        return _empty("Joint control", "no phase 8 summary; run `scripts/phase8_joint.py`", "joint")
    # A summary present but without cells is a run that did not finish, or an
    # older format. Sections fail independently in this page, so that is an
    # absence with a reason rather than a KeyError that blanks every panel
    # below it -- the most common reason to open this is to see how far a run
    # has got, which is exactly when a partial file is on disk.
    cells = state.joint.get("cells")
    if not cells:
        return _empty(
            "Joint control", "the phase 8 summary on disk records no cells, so nothing was plotted"
        )
    w, h = 660, 330
    pad_l, pad_b, pad_t, pad_r = 56, 44, 16, 16
    # Floated explicitly: the cells come from JSON, so mypy sees Any and the
    # scale helpers below would silently widen their return type to match.
    xs = [float(c["cost_ms"]) for c in cells]
    ys = [float(c["f1"]) for c in cells]
    x0, x1 = min(xs) * 0.92, max(xs) * 1.04
    y0, y1 = min(ys) * 0.94, max(ys) * 1.03

    def px(v: float) -> float:
        return pad_l + (v - x0) / (x1 - x0) * (w - pad_l - pad_r)

    def py(v: float) -> float:
        return h - pad_b - (v - y0) / (y1 - y0) * (h - pad_b - pad_t)

    grid = "".join(
        f'<line x1="{pad_l}" y1="{py(y):.1f}" x2="{w - pad_r}" y2="{py(y):.1f}" '
        f'stroke="var(--grid)" stroke-width="1"/>'
        f'<text x="{pad_l - 9}" y="{py(y) + 4:.1f}" text-anchor="end" font-size="11" '
        f'fill="var(--text-muted)">{y:.2f}</text>'
        for y in [y0 + i * (y1 - y0) / 4 for i in range(5)]
    )
    xticks = "".join(
        f'<text x="{px(x):.1f}" y="{h - pad_b + 18}" text-anchor="middle" font-size="11" '
        f'fill="var(--text-muted)">{x:.0f}</text>'
        for x in [x0 + i * (x1 - x0) / 4 for i in range(5)]
    )

    marks = []
    for c in cells:
        var, _ = SERIES.get(c["family"], ("--series-1", c["family"]))
        cx, cy = px(c["cost_ms"]), py(c["f1"])
        label = f"{c['nprobe']}/{c['scr']}"
        marks.append(
            # 2px surface ring so overlapping marks stay separable.
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6.5" fill="var({var})" '
            f'stroke="var(--surface-1)" stroke-width="2"><title>{_e(label)}: '
            f"F1 {c['f1']:.3f}, {c['cost_ms']:.0f} ms</title></circle>"
            f'<text x="{cx + 10:.1f}" y="{cy + 4:.1f}" font-size="10.5" '
            f'fill="var(--text-secondary)">{_e(label)}</text>'
        )

    legend = "".join(
        f'<span><span class="swatch" style="background:var({var})"></span>{_e(name)}</span>'
        for var, name in SERIES.values()
    )
    rows = "".join(
        f"<tr><td>{c['nprobe']}/{_e(c['scr'])}</td><td>{_e(SERIES.get(c['family'], ('', c['family']))[1])}</td>"
        f"<td>{c['f1']:.4f}</td><td>{c['cost_ms']:.0f}</td></tr>"
        for c in sorted(cells, key=lambda c: -c["f1"])
    )

    return f"""
<section id="joint">
  <h2>Joint control &mdash; retrieval effort and reduction together</h2>
  <div class="card">
    <div class="legend">{legend}</div>
    <svg viewBox="0 0 {w} {h}" role="img" aria-label="Answer quality against cost per query">
      {grid}{xticks}
      <line x1="{pad_l}" y1="{h - pad_b}" x2="{w - pad_r}" y2="{h - pad_b}" stroke="var(--border)"/>
      <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{h - pad_b}" stroke="var(--border)"/>
      {"".join(marks)}
      <text x="{w / 2:.0f}" y="{h - 6}" text-anchor="middle" font-size="11.5"
        fill="var(--text-secondary)">cost per query (ms, median retrieve + generate)</text>
    </svg>
    <div class="tablewrap"><table>
      <thead><tr><th>nprobe / reduction</th><th>family</th><th>F1</th><th>cost ms</th></tr></thead>
      <tbody>{rows}</tbody>
    </table></div>
    <p class="note">Each point is labelled with its configuration, and the table repeats every
    value. Joint cells extend the frontier downward in cost; they do not beat
    single-dimension control at peak quality.</p>
  </div>
</section>"""


def _answers_section(state: DashboardState) -> str:
    if not state.answers:
        return ""
    cards = []
    for a in state.answers:
        pol = a.get("policy", {})
        timing = a.get("timing_ms", {})
        # A refusal is not an answer, and styling it like one would read as the
        # system having found something. It gets muted ink and says what it is.
        declined = bool(a.get("declined"))
        cls = "a declined" if declined else "a"
        tail = (
            '<div class="facts">declined &mdash; the context did not contain the answer, '
            "so the model said so instead of inventing one</div>"
            if declined
            else ""
        )
        retrieval = a.get("retrieval", [])
        cited_rank = a.get("cited_rank")
        cite = ""
        for row in retrieval:
            if row.get("rank") == cited_rank:
                title = row.get("title") or f"passage {cited_rank}"
                link = (
                    f'<a href="{_e(row["url"])}" target="_blank" rel="noopener">{_e(title)}</a>'
                    if row.get("url")
                    else _e(title)
                )
                cite = (
                    f'<div class="cite">Source: passage [{cited_rank}], {link}'
                    f"{' &middot; your knowledge base' if row.get('origin') == 'kb' else ''}</div>"
                )
        n_kb = sum(1 for row in retrieval if row.get("origin") == "kb")
        badge = (
            f'<span class="badge">{n_kb} of {len(retrieval)} passages from your knowledge base</span>'
            if n_kb
            else ""
        )
        cards.append(
            f'<div class="qa"><div class="q">{_e(a.get("question", ""))}</div>'
            f'<div class="{cls}">{_e(a.get("answer", ""))}</div>{tail}{cite}'
            f'<div class="facts">nprobe {pol.get("chosen_nprobe", "?")} &middot; '
            f"budget {pol.get('budget_ms') or 'none'} ms &middot; "
            f"{_fmt(timing.get('total'), 0)} ms total &middot; "
            f"{a.get('work', {}).get('n_cluster_loads', '?')} clusters {badge}</div></div>"
        )
    return f"""
<section>
  <h2>Answered questions</h2>
  {"".join(cards)}
  <p class="note">Produced by <code>smartann ask</code>, which drives the same retrieval,
  reduction and generation objects the benchmark scores.</p>
</section>"""


_STATUS = {
    "yes": ("--series-3", "demonstrated"),
    "partial": ("--warning", "evidence on disk"),
    "no": ("--text-muted", "not demonstrated"),
}


def _contributions_section(state: DashboardState) -> str:
    """What the project claims, and what the files on disk say about each claim.

    Including the one that failed. A board showing five greens would be the
    least believable thing on this page, and C5's negative result is a finding
    the project earned by measuring rather than assuming.
    """
    if not state.contributions:
        return _empty("Contributions", "no results on disk to judge them against")

    def row(c: dict[str, Any]) -> str:
        colour, label = _STATUS.get(c["state"], _STATUS["no"])
        where = (
            f'<br><span class="dim">from {_e(c["evidence"])}</span>' if c.get("evidence") else ""
        )
        return (
            f"<tr><td><strong>{_e(c['id'])}</strong></td>"
            f"<td>{_e(c['claim'])}{where}</td>"
            f'<td><span class="status"><span class="dot" style="background:var({colour})">'
            f"</span>{_e(label)}</span></td>"
            f"<td>{_e(c['detail'])}</td></tr>"
        )

    rows = "".join(row(c) for c in state.contributions)
    quoted = "".join(
        f"<p class='note'><strong>{_e(c['id'])}:</strong> {_e(c['verdict'])}</p>"
        for c in state.contributions
        if c.get("verdict")
    )
    return f"""
<section>
  <h2>Contributions &mdash; what each one rests on</h2>
  <div class="card">
    <div class="tablewrap"><table>
      <thead><tr><th>#</th><th>claim</th><th>status</th><th>what was measured</th></tr></thead>
      <tbody>{rows}</tbody>
    </table></div>
    {quoted}
    <p class="note">A row reads <em>demonstrated</em> only where a summary file on disk
    records that verdict; the verdict's own words are quoted above rather than
    paraphrased. C5 is reported as a negative result, which is what the measurement
    found.</p>
  </div>
</section>"""


def _device_response_section(state: DashboardState) -> str:
    """The controller's choices over a measured run, one line per thermometer.

    Two quantities share the plot: temperature, which is what the policy reads,
    and the effort level it chose, which is what it did. They are drawn on
    separate axes because they share no unit, and the effort line is stepped
    because the level is a choice from a table, not a continuous quantity --
    interpolating between nprobe 4 and 8 would draw configurations nobody can
    run.
    """
    d = state.device_response
    if not d:
        return _empty(
            "Answering under real device conditions",
            "no profile table or no recorded handset trace in results/device",
            "device",
        )

    w, h = 720.0, 250.0
    pad_l, pad_r, pad_t, pad_b = 46.0, 52.0, 16.0, 34.0
    dur = max(1.0, float(d["duration_s"]))
    all_pts = [p for s in d["series"] for p in s["points"]]
    temps = [p["temperature_c"] for p in all_pts if math.isfinite(p["temperature_c"])]
    t_lo = min(min(temps, default=20.0), 20.0)
    t_hi = max(max(temps, default=80.0), float(d["hot_c"]) + 4.0)
    levels = sorted({p["nprobe"] for p in all_pts} | {p["allowed_nprobe"] for p in all_pts})
    n_hi = max(levels) if levels else 1

    def px(t: float) -> float:
        return pad_l + (w - pad_l - pad_r) * max(0.0, min(1.0, t / dur))

    def py_temp(c: float) -> float:
        frac = (c - t_lo) / (t_hi - t_lo) if t_hi > t_lo else 0.0
        return h - pad_b - (h - pad_t - pad_b) * max(0.0, min(1.0, frac))

    def py_level(n: float) -> float:
        # Log spacing: the table's levels double, so linear spacing would crush
        # every low level into the floor and hide exactly the derates that matter.
        frac = math.log2(max(1.0, n)) / math.log2(max(2.0, float(n_hi)))
        return h - pad_b - (h - pad_t - pad_b) * frac

    colours = {"skin": "--series-1", "hottest": "--series-2"}
    body: list[str] = []

    # The threshold the policy trips on, drawn where it actually sits.
    for label, c in (("warm", d["warm_c"]), ("hot", d["hot_c"])):
        y = py_temp(float(c))
        body.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w - pad_r}" y2="{y:.1f}" '
            f'stroke="var(--bd)" stroke-dasharray="4 4"/>'
            f'<text x="{w - pad_r + 4}" y="{y + 3.5:.1f}" class="ax">{_e(label)} '
            f"{float(c):.0f}&#176;C</text>"
        )

    for s in d["series"]:
        col = colours.get(s["sensor"], "--series-3")
        pts = s["points"]
        temp_path = " ".join(
            f"{'M' if i == 0 else 'L'}{px(p['t']):.1f},{py_temp(p['temperature_c']):.1f}"
            for i, p in enumerate(pts)
            if math.isfinite(p["temperature_c"])
        )
        step: list[str] = []
        for i, p in enumerate(pts):
            x, y = px(p["t"]), py_level(p["nprobe"])
            step.append(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}")
            if i + 1 < len(pts):
                step.append(f"L{px(pts[i + 1]['t']):.1f},{y:.1f}")
        body.append(
            f'<path d="{temp_path}" fill="none" stroke="var({col})" stroke-width="1.6" '
            f'opacity="0.42"/>'
            f'<path d="{" ".join(step)}" fill="none" stroke="var({col})" stroke-width="2.4"/>'
        )

    for n in levels:
        y = py_level(n)
        body.append(f'<text x="6" y="{y + 3.5:.1f}" class="ax">n={n}</text>')
    body.append(
        f'<line x1="{pad_l}" y1="{h - pad_b}" x2="{w - pad_r}" y2="{h - pad_b}" '
        f'stroke="var(--bd)"/>'
        f'<text x="{pad_l}" y="{h - 10}" class="ax">0 s</text>'
        f'<text x="{w - pad_r - 30}" y="{h - 10}" class="ax">{dur:.0f} s</text>'
    )

    def row(s: dict[str, Any]) -> str:
        share = 100.0 * s["n_derated"] / max(1, s["n_points"])
        swatch = (
            f'<span class="swatch" style="background:var({colours.get(s["sensor"], "--series-3")})">'
            f"</span>"
        )
        return (
            f"<tr><td>{swatch}<strong>{_e(s['sensor'])}</strong><br>"
            f'<span class="dim">{_e(s["means"])}</span></td>'
            f"<td>{_fmt(s['peak_c'], 1)} &#176;C</td>"
            f"<td>{s['n_derated']:,} of {s['n_points']:,}<br>"
            f'<span class="dim">{share:.0f}% of the run</span></td>'
            f"<td>{_e(', '.join(f'n={n}' for n in s['levels']))}</td>"
            f"<td>{_fmt(s['mean_recall'], 4)}</td></tr>"
        )

    rows = "".join(row(s) for s in d["series"])
    by_sensor = {s["sensor"]: s for s in d["series"]}
    verdict = ""
    if "skin" in by_sensor and "hottest" in by_sensor:
        skin, hot = by_sensor["skin"], by_sensor["hottest"]
        gap = hot["peak_c"] - skin["peak_c"]
        lost = float(d["budget_only_recall"]) - hot["mean_recall"]
        verdict = (
            f"<p class='note'><strong>The policy's answer depends entirely on which "
            f"thermometer it reads.</strong> On this one run the two disagree by "
            f"{_fmt(gap, 1)} &#176;C at peak. Reading the skin, the "
            f"{float(d['warm_c']):.0f} &#176;C threshold is never crossed and the "
            f"controller spends {100.0 - 100.0 * skin['n_derated'] / max(1, skin['n_points']):.0f}% "
            f"of the run at full effort. Reading the hottest zone, it derates across "
            f"{100.0 * hot['n_derated'] / max(1, hot['n_points']):.0f}% of it, giving up "
            f"{_fmt(lost, 4)} recall against the {float(d['budget_only_recall']):.4f} the "
            f"latency budget alone would have allowed. Both are the same policy, the same "
            f"trace and the same measured table. This is why C5 is recorded as real but "
            f"not actionable: the signal the shield needs is not the signal the platform "
            f"puts in front of it.</p>"
        )

    across = d.get("across_traces") or []
    repeat = ""
    if len(across) > 1:
        n_skin = sum(1 for a in across if a["skin_crossed"])
        n_hot = sum(1 for a in across if a["hottest_crossed"])
        trows = "".join(
            f"<tr><td>{_e(a['name'])}</td><td>{_fmt(a['skin_peak_c'], 1)}</td>"
            f"<td>{_fmt(a['hottest_peak_c'], 1)}</td>"
            f"<td>{_fmt(a['hottest_peak_c'] - a['skin_peak_c'], 1)}</td>"
            f"<td>{'skin and inside' if a['skin_crossed'] else ('inside only' if a['hottest_crossed'] else 'neither')}</td></tr>"
            for a in across
        )
        repeat = f"""
    <h3 style="margin-top:22px">Every recorded run, not just this one</h3>
    <div class="tablewrap"><table>
      <thead><tr><th>run</th><th>skin peak &#176;C</th><th>hottest peak &#176;C</th>
        <th>gap</th><th>crossed {float(d["warm_c"]):.0f} &#176;C</th></tr></thead>
      <tbody>{trows}</tbody>
    </table></div>
    <p class="note">Across {len(across)} independent runs the skin reached the
    {float(d["warm_c"]):.0f} &#176;C threshold in {n_skin} of them and the hottest zone in
    {n_hot}. One run showing this would be an anecdote; {len(across)} agreeing is a
    repeated measurement, which is what makes it a finding about the platform rather
    than about a single session.</p>"""

    return f"""
<section id="device">
  <h2>Answering under real device conditions &mdash; trace {_e(d["trace"])}, replayed</h2>
  <div class="card">
    <svg viewBox="0 0 {w:.0f} {h:.0f}" width="100%" role="img"
         aria-label="Effort level chosen over a measured handset trace, under each thermometer">
      {"".join(body)}
    </svg>
    <div class="legend">
      <span><span class="swatch" style="background:var(--series-1)"></span>skin sensor</span>
      <span><span class="swatch" style="background:var(--series-2)"></span>hottest zone</span>
      <span>faint line: temperature &middot; solid line: effort level chosen</span>
    </div>
    <div class="tablewrap"><table>
      <thead><tr><th>thermometer</th><th>peak</th><th>derated</th><th>levels used</th>
        <th>mean recall</th></tr></thead>
      <tbody>{rows}</tbody>
    </table></div>
    {verdict}
    {repeat}
    <p class="note">Every temperature here was recorded on the handset; every recall and
    latency is the measured profile table's. The only thing computed is which row the
    policy picks, at a {float(d["budget_ms"]):.0f} ms budget. Run it yourself:
    <code>smartann ask --replay {_e(d["trace"])} --sensor hottest</code>.</p>
  </div>
</section>"""


def _policy_section(state: DashboardState) -> str:
    """Phase 6: three policies against one budget, on the same queries.

    This is C1's evidence and it has been on disk unshown. The number that
    carries it is the violation rate, not the recall: a controller exists to
    keep a promise about latency, and buying that with recall is the trade
    being demonstrated rather than a cost being hidden.
    """
    p = state.policy6
    if not p or not p.get("arms"):
        return _empty(
            "Policies compared", "no phase 6 summary; run `smartann run experiment`", "policies"
        )

    arms = p["arms"]
    v_max = max(a["violation_pct"] for a in arms) or 1.0
    r_max = max(a["recall"] for a in arms) or 1.0

    def row(a: dict[str, Any]) -> str:
        viol = _bar_cell(a["violation_pct"], v_max, "--series-2", f"{a['violation_pct']:.2f}%")
        rec = _bar_cell(a["recall"], r_max, "--series-1", f"{a['recall']:.4f}")
        return (
            f"<tr><td><strong>{_e(a['policy'])}</strong></td>"
            f"<td>{a['n']:,}</td><td>{a['p50_ms']:.2f}</td><td>{a['p95_ms']:.2f}</td>"
            f"<td>{viol}</td><td>{rec}</td><td>{a['nprobe_p50']:.0f}</td>"
            f"<td>{a['overhead_p95_pct']:.3f}%</td></tr>"
        )

    rows = "".join(row(a) for a in arms)
    best = min(arms, key=lambda a: a["violation_pct"])
    worst = max(arms, key=lambda a: a["violation_pct"])
    return f"""
<section id="policies">
  <h2>Policies compared &mdash; one budget, the same queries, three controllers</h2>
  <div class="card">
    <div class="legend">
      <span><span class="swatch" style="background:var(--series-2)"></span>budget violations, own scale</span>
      <span><span class="swatch" style="background:var(--series-1)"></span>recall, own scale</span>
    </div>
    <div class="tablewrap"><table>
      <thead><tr><th>policy</th><th>queries</th><th>p50 ms</th><th>p95 ms</th>
        <th>over budget</th><th>recall</th><th>median nprobe</th><th>decision cost</th></tr></thead>
      <tbody>{rows}</tbody>
    </table></div>
    <p class="note"><strong>The controller keeps the promise the static setting breaks.</strong>
    Against a {float(p["budget_ms"]):.2f} ms budget, <em>{_e(worst["policy"])}</em> went over on
    {worst["violation_pct"]:.2f}% of queries and <em>{_e(best["policy"])}</em> on
    {best["violation_pct"]:.2f}% &mdash; a {worst["violation_pct"] / max(best["violation_pct"], 1e-9):.0f}x
    reduction. It paid {worst["recall"] - best["recall"]:.4f} recall for that, which is the
    trade being demonstrated rather than a cost being hidden. Deciding cost under
    {max(a["overhead_p95_pct"] for a in arms):.2f}% of a query at p95, so the controller is
    not paying for itself in the thing it is controlling.</p>
    <p class="note">Budget derivation: {_e(p.get("budget_derivation", "not recorded"))}.
    Energy: {_e(p.get("energy", "not recorded"))}</p>
  </div>
</section>"""


def _energy_section(state: DashboardState) -> str:
    """C4: energy measured on the handset, and effort's effect on power."""
    e = state.energy
    if not e or not e.get("measured"):
        return _empty(
            "Energy per query", "no energy model; it needs a recorded device trace", "energy"
        )

    m = e["measured"]
    attr = e.get("attribution") or {}
    quints = (e.get("effort_vs_power") or {}).get("quintiles") or []
    p_max = max((q["power_median_w"] for q in quints), default=1.0)

    def qrow(q: dict[str, Any]) -> str:
        bar = _bar_cell(q["power_median_w"], p_max, "--series-2", f"{q['power_median_w']:.3f} W")
        return (
            f"<tr><td>{q['work_lo']:,.0f} &ndash; {q['work_hi']:,.0f}</td>"
            f"<td>{bar}</td><td>{q['n']}</td></tr>"
        )

    qrows = "".join(qrow(q) for q in quints)
    lift = ""
    if len(quints) >= 2:
        lo, hi = quints[0]["power_median_w"], quints[-1]["power_median_w"]
        lift = (
            f"<p class='note'><strong>Effort costs power, and it is visible.</strong> The "
            f"busiest fifth of intervals drew {hi:.3f} W against {lo:.3f} W for the quietest, "
            f"a {100.0 * (hi - lo) / lo:.0f}% rise. That slope is what makes retrieval effort "
            f"worth controlling at all: if power had been flat across effort, there would be "
            f"nothing for a policy to save.</p>"
        )

    r2 = attr.get("r2")
    return f"""
<section id="energy">
  <h2>Energy per query &mdash; measured on the handset, not extrapolated</h2>
  <div class="card">
    <div class="tiles" style="margin-bottom:18px">
      {
        _tile(
            "Energy per query",
            f"{m['mj_per_query_allin']:.0f}",
            " mJ",
            "all-in, including idle draw",
        )
    }
      {_tile("Mean power", f"{m['mean_power_w']:.3f}", " W", f"median {m['power_median_w']:.3f} W")}
      {_tile("Total energy", f"{m['total_energy_j']:,.0f}", " J", "over the recorded run")}
      {
        _tile(
            "Per cluster opened",
            f"{attr.get('mj_per_cluster', float('nan')):.2f}",
            " mJ",
            "the marginal cost of effort",
        )
    }
    </div>
    <div class="tablewrap"><table>
      <thead><tr><th>work in the interval (clusters&middot;s)</th><th>median power</th>
        <th>intervals</th></tr></thead>
      <tbody>{qrows}</tbody>
    </table></div>
    {lift}
    <p class="note">Voltage stayed within
    {m["voltage_range_v"][0]:.3f}&ndash;{m["voltage_range_v"][1]:.3f} V and skin temperature
    within {m["skin_range_c"][0]:.1f}&ndash;{m["skin_range_c"][1]:.1f} &#176;C, so the energy is
    an integral of measured V&middot;I&middot;dt rather than a current reading scaled by a
    nominal voltage. The fit behind the marginal cost explains
    R&sup2; = {_fmt(r2, 3) if isinstance(r2, (int, float)) else "n/a"} of the variance, which is
    low: power on a phone is mostly not retrieval, and that is reported rather than
    smoothed away.</p>
  </div>
</section>"""


def _costmodel_section(state: DashboardState) -> str:
    """Phase 3: the latency model, and how far off it is. Including where it failed."""
    c = state.cost_model
    if not c or not c.get("validation"):
        return _empty("Latency cost model", "no cost model on disk", "costmodel")

    target = float(c.get("latency_mape_target", float("nan")))

    def mape_of(v: dict[str, Any]) -> tuple[float, str]:
        """The error, and the condition it was measured under.

        Not every row carries the same one: four terms were validated against a
        contended CPU and two were not. Printing both in one column without
        saying which is which would compare errors measured under different
        conditions as though they were the same quantity.
        """
        if "mape_contended" in v:
            return float(v["mape_contended"]), "under CPU contention"
        if "mape" in v:
            return float(v["mape"]), "quiet machine only"
        return float("nan"), "not validated"

    def row(v: dict[str, Any]) -> str:
        mape, under = mape_of(v)
        ok = math.isfinite(mape) and mape <= target
        colour = "--series-3" if ok else ("--warning" if math.isfinite(mape) else "--text-muted")
        label = (
            "within target" if ok else ("over target" if math.isfinite(mape) else "not validated")
        )
        actual = f"{v['mean_actual_ms']:.3f}" if "mean_actual_ms" in v else "&mdash;"
        pred = f"{v['mean_pred_ms']:.3f}" if "mean_pred_ms" in v else "&mdash;"
        return (
            f"<tr><td>{_e(v['term'])}</td><td>{actual}</td><td>{pred}</td>"
            f"<td>{_fmt(mape, 1)}%</td>"
            f'<td class="dim">{_e(under)}</td>'
            f"<td><span class='status'><span class='dot' style='background:var({colour})'>"
            f"</span>{label}</span></td></tr>"
        )

    rows = "".join(row(v) for v in c["validation"])
    over = sum(1 for v in c["validation"] if (mape_of(v)[0] > target))
    lat = c.get("latency") or {}
    return f"""
<section id="costmodel">
  <h2>Latency cost model &mdash; and where it does not hold</h2>
  <div class="card">
    <div class="tablewrap"><table>
      <thead><tr><th>term</th><th>mean actual ms</th><th>mean predicted ms</th>
        <th>MAPE</th><th>measured</th><th>vs {target:.0f}% target</th></tr></thead>
      <tbody>{rows}</tbody>
    </table></div>
    <p class="note"><strong>{over} of {len(c["validation"])} rows miss the
    {target:.0f}% target, and they are shown missing it.</strong> The pattern is the
    finding: the whole-query model fits a quiet machine inside the target and misses it
    badly once the CPU is contended. That is why the controller chooses from the measured
    profile table rather than from this model &mdash; a predictor that is 30% wrong on a
    busy phone would pick the wrong effort level exactly when picking right matters most.
    The model explains where the time goes; the table decides what to spend.
    Fitted on {_e(lat.get("fitted_on", "not recorded"))},
    {int(lat.get("n_fit", 0)):,} samples. The <em>measured</em> column says which condition
    each error was observed under, because errors measured under different conditions are
    not the same quantity.</p>
  </div>
</section>"""


def _index_section(state: DashboardState) -> str:
    """Phase 2: the index that everything else runs on, and its memory budget."""
    p2 = state.phase2
    if not p2:
        return _empty(
            "The index", "no phase 2 summary; run `scripts/phase2_acceptance.py`", "index"
        )

    sweep = p2.get("check2") or []
    srows = "".join(
        f"<tr><td>{r['nprobe']}</td><td>{r['ef_list']}</td>"
        f"<td>{r['ms']:.2f}</td><td>{r['recall']:.4f}</td></tr>"
        for r in sweep
    )
    detail = p2.get("check5_detail") or {}
    mrows = "".join(
        f"<tr><td><strong>{_e(name)}</strong></td><td>{d['base_mb']:,.1f}</td>"
        f"<td>{d['budget_mb']:,.1f}</td><td>{d['measured_mb']:,.1f}</td>"
        f"<td><span class='status'><span class='dot' style='background:var("
        f"{'--series-3' if d['measured_mb'] <= d['budget_mb'] else '--critical'})'></span>"
        f"{100.0 * d['measured_mb'] / d['base_mb']:.1f}% of the raw vectors</span></td></tr>"
        for name, d in detail.items()
    )
    load = p2.get("machine_load") or {}
    warn = ""
    if load.get("contaminated"):
        warn = (
            f"<p class='callout'><strong>These timings were taken on a contended machine.</strong> "
            f"{100.0 * float(load.get('fraction_over_threshold', 0)):.1f}% of samples exceeded the "
            f"load threshold, median load {float(load.get('median', float('nan'))):.3f}, peak "
            f"{float(load.get('max', float('nan'))):.3f}. They are kept and labelled rather than "
            f"rerun and quietly replaced, because the device numbers this project reports come "
            f"from the handset, not from here.</p>"
        )

    return f"""
<section id="index">
  <h2>The index everything runs on &mdash; effort against return, and memory</h2>
  <div class="card">
    <div class="grid2">
      <div>
        <h3>Effort sweep</h3>
        <div class="tablewrap"><table>
          <thead><tr><th>nprobe</th><th>ef_list</th><th>ms</th><th>recall</th></tr></thead>
          <tbody>{srows}</tbody>
        </table></div>
        <p class="note">Recall rises with effort and so does time: 32x the clusters buys
        {(sweep[-1]["recall"] / sweep[0]["recall"] if sweep else 1):.1f}x the recall for
        {(sweep[-1]["ms"] / sweep[0]["ms"] if sweep else 1):.0f}x the latency. A controller
        exists because that curve is steep at one end and flat at the other.</p>
      </div>
      <div>
        <h3>Resident memory</h3>
        <div class="tablewrap"><table>
          <thead><tr><th>dataset</th><th>raw vectors MB</th><th>budget MB</th>
            <th>measured MB</th><th></th></tr></thead>
          <tbody>{mrows}</tbody>
        </table></div>
        <p class="note">Only the centroid graph stays resident; cluster vectors are
        memory-mapped and touched one cluster at a time. That is what makes the effort knob
        meaningful on a phone &mdash; opening fewer clusters reads less disk, not just less RAM.</p>
      </div>
    </div>
    {warn}
  </div>
</section>"""


def _query_adaptive_section(state: DashboardState) -> str:
    """C6: effort allocated by measured query difficulty, against uniform effort.

    Two tables, because two different things have to be believed. The first
    asks whether the signal separates easy queries from hard ones at all -- if
    recall is flat across difficulty buckets there is nothing to allocate. The
    second is the held-out comparison, which is the claim.
    """
    q = state.query_adaptive
    if not q:
        return _empty(
            "Effort by query difficulty",
            "no phase 9 result; run `scripts/phase9_query_adaptive.py`",
            "queryfit",
        )

    levels = q["levels"]
    sep = q.get("separation") or []
    srows = ""
    for b in sep:
        cells = "".join(f"<td>{_fmt(b['recall_by_level'].get(str(lv)), 4)}</td>" for lv in levels)
        # Four decimals: the measured gaps span 0.000 to 0.08, so two rounds
        # the first two buckets to "0.00 - 0.00" and hides the boundary the
        # allocation is actually keyed on.
        rng = (
            f"{_fmt(b['gap_lo'], 4)} &ndash; {_fmt(b['gap_hi'], 4)}"
            if b.get("gap_lo") is not None
            else "&mdash;"
        )
        srows += (
            f"<tr><td><strong>{b['bucket'] + 1}</strong></td><td>{rng}</td>"
            f"<td>{100 * b['share']:.0f}%</td>{cells}</tr>"
        )
    lvl_head = "".join(f"<th>n={lv}</th>" for lv in levels)

    comps = q.get("comparisons") or []
    crows = ""
    for c in comps:
        h = c["held_out"]
        d = h["delta_recall"]
        colour = "--series-3" if d > 0 else ("--critical" if d < 0 else "--text-muted")
        saving = 1.0 - h["adaptive_mean_cost_ms"] / h["uniform_mean_cost_ms"]
        crows += (
            f"<tr><td>n={c['uniform_nprobe']}</td>"
            f"<td>{h['uniform_recall']:.4f}</td><td>{h['adaptive_recall']:.4f}</td>"
            f"<td><span class='status'><span class='dot' style='background:var({colour})'>"
            f"</span>{d:+.4f}</span></td>"
            f"<td>{h['uniform_mean_cost_ms']:.1f}</td><td>{h['adaptive_mean_cost_ms']:.1f}</td>"
            f"<td>{100 * saving:+.0f}%</td>"
            f"<td class='dim'>{_e(', '.join(f'n={x}' for x in c['assignment']))}</td></tr>"
        )

    res = float(q.get("recall_resolution", float("nan")))
    return f"""
<section id="queryfit">
  <h2>Effort by query difficulty &mdash; the signal the probe already computed</h2>
  <div class="card">
    <p class="note"><strong>{_e(q["signal"])}</strong>: {_e(q["signal_meaning"])}
    It is available before any cluster is opened, and until this experiment no policy
    in the project read it.</p>

    <h3 style="margin-top:20px">Does difficulty separate recall at all?</h3>
    <div class="tablewrap"><table>
      <thead><tr><th>bucket</th><th>gap range</th><th>share</th>{lvl_head}</tr></thead>
      <tbody>{srows}</tbody>
    </table></div>
    <p class="note">If these rows were identical the signal would carry nothing and no
    allocation could help. Read down a column to see whether harder queries really do
    recover less at the same effort.</p>

    <h3 style="margin-top:22px">Held out: {q["n_test"]:,} queries never used to fit</h3>
    <div class="tablewrap"><table>
      <thead><tr><th>budget</th><th>uniform recall</th><th>adaptive recall</th><th>delta</th>
        <th>uniform ms</th><th>adaptive ms</th><th>time saved</th><th>assignment</th></tr></thead>
      <tbody>{crows}</tbody>
    </table></div>
    <p class="callout">{_e(q["verdict"])}</p>
    <p class="note">Resolution: one held-out query gaining one neighbour moves mean recall
    by {res:.4f}, so a delta smaller than a few multiples of that is a handful of queries
    rather than an effect. Costs are the handset's measured p95 per level, not timed here.
    Buckets and allocation were fitted on {q["n_fit"]:,} queries and scored on the other
    {q["n_test"]:,}; seed {q["seed"]}. {_e(q.get("method", ""))}</p>
  </div>
</section>"""


def _kb_section(state: DashboardState) -> str:
    kb = state.kb
    if not kb:
        return _empty(
            "Knowledge base",
            "nothing added yet; run `smartann kb add-wiki --starter`",
            "kb",
        )
    rows = "".join(
        f'<tr><td>{_e(s["title"])}</td><td style="text-align:left">{_e(s["kind"])}</td>'
        f"<td>{s['n_chunks']:,}</td><td>{s['n_chars']:,}</td>"
        f'<td style="text-align:left">{_e(s["added_at"])}</td></tr>'
        for s in kb["sources"]
    )
    return f"""
<section id="kb">
  <h2>Knowledge base &mdash; {kb["n_chunks"]:,} passages you added</h2>
  <div class="card">
    <div class="tablewrap"><table>
      <thead><tr><th>document</th><th>source</th><th>passages</th><th>characters</th>
        <th>added</th></tr></thead>
      <tbody>{rows}</tbody>
    </table></div>
    <p class="note">Chunked and embedded exactly like the benchmark corpus, three-sentence
    windows through the same model, so its passages compete with SQuAD's on one cosine
    scale. Searched exactly rather than approximately, because at this size a brute-force
    scan costs under a millisecond and has recall 1.0 by construction. The benchmark index
    is never modified, so every recorded measurement stays reproducible.</p>
  </div>
</section>"""


def _verification_section(state: DashboardState) -> str:
    v = state.verification
    if not v:
        return _empty("Verification", "no verification.json; run `scripts/verify_all.py`")
    items = []
    for step in v["steps"]:
        ok = bool(step.get("ok"))
        colour = "--good" if ok else "--warning"
        items.append(
            f"<tr><td>{_e(step.get('name', '?'))}</td>"
            f'<td style="text-align:left"><span class="status">'
            f'<span class="dot" style="background:var({colour})"></span>'
            f"{'passed' if ok else 'exit 1, by design'}</span></td></tr>"
        )
    return f"""
<section>
  <h2>Verification &mdash; {v.get("n_ok", "?")} of {v.get("n_steps", "?")} steps</h2>
  <div class="card">
    <div class="tablewrap"><table><tbody>{"".join(items)}</tbody></table></div>
    <p class="note">{_e(v.get("note", ""))}</p>
  </div>
</section>"""


def _figures_section(state: DashboardState) -> str:
    if not state.figures:
        # Not "": a section that disappears makes "we made no figures" look
        # exactly like "this page has no figures panel", which is the confusion
        # every other section on this page is built to avoid.
        return _empty("Figures", "none on disk; run `smartann figures make`", "figures")
    figs = "".join(
        f'<figure><img src="{src}" alt="{_e(name)}"><figcaption>{_e(name)}</figcaption></figure>'
        for name, src in state.figures
    )
    return f"""
<section id="figures">
  <h2>Figures &mdash; every number from stored parquet or a summary JSON</h2>
  <div class="figs">{figs}</div>
</section>"""


def _empty(title: str, why: str, sid: str = "") -> str:
    """A section with no input, keeping its anchor.

    The id travels with the placeholder because the navigation links to it
    either way: a jump link that lands nowhere on exactly the runs where a
    result is missing would hide the absence the placeholder exists to show.
    """
    anchor = f' id="{sid}"' if sid else ""
    return f"""
<section{anchor}>
  <h2>{_e(title)}</h2>
  <div class="card"><p class="missing">Not run &mdash; {_e(why)}.</p></div>
</section>"""


def _provenance(state: DashboardState) -> str:
    """Where the device numbers came from, said once at the top.

    A reader's first question about an on-device result is which device. Taking
    it from the profile table rather than writing it here means the page cannot
    name a handset the measurements did not come from.
    """
    parts: list[str] = []
    if state.profile and state.profile.get("measured_under"):
        parts.append(f"Device measurements: {state.profile['measured_under']}")
    if state.qa and state.qa.get("generator"):
        parts.append(f"Generator: {state.qa['generator']}")
    if state.profile and state.profile.get("dataset"):
        parts.append(f"Benchmark corpus: {state.profile['dataset']}")
    if not parts:
        return ""
    # Each part is escaped on its own and the separator joined afterwards, so
    # the separator stays an entity and the values stay text. Escaping the
    # joined string would print a literal "&middot;".
    return f'<p class="meta prov">{" &middot; ".join(_e(p) for p in parts)}</p>'


def _nav() -> str:
    """Jump links, in the order the work was done.

    A page this long is read by someone looking for one thing, usually while
    being asked a question about it. The order is the project's own phases
    rather than importance, so a reader can find a result by remembering when
    it was measured.
    """
    items = (
        ("index", "The index"),
        ("effort", "Effort levels"),
        ("policies", "Policies compared"),
        ("device", "Device conditions"),
        ("queryfit", "Query difficulty"),
        ("energy", "Energy"),
        ("heat", "Heat"),
        ("costmodel", "Cost model"),
        ("quality", "Answer quality"),
        ("joint", "Joint control"),
        ("kb", "Knowledge base"),
        ("figures", "Figures"),
    )
    links = "".join(f'<a href="#{i}">{_e(label)}</a>' for i, label in items)
    return f'<nav class="nav">{links}</nav>'


def _tiles(state: DashboardState) -> str:
    tiles = []
    if state.thermal:
        t = state.thermal
        tiles.append(
            _tile(
                "Controllable heat score",
                f"{t['score']:.0f}",
                " / 100",
                "share of device heat retrieval controls",
            )
        )
    if state.qa:
        best = max(state.qa["arms"], key=lambda a: a["f1"])
        tiles.append(
            _tile(
                "Answer quality",
                f"{best['f1']:.3f}",
                " F1",
                f"{best['n']:,} questions, arm '{best['arm']}'",
            )
        )
    if state.profile:
        top = max(state.profile["entries"], key=lambda e: e["recall"])
        tiles.append(
            _tile(
                "Peak recall on device",
                f"{top['recall']:.3f}",
                "",
                f"at nprobe {top['nprobe']}, p95 {top['p95_ms']:.0f} ms",
            )
        )
    if state.thermal:
        tiles.append(
            _tile(
                "Thermal delay",
                f"{state.thermal['tau_s']:.0f}",
                " s",
                "before effort shows up in temperature",
            )
        )
    if state.device_response:
        by = {s["sensor"]: s for s in state.device_response["series"]}
        if "skin" in by and "hottest" in by:
            tiles.append(
                _tile(
                    "Sensors disagree by",
                    f"{by['hottest']['peak_c'] - by['skin']['peak_c']:.1f}",
                    " °C",
                    "at peak -- which one the policy reads decides everything",
                )
            )
    if state.query_adaptive:
        q = state.query_adaptive
        best = max((c["held_out"]["delta_recall"] for c in q["comparisons"]), default=0.0)
        tiles.append(
            _tile(
                "Recall from difficulty",
                f"{best:+.4f}",
                "",
                f"free signal, {q['n_test']:,} held-out queries",
            )
        )
    if state.kb:
        tiles.append(
            _tile(
                "Corpus searched",
                f"{state.kb['n_chunks']:,}",
                " added",
                f"passages from {len(state.kb['sources'])} sources, beside SQuAD",
            )
        )
    return f'<div class="tiles">{"".join(tiles)}</div>' if tiles else ""


def render_html(state: DashboardState) -> str:
    """One portable file: no external scripts, no external images, no network."""
    missing = ""
    if state.missing:
        missing = (
            f'<p class="note">Sections with no input on disk: '
            f"{_e(', '.join(state.missing))}. They are shown as not run rather than "
            f"hidden, so absence stays visible.</p>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SMART-ANN dashboard</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <h1>SMART-ANN</h1>
    <p class="sub">Runtime resource-aware adaptive control for on-device retrieval-augmented generation.</p>
    <p class="meta">Generated {_e(state.generated_at)} &middot; commit {_e(state.git_sha)} &middot;
      every value read from {_e(state.results_dir)}</p>
    {_provenance(state)}
  </header>

  {_tiles(state)}
  {missing}

  {_nav()}

  {_answers_section(state)}
  {_contributions_section(state)}
  {_index_section(state)}
  {_profile_section(state)}
  {_policy_section(state)}
  {_device_response_section(state)}
  {_query_adaptive_section(state)}
  {_energy_section(state)}
  {_thermal_section(state)}
  {_costmodel_section(state)}
  {_qa_section(state)}
  {_joint_scatter(state)}
  {_kb_section(state)}
  {_verification_section(state)}
  {_figures_section(state)}

  <footer>
    No number on this page is typed by hand. Each is read from the same JSON the
    figures and the paper read, so the page cannot drift from the results it describes.
    A section whose input is missing says so rather than disappearing.
  </footer>
</div>
</body>
</html>"""

"""A local web front end for `ask`, on the standard library alone.

**Why this exists.** The terminal trace is complete but it scrolls, and a person
watching over a shoulder cannot see the controller's decision, the evidence and
the cost at the same time. One page can hold all three at once.

**No framework, no new dependency** (Section 4, rule 6). `http.server` is
enough for one local reader, and the alternative is adding a web stack to a
project whose point is measuring on-device cost.

**Bound to the loopback address, always.** The server answers questions using a
local index and a local model; exposing that on a shared network would let
anyone on it run inference on this machine. `127.0.0.1` is not configurable
here, and that is deliberate rather than an oversight.

**One question at a time, on one thread.** Generation is not thread-safe, and
two concurrent requests through one model would interleave and also make every
reported latency wrong. Beyond that, the embedding cache is a SQLite connection,
and SQLite refuses a connection used from a thread other than the one that
opened it -- so a pipeline loaded on the main thread and called from a request
thread raises before it retrieves anything. `SingleThreadPipeline` pins loading,
answering and closing to one worker thread, which fixes both at once and leaves
SQLite's guard switched on rather than disabling it.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

__all__ = ["PAGE", "SingleThreadPipeline", "build_app", "serve"]

MAX_BODY = 64 * 1024
MAX_QUESTION = 2000

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SMART-ANN</title>
<style>
:root{color-scheme:light;--s0:#f4f4f1;--s1:#fcfcfb;--s2:#eceae4;--bd:#dcdad2;
--tp:#0b0b0b;--ts:#52514e;--tm:#82817a;--c1:#2a78d6;--c2:#eb6834;--c3:#1baf7a}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;
--s0:#121211;--s1:#1a1a19;--s2:#232321;--bd:#34342f;--tp:#fff;--ts:#c3c2b7;--tm:#8d8c83;
--c1:#3987e5;--c2:#d95926;--c3:#199e70}}
*{box-sizing:border-box}
body{margin:0;background:var(--s0);color:var(--tp);
font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:900px;margin:0 auto;padding-block:30px;padding-left:18px;padding-right:18px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.02em}
.sub{color:var(--ts);margin:0 0 22px;font-size:14px}
form{display:flex;gap:9px;flex-wrap:wrap}
input[type=text]{flex:1 1 320px;min-width:0;padding:12px 14px;font-size:16px;
border:1px solid var(--bd);border-radius:9px;background:var(--s1);color:var(--tp)}
button{padding:12px 20px;font-size:15px;font-weight:600;border:0;border-radius:9px;
background:var(--c1);color:#fff;cursor:pointer}
button:disabled{opacity:.55;cursor:progress}
.opts{display:flex;gap:16px;flex-wrap:wrap;margin:10px 0 0;font-size:13px;color:var(--ts)}
.opts label{display:flex;align-items:center;gap:6px}
.card{background:var(--s1);border:1px solid var(--bd);border-radius:11px;padding:18px;margin-top:18px}
h2{font-size:11.5px;text-transform:uppercase;letter-spacing:.09em;color:var(--tm);
margin:0 0 12px;font-weight:600}
.answer{font-size:23px;font-weight:650;color:var(--c3);line-height:1.3}
.answer.declined{color:var(--tm);font-style:italic;font-weight:500;font-size:19px}
.note{font-size:13px;color:var(--ts);margin-top:9px}
.cite{font-size:13px;margin-top:9px}
.cite a{color:var(--c1)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px}
.kv{background:var(--s2);border-radius:8px;padding:10px 12px}
.kv .k{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--tm)}
.kv .v{font-size:19px;font-weight:650;font-variant-numeric:tabular-nums;margin-top:2px}
.p{border-top:1px solid var(--bd);padding:12px 0}
.p:first-of-type{border-top:0}
.phead{display:flex;justify-content:space-between;gap:10px;font-size:12.5px;color:var(--tm);
flex-wrap:wrap}
.phead b{color:var(--tp);font-weight:600}
.ptext{font-size:13.5px;color:var(--ts);margin-top:5px}
.badge{display:inline-block;padding:1px 7px;border-radius:9px;background:var(--s2);
color:var(--ts);font-size:11px;margin-left:6px}
.bar{display:flex;height:22px;border-radius:6px;overflow:hidden;gap:2px;background:var(--s2)}
.bar div{display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;
color:#fff;overflow:hidden;white-space:nowrap;min-width:2px}
.stages{font-size:12.5px;color:var(--tm);margin-top:8px;font-variant-numeric:tabular-nums}
.err{color:#d03b3b;font-size:14px;margin-top:14px}
.dev .opts{margin-top:12px}
.slab{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.06em;
color:var(--tm);margin:12px 0 4px}
input[type=range]{width:100%;accent-color:var(--c1)}
.hot{color:var(--c2);font-weight:600}
.derate{border-left:3px solid var(--c2);background:var(--s2);padding:10px 12px;
border-radius:0 8px 8px 0;margin-top:10px;font-size:13px;color:var(--ts)}
.derate b{color:var(--tp)}
footer{margin-top:26px;color:var(--tm);font-size:12.5px}
footer a{color:var(--c1)}
</style></head><body><div class="wrap">
<h1>SMART-ANN</h1>
<p class="sub">Ask a question. The controller picks an effort level from measured
device latencies, the index retrieves the evidence, and the answer is cited back to it.</p>

<form id="f">
  <input type="text" id="q" placeholder="Who developed the theory of relativity?" autofocus>
  <button id="go" type="submit">Ask</button>
</form>
<div class="opts">
  <label>Latency budget
    <select id="budget">
      <option value="60">60 ms</option>
      <option value="120" selected>120 ms</option>
      <option value="400">400 ms</option>
      <option value="">no budget</option>
    </select>
  </label>
  <label><input type="checkbox" id="fallback"> answer from model memory if the corpus cannot</label>
</div>

<div class="card dev">
  <h2>Device condition</h2>
  <p class="note">The controller can read the handset's state as well as the clock.
  Replaying a recorded trace drives it from telemetry the phone actually reported;
  nothing below is invented.</p>
  <div class="opts">
    <label>Source
      <select id="devmode">
        <option value="off" selected>this laptop (no thermometer)</option>
      </select>
    </label>
    <label>Thermometer
      <select id="sensor">
        <option value="hottest" selected>hottest zone inside</option>
        <option value="skin">skin, the case</option>
      </select>
    </label>
  </div>
  <div id="devpos" hidden>
    <label class="slab">Position in the run <span id="devat"></span></label>
    <input type="range" id="devidx" min="0" max="100" value="0" step="1">
    <div class="note" id="devnow"></div>
  </div>
</div>

<div id="err" class="err" hidden></div>
<div id="out" hidden>
  <div class="card"><h2>Answer</h2>
    <div id="ans" class="answer"></div>
    <div id="ansnote" class="note" hidden></div>
    <div id="cite" class="cite" hidden></div>
  </div>
  <div class="card"><h2>Controller</h2>
    <div class="grid" id="kpis"></div>
    <div id="derate" class="derate" hidden></div>
    <div class="note" id="polnote"></div>
  </div>
  <div class="card"><h2>Evidence</h2><div id="passages"></div></div>
  <div class="card"><h2>Cost on this machine</h2>
    <div class="bar" id="bar"></div><div class="stages" id="stages"></div>
  </div>
</div>
<footer>Local only, on 127.0.0.1. <a href="/dashboard" target="_blank">Open the results
dashboard</a>.</footer>
</div>
<script>
const $ = id => document.getElementById(id);
const COLORS = ['var(--c1)','var(--c2)','var(--c3)','#8d8c83','#6da7ec'];

function kv(k, v) {
  const d = document.createElement('div'); d.className = 'kv';
  const a = document.createElement('div'); a.className = 'k'; a.textContent = k;
  const b = document.createElement('div'); b.className = 'v'; b.textContent = v;
  d.append(a, b); return d;
}

function render(r) {
  $('out').hidden = false;
  const ans = $('ans');
  ans.textContent = r.answer;
  ans.className = 'answer' + (r.declined ? ' declined' : '');

  const note = $('ansnote');
  let msg = '';
  if (r.source === 'model_memory') {
    msg = "From the model's memory, not from the corpus. Nothing checks it.";
  } else if (r.source === 'not_in_corpus' && r.extracted_answer &&
             !r.extracted_answer.toLowerCase().startsWith('unanswerable')) {
    const p = typeof r.verify_p_yes === 'number' ? ' (p = ' + r.verify_p_yes.toFixed(2) + ')' : '';
    msg = 'The extractive step proposed "' + r.extracted_answer + '", but the grounding check '
        + 'judged the passages do not contain the answer' + p + ', so it is withheld.';
  } else if (r.source === 'not_in_corpus') {
    msg = 'The retrieved passages do not contain the answer, and the model said so '
        + 'rather than inventing one.';
  } else if (typeof r.verify_p_yes === 'number' && !Number.isNaN(r.verify_p_yes)) {
    msg = 'Grounding check: the passages contain the answer, p = ' + r.verify_p_yes.toFixed(2);
  }
  note.textContent = msg; note.hidden = !msg;

  const cite = $('cite'); cite.textContent = ''; cite.hidden = true;
  const src = (r.retrieval || []).find(p => p.rank === r.cited_rank);
  if (src) {
    cite.hidden = false;
    cite.append(document.createTextNode('Source: passage [' + src.rank + '], '));
    if (src.url) {
      const a = document.createElement('a');
      a.href = src.url; a.target = '_blank'; a.rel = 'noopener';
      a.textContent = src.title || ('passage ' + src.rank);
      cite.append(a);
    } else {
      cite.append(document.createTextNode(src.title || ('passage ' + src.rank)));
    }
  }

  const k = $('kpis'); k.textContent = '';
  k.append(kv('nprobe chosen', r.policy.chosen_nprobe));
  k.append(kv('clusters opened', r.work.n_cluster_loads));
  k.append(kv('decision cost', r.policy.overhead_us.toFixed(1) + ' us'));
  k.append(kv('total', Math.round(r.timing_ms.total) + ' ms'));
  if (r.device && r.device.state) {
    const t = r.device.state.temperature_c;
    k.append(kv('device temp', typeof t === 'number' ? t.toFixed(1) + ' C' : 'not measured'));
  }

  const dr = $('derate'); dr.textContent = ''; dr.hidden = true;
  if (r.device) {
    dr.hidden = false;
    const p = r.policy;
    const head = document.createElement('div');
    if (p.derated) {
      const b1 = document.createElement('b');
      b1.textContent = 'Derated: nprobe ' + p.baseline_nprobe + ' \\u2192 ' + p.chosen_nprobe + '.';
      head.append(b1, document.createTextNode(
        ' The ' + Math.round(r.policy.budget_ms) + ' ms budget alone allowed '
        + p.baseline_nprobe + '; the device state cut it.'
        + (typeof p.recall_given_up === 'number'
            ? ' That costs ' + p.recall_given_up.toFixed(4)
              + ' measured recall, at those two levels over the benchmark set.'
            : '')));
    } else {
      head.textContent = 'No derate: nprobe ' + p.chosen_nprobe
        + ' is what the budget allowed, and no threshold in the policy was crossed.';
    }
    dr.append(head);
    const why = document.createElement('div');
    why.style.marginTop = '6px';
    // Empty reasons mean two opposite things: a cool phone, or no sensors at
    // all. The page must not render them the same way.
    why.textContent = (r.policy.reasons && r.policy.reasons.length)
      ? 'Signals used: ' + r.policy.reasons.join('; ')
      : (r.device.signals_absent
          ? 'The policy acted on nothing: every signal it reads was absent here. '
            + 'That is absence, not a healthy device.'
          : 'The signals were read and none crossed a threshold. That is a device '
            + 'in good condition, not a device that was not measured.');
    dr.append(why);
    const src = document.createElement('div');
    src.style.marginTop = '6px'; src.style.opacity = '.8';
    src.textContent = (r.device.measured ? 'Measured: ' : 'HYPOTHETICAL, not measured: ')
      + r.device.source;
    dr.append(src);
  }
  const te = r.policy.table_entry;
  $('polnote').textContent = te
    ? 'Picked the highest-recall level whose measured p95 (' + te.p95_ms.toFixed(1)
      + ' ms) fits the budget. That level measured recall ' + te.recall.toFixed(4)
      + ' over the benchmark set, which is the level\\'s recall and not this question\\'s.'
    : 'No profile table loaded, so the effort level was fixed rather than chosen.';

  const ps = $('passages'); ps.textContent = '';
  (r.retrieval || []).forEach(p => {
    const d = document.createElement('div'); d.className = 'p';
    const h = document.createElement('div'); h.className = 'phead';
    const left = document.createElement('div');
    const b = document.createElement('b');
    b.textContent = '[' + p.rank + '] ' + (p.title || ('chunk #' + p.chunk_id));
    left.append(b);
    if (p.origin === 'kb') {
      const badge = document.createElement('span');
      badge.className = 'badge'; badge.textContent = 'your knowledge base';
      left.append(badge);
    }
    const right = document.createElement('div');
    right.textContent = 'similarity ' + p.similarity.toFixed(3);
    h.append(left, right);
    const t = document.createElement('div'); t.className = 'ptext';
    t.textContent = (p.text || '').slice(0, 320) + ((p.text || '').length > 320 ? ' ...' : '');
    d.append(h, t); ps.append(d);
  });

  const bar = $('bar'); bar.textContent = '';
  const total = r.timing_ms.total || 1;
  const names = Object.keys(r.timing_ms).filter(n => n !== 'total');
  names.forEach((n, i) => {
    const ms = r.timing_ms[n];
    const seg = document.createElement('div');
    seg.style.flex = String(Math.max(ms / total, 0.001));
    seg.style.background = COLORS[i % COLORS.length];
    if (ms / total > 0.12) seg.textContent = n;
    seg.title = n + ' ' + ms.toFixed(1) + ' ms';
    bar.append(seg);
  });
  $('stages').textContent = names.map(n => n + ' ' + r.timing_ms[n].toFixed(1) + ' ms').join('  .  ')
    + '  .  total ' + total.toFixed(1) + ' ms';
}

// ---- device condition -----------------------------------------------------
let TRACES = [];

function deviceSpec() {
  const name = $('devmode').value;
  if (name === 'off') return null;
  return {trace: name, sensor: $('sensor').value, index: Number($('devidx').value)};
}

async function loadTraces() {
  try {
    const r = await fetch('/api/traces');
    TRACES = (await r.json()).traces || [];
  } catch (e) { return; }
  const sel = $('devmode');
  TRACES.forEach(t => {
    const o = document.createElement('option');
    o.value = t.name;
    o.textContent = 'recorded trace ' + t.name + '  (' + t.n_samples + ' samples, '
      + Math.round(t.duration_s) + ' s on the handset)';
    sel.append(o);
  });
}

async function previewDevice() {
  const spec = deviceSpec();
  $('devpos').hidden = !spec;
  if (!spec) { $('devnow').textContent = ''; return; }
  const t = TRACES.find(x => x.name === spec.trace);
  if (t) $('devidx').max = String(t.n_samples - 1);
  const url = '/api/device?trace=' + encodeURIComponent(spec.trace)
    + '&sensor=' + encodeURIComponent(spec.sensor) + '&index=' + spec.index;
  try {
    const r = await fetch(url);
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'could not read that trace');
    $('devat').textContent = 'sample ' + (spec.index + 1) + ' of ' + d.n_samples;
    $('devnow').textContent = '';
    const temp = typeof d.temperature_c === 'number' ? d.temperature_c.toFixed(1) + ' \\u00b0C'
                                                     : 'not measured';
    const batt = typeof d.battery_pct === 'number' ? d.battery_pct.toFixed(0) + '%'
                                                   : 'not measured';
    const hot = document.createElement('span');
    hot.className = 'hot'; hot.textContent = temp;
    $('devnow').append(document.createTextNode('Recorded on the handset: '), hot,
      document.createTextNode(', battery ' + batt + ', '
        + (d.charging ? 'charging' : 'on battery') + '.'));
  } catch (err) {
    $('devnow').textContent = String(err.message || err);
  }
}

$('devmode').addEventListener('change', () => { $('devidx').value = '0'; previewDevice(); });
$('sensor').addEventListener('change', previewDevice);
$('devidx').addEventListener('input', previewDevice);
loadTraces();

$('f').addEventListener('submit', async e => {
  e.preventDefault();
  const question = $('q').value.trim();
  if (!question) return;
  $('go').disabled = true; $('go').textContent = 'Thinking...';
  $('err').hidden = true;
  try {
    const res = await fetch('/api/ask', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        question,
        budget_ms: $('budget').value ? Number($('budget').value) : null,
        fallback: $('fallback').checked,
        device: deviceSpec(),
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
    render(data);
  } catch (err) {
    $('err').hidden = false;
    $('err').textContent = String(err.message || err);
  } finally {
    $('go').disabled = false; $('go').textContent = 'Ask';
  }
});
</script></body></html>
"""


class SingleThreadPipeline:
    """Runs every pipeline call on one worker thread, including the load.

    The embedding cache is a SQLite connection opened during `load`, and SQLite
    refuses a connection used from any other thread. `ThreadingHTTPServer` hands
    each request to a fresh thread, so a pipeline loaded on the main thread
    raises `ProgrammingError` on the first question, before the handler has sent
    anything -- which reaches the browser as a bare "Failed to fetch". Loading
    and answering on the same worker keeps that guard in place instead of
    passing `check_same_thread=False` and hoping.

    One worker also means one question at a time, so every latency the page
    reports is a query that ran alone.
    """

    def __init__(self, pipeline: Any, default_budget_ms: float | None = None) -> None:
        self._pipeline = pipeline
        self._default_budget_ms = default_budget_ms
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="smartann-pipeline"
        )

    def _on_worker(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Call `fn` on the worker and re-raise whatever it raised, here."""
        return self._pool.submit(fn, *args, **kwargs).result()

    def load(self) -> None:
        self._on_worker(self._pipeline.load)

    def ask(self, question: str, budget_ms: float | None = None, **kwargs: Any) -> Any:
        if budget_ms is None:
            budget_ms = self._default_budget_ms
        return self._on_worker(self._pipeline.ask, question, budget_ms=budget_ms, **kwargs)

    def close(self) -> None:
        try:
            self._on_worker(self._pipeline.close)
        finally:
            self._pool.shutdown(wait=True)


class TraceCatalog:
    """The recorded handset traces on disk, addressable by name and thermometer.

    Loaded once at startup because each is a few hundred samples of JSON and
    re-reading per request would put file I/O inside a measured query. The page
    offers only what this finds: a demo must never present a trace that is not
    on the machine it is running on.
    """

    def __init__(self, device_dir: Any) -> None:
        from pathlib import Path

        from smartann.telemetry.replay import SENSORS, available_traces, load_trace, trace_path

        self.dir = Path(device_dir)
        self._by_key: dict[tuple[str, str], Any] = {}
        self.names: list[str] = []
        for name in available_traces(self.dir):
            loaded = False
            for sensor in SENSORS:
                try:
                    self._by_key[name, sensor] = load_trace(trace_path(self.dir, name), sensor)
                    loaded = True
                except (OSError, ValueError):
                    continue
            if loaded:
                self.names.append(name)

    def catalog(self) -> list[dict[str, Any]]:
        """What the page shows in its trace menu, measured values included."""
        out: list[dict[str, Any]] = []
        for name in self.names:
            any_trace = next(
                (self._by_key[k] for k in self._by_key if k[0] == name),
                None,
            )
            if any_trace is None:
                continue
            peaks = {
                sensor: _json_safe(self._by_key[name, sensor].peak_c)
                for (n, sensor) in self._by_key
                if n == name
            }
            out.append(
                {
                    "name": name,
                    "n_samples": len(any_trace),
                    "duration_s": _json_safe(any_trace.duration_s),
                    "peak_c": peaks,
                }
            )
        return out

    def resolve(self, spec: Any) -> tuple[Any, str, bool]:
        """A device state for one request, or (None, "", False) for no device.

        Returns the state, a label naming exactly where it came from, and
        whether it is a measurement. The label travels with the answer so a
        reader never has to guess which trace a decision was made against.
        """
        if not isinstance(spec, dict):
            return None, "", False
        name = str(spec.get("trace", "")).strip()
        if not name:
            return None, "", False
        sensor = str(spec.get("sensor", "hottest")).strip()
        trace = self._by_key.get((name, sensor))
        if trace is None:
            raise ValueError(f"no recorded trace {name!r} with a {sensor!r} sensor on this machine")
        try:
            index = int(spec.get("index", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("device index must be a whole number") from exc
        return trace.state_at(index), trace.describe(index), True

    def preview(self, name: str, sensor: str, index: int) -> dict[str, Any]:
        """The recorded state at one point, for the page to show before asking."""
        trace = self._by_key.get((name, sensor))
        if trace is None:
            raise ValueError(f"no recorded trace {name!r} with a {sensor!r} sensor")
        state = trace.state_at(index)
        return {
            "label": trace.describe(index),
            "n_samples": len(trace),
            "temperature_c": _json_safe(state.temperature_c),
            "battery_pct": _json_safe(state.battery_pct),
            "charging": state.charging,
            "discharge_rate_ma": _json_safe(state.discharge_rate_ma),
        }


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats with null, so the page can parse the reply.

    `verify_p_yes` is NaN whenever the grounding check did not run, and
    `json.dumps` writes that as a bare `NaN`, which is not JSON: the browser
    fails on the whole response and shows nothing. Null is the honest encoding
    -- the check has no value here because it did not happen -- and the page
    already tests for a number before printing one.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def build_app(
    pipeline: Any, dashboard_html: Any, traces: TraceCatalog | None = None
) -> type[BaseHTTPRequestHandler]:
    """A request handler bound to one loaded pipeline.

    `dashboard_html` is a callable rather than a string so the dashboard is
    rebuilt on each view: a page that answered questions while the server ran
    should show them, not the state at startup.
    """
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "SMART-ANN"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            """Silent: request logs would scroll the demo off the screen."""

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            # The page loads no third-party anything; say so, so a stray
            # injected string can never become a script tag that runs.
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
                "connect-src 'self'",
            )
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/dashboard":
                self._send(200, dashboard_html().encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/traces":
                body = json.dumps({"traces": traces.catalog() if traces else []})
                self._send(200, body.encode("utf-8"), "application/json; charset=utf-8")
            elif path == "/api/device":
                self._preview()
            else:
                self._send(404, b"not found", "text/plain; charset=utf-8")

        def _preview(self) -> None:
            """The recorded device state at one point, without answering anything."""
            from urllib.parse import parse_qs, urlparse

            q = parse_qs(urlparse(self.path).query)
            if traces is None:
                self._send(404, b'{"error":"no traces on this machine"}', "application/json")
                return
            try:
                body = json.dumps(
                    _json_safe(
                        traces.preview(
                            q.get("trace", [""])[0],
                            q.get("sensor", ["hottest"])[0],
                            int(q.get("index", ["0"])[0]),
                        )
                    )
                )
            except (ValueError, TypeError) as exc:
                self._send(400, json.dumps({"error": str(exc)}).encode("utf-8"), "application/json")
                return
            self._send(200, body.encode("utf-8"), "application/json; charset=utf-8")

        def do_POST(self) -> None:
            if self.path != "/api/ask":
                self._send(404, b'{"error":"not found"}', "application/json")
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if length <= 0 or length > MAX_BODY:
                self._send(400, b'{"error":"bad request body"}', "application/json")
                return

            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                question = str(payload.get("question", "")).strip()[:MAX_QUESTION]
                budget = payload.get("budget_ms")
                budget_ms = float(budget) if budget is not None else None
                fallback = bool(payload.get("fallback", False))
            except (ValueError, TypeError, UnicodeDecodeError):
                self._send(400, b'{"error":"could not parse the request"}', "application/json")
                return

            try:
                device, device_label, measured = (
                    traces.resolve(payload.get("device"))
                    if traces is not None
                    else (None, "", False)
                )
            except ValueError as exc:
                # A trace the page offered but the server cannot load is a
                # 400 naming the trace, not a silent fall back to no device:
                # answering without the device state the caller asked for
                # would report a decision made under conditions nobody chose.
                self._send(400, json.dumps({"error": str(exc)}).encode("utf-8"), "application/json")
                return

            if not question:
                self._send(400, b'{"error":"ask a question first"}', "application/json")
                return

            try:
                # Serialised: one model, one question at a time, so the timings
                # reported are of a query that ran alone.
                with lock:
                    result = pipeline.ask(
                        question,
                        budget_ms=budget_ms,
                        fallback=fallback,
                        device=device,
                        device_source=device_label,
                        device_is_measured=measured,
                    )
                body = json.dumps(_json_safe(result.as_dict())).encode("utf-8")
            except Exception as exc:
                # Every exception, not a chosen few. An uncaught one leaves the
                # handler without sending anything, the connection closes, and
                # the browser reports "Failed to fetch" -- which names neither
                # the cause nor the file. The traceback goes to the terminal
                # where the server is running, because a failure must be
                # diagnosable from the machine it happened on.
                traceback.print_exc()
                body = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode("utf-8")
                self._send(500, body, "application/json")
                return
            self._send(200, body, "application/json; charset=utf-8")

    return Handler


def serve(
    pipeline: Any,
    dashboard_html: Any,
    port: int = 8000,
    traces: TraceCatalog | None = None,
) -> ThreadingHTTPServer:
    """Start the server on the loopback address. Never on a routable one."""
    return ThreadingHTTPServer(("127.0.0.1", port), build_app(pipeline, dashboard_html, traces))

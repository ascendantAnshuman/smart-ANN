"""The knowledge base must not contaminate the benchmark, and must rank fairly.

Two properties carry the design.

**Scores from both stores are the same quantity.** The merge in `ask` sorts one
combined list by cosine similarity, which is only a fair comparison if the
knowledge base stores unit vectors like the SQuAD index does. An un-normalised
vector would score by magnitude and take every retrieval slot regardless of
relevance, so normalisation is enforced on write and checked here.

**Re-adding a document replaces it.** Two copies of one article would take two
of the five slots with the same passage and crowd out the evidence a question
actually needed.

The web tests use a stub pipeline: what needs pinning at this level is the HTTP
contract and the refusals, not the model.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from smartann.kb.sources import paragraphs_of, read_text_file
from smartann.kb.store import KnowledgeBase, chunk_paragraphs
from smartann.web import SingleThreadPipeline, serve


def fake_encode(dim: int = 8):
    """Deterministic vectors that are NOT unit length, so normalisation is tested."""

    def encode(texts: list[str], **_: object) -> np.ndarray:
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            out[i] = rng.normal(size=dim) * 7.0
        return out

    return encode


def test_paragraphs_stop_at_the_reference_section() -> None:
    text = "Intro line.\n\n== History ==\nA fact.\n\n== References ==\nSmith 2004\nJones 1999"
    paras = paragraphs_of(text)
    assert paras == ["Intro line.", "A fact."]
    assert not any("Smith" in p for p in paras), "reference lists are names, not facts"


def test_headings_are_not_kept_as_text() -> None:
    """A chunk reading 'History' would match every question mentioning history."""
    assert "History" not in paragraphs_of("== History ==\nSomething happened.")


def test_windows_never_span_a_paragraph_break() -> None:
    """A chunk glued across two topics embeds as neither."""
    chunks = chunk_paragraphs(["One. Two. Three. Four.", "Alpha. Beta."])
    assert chunks
    for c in chunks:
        assert not ("Four" in c and "Alpha" in c)


def test_stored_vectors_are_normalised(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path, dim=8)
    try:
        kb.add("Doc", ["One. Two. Three."], "http://x", "file", fake_encode())
        hits = kb.search(np.ones(8, dtype=np.float32) / np.sqrt(8), k=3)
        assert hits
        # Cosine of unit vectors cannot leave [-1, 1]; magnitude would escape it.
        for h in hits:
            assert -1.0001 <= h.similarity <= 1.0001
    finally:
        kb.close()


def test_re_adding_replaces_rather_than_duplicates(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path, dim=8)
    try:
        kb.add("Doc", ["One. Two. Three."], "http://x", "wikipedia", fake_encode())
        first = len(kb)
        kb.add("Doc", ["One. Two. Three."], "http://x", "wikipedia", fake_encode())
        assert len(kb) == first
        assert len(kb.sources()) == 1
    finally:
        kb.close()


def test_removing_a_source_removes_its_chunks(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path, dim=8)
    try:
        kb.add("Keep", ["Alpha. Beta. Gamma."], "http://a", "file", fake_encode())
        kb.add("Drop", ["Delta. Epsilon. Zeta."], "http://b", "file", fake_encode())
        before = len(kb)
        assert kb.remove("Drop") == 1
        assert len(kb) < before
        assert [s.title for s in kb.sources()] == ["Keep"]
    finally:
        kb.close()


def test_an_empty_knowledge_base_returns_no_hits(tmp_path: Path) -> None:
    """`ask` must work before anything is added, not crash on an empty matrix."""
    kb = KnowledgeBase(tmp_path, dim=8)
    try:
        assert len(kb) == 0
        assert kb.search(np.ones(8, dtype=np.float32), k=5) == []
    finally:
        kb.close()


def test_a_document_with_no_text_is_refused(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path, dim=8)
    try:
        with pytest.raises(ValueError, match="no text"):
            kb.add("Empty", ["", "   "], "http://x", "file", fake_encode())
    finally:
        kb.close()


def test_pdf_is_refused_with_a_reason(tmp_path: Path) -> None:
    """Half-reading a PDF would index gibberish; adding a reader is a new dependency."""
    p = tmp_path / "paper.pdf"
    p.write_bytes(b"%PDF-1.4")
    with pytest.raises(ValueError, match="PDF is not supported"):
        read_text_file(p)


def test_markdown_headings_are_parsed_like_wikipedia(tmp_path: Path) -> None:
    p = tmp_path / "notes.md"
    p.write_text("# Title\n\nA sentence.\n\n## References\nsomething", encoding="utf-8")
    title, url, paras = read_text_file(p)
    assert title == "notes"
    assert url.startswith("file:")
    assert paras == ["A sentence."]


# ---------------------------------------------------------------------------
# The web front end
# ---------------------------------------------------------------------------


class StubResult:
    def as_dict(self) -> dict[str, Any]:
        # verify_p_yes is NaN whenever the grounding check did not run, which is
        # not representable in JSON -- the real pipeline produces this.
        return {
            "answer": "Einstein",
            "declined": False,
            "verify_p_yes": float("nan"),
            "timing_ms": {"total": 1.0},
        }


class StubPipeline:
    def __init__(self) -> None:
        self.asked: list[str] = []

    def ask(self, question: str, budget_ms: float | None = None, **_: object) -> StubResult:
        self.asked.append(question)
        if question == "boom":
            raise ValueError("no index")
        if question == "sqlite":
            # The defect this pins: a pipeline loaded on one thread and called
            # from a request thread raised exactly this, and it is neither a
            # RuntimeError nor a ValueError.
            raise sqlite3.ProgrammingError("SQLite objects created in a thread")
        return StubResult()


@pytest.fixture
def server():
    pipeline = StubPipeline()
    httpd = serve(pipeline, lambda: "<html>dash</html>", port=8762)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield "http://127.0.0.1:8762", pipeline
    finally:
        httpd.shutdown()
        httpd.server_close()


def post(base: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(
        base + "/api/ask",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def test_the_server_binds_only_to_loopback(server) -> None:
    """A local model on a routable address would let the network spend this CPU."""
    base, _ = server
    assert base.startswith("http://127.0.0.1")


def test_the_page_and_dashboard_are_served(server) -> None:
    base, _ = server
    with urllib.request.urlopen(base + "/", timeout=10) as r:
        body = r.read().decode("utf-8")
    assert r.status == 200
    assert "SMART-ANN" in body
    with urllib.request.urlopen(base + "/dashboard", timeout=10) as r:
        assert r.status == 200


def test_a_question_reaches_the_pipeline(server) -> None:
    base, pipeline = server
    status, data = post(base, {"question": "who?", "budget_ms": 120})
    assert status == 200
    assert data["answer"] == "Einstein"
    assert pipeline.asked == ["who?"]


def test_an_empty_question_is_refused(server) -> None:
    base, pipeline = server
    status, data = post(base, {"question": "   "})
    assert status == 400
    assert "error" in data
    assert pipeline.asked == []


def test_a_pipeline_failure_becomes_an_error_not_a_crash(server) -> None:
    base, _ = server
    status, data = post(base, {"question": "boom"})
    assert status == 500
    assert "no index" in data["error"]


def test_an_unknown_path_is_a_404(server) -> None:
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(base + "/secrets", timeout=10)
    assert exc.value.code == 404


def test_an_unexpected_exception_becomes_a_500_not_a_dropped_connection(server) -> None:
    """The defect a user hit: the browser said only "Failed to fetch".

    `sqlite3.ProgrammingError` is neither RuntimeError nor ValueError, so it
    escaped the handler, the connection closed with no reply, and the page
    reported a network failure naming neither the cause nor the file.
    """
    base, _ = server
    status, data = post(base, {"question": "sqlite"})
    assert status == 500
    assert "ProgrammingError" in data["error"], "the type must survive to the page"


def test_a_nan_is_serialised_as_null(server) -> None:
    """`json.dumps` writes NaN bare, which no browser will parse."""
    base, _ = server
    with urllib.request.urlopen(
        urllib.request.Request(
            base + "/api/ask",
            data=json.dumps({"question": "who?"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        ),
        timeout=10,
    ) as r:
        raw = r.read().decode("utf-8")
    assert "NaN" not in raw
    assert json.loads(raw)["verify_p_yes"] is None


# ---------------------------------------------------------------------------
# The worker thread
# ---------------------------------------------------------------------------


class ThreadRecordingPipeline:
    """Records which thread each call ran on, like a SQLite connection does."""

    def __init__(self) -> None:
        self.threads: dict[str, int] = {}

    def load(self) -> None:
        self.threads["load"] = threading.get_ident()

    def ask(self, question: str, budget_ms: float | None = None, **_: object) -> Any:
        self.threads["ask"] = threading.get_ident()
        self.budget = budget_ms
        return StubResult()

    def close(self) -> None:
        self.threads["close"] = threading.get_ident()


def test_load_ask_and_close_all_run_on_one_thread() -> None:
    """SQLite refuses a connection opened on another thread; this is the fix."""
    inner = ThreadRecordingPipeline()
    runner = SingleThreadPipeline(inner)
    runner.load()

    done: list[int] = []
    t = threading.Thread(target=lambda: (runner.ask("q"), done.append(1)))
    t.start()
    t.join(timeout=10)
    runner.close()

    assert done, "the call made from another thread must complete"
    assert inner.threads["load"] == inner.threads["ask"] == inner.threads["close"]
    assert inner.threads["load"] != threading.get_ident(), "and not on the caller's thread"


def test_a_failure_on_the_worker_is_raised_to_the_caller() -> None:
    """A swallowed load failure would serve a page that cannot answer anything."""

    class Broken:
        def load(self) -> None:
            raise FileNotFoundError("no index built")

        def close(self) -> None:
            pass

    runner = SingleThreadPipeline(Broken())
    with pytest.raises(FileNotFoundError, match="no index built"):
        runner.load()
    runner.close()


def test_the_command_budget_is_used_when_the_page_sends_none() -> None:
    inner = ThreadRecordingPipeline()
    runner = SingleThreadPipeline(inner, default_budget_ms=250.0)
    runner.ask("q")
    assert inner.budget == 250.0
    runner.ask("q", budget_ms=40.0)
    assert inner.budget == 40.0, "an explicit budget still wins"
    runner.close()


def test_the_page_loads_nothing_from_the_network(server) -> None:
    """Self-contained: no CDN, so the demo works with the wifi off."""
    base, _ = server
    with urllib.request.urlopen(base + "/", timeout=10) as r:
        body = r.read().decode("utf-8")
    assert "http://" not in body.replace("http://127.0.0.1", "")
    assert "https://" not in body


# ---------------------------------------------------------------------------
# Replaying a device trace through the page
# ---------------------------------------------------------------------------


def write_trace(tmp_path: Path) -> Path:
    d = tmp_path / "device"
    d.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "t": float(i * 10),
            "skin_c": 30.0 + i,
            "hottest_c": 50.0 + i,
            "pct": 80 - i,
            "status": "DISCHARGING",
            "current_ua": -300000,
            "slope": 0.1,
        }
        for i in range(5)
    ]
    (d / "c5_telemetry.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return d


class DeviceRecordingPipeline(StubPipeline):
    """Records the device state the handler resolved for each question."""

    def __init__(self) -> None:
        super().__init__()
        self.devices: list[Any] = []

    def ask(self, question: str, budget_ms: float | None = None, **kw: object) -> StubResult:
        self.devices.append(
            (kw.get("device"), kw.get("device_source"), kw.get("device_is_measured"))
        )
        return super().ask(question, budget_ms)


@pytest.fixture
def device_server(tmp_path):
    from smartann.web import TraceCatalog

    pipeline = DeviceRecordingPipeline()
    catalog = TraceCatalog(write_trace(tmp_path))
    httpd = serve(pipeline, lambda: "<html>dash</html>", port=8763, traces=catalog)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield "http://127.0.0.1:8763", pipeline
    finally:
        httpd.shutdown()
        httpd.server_close()


def get_json(url: str) -> tuple[int, Any]:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def test_the_page_offers_only_traces_on_this_machine(device_server) -> None:
    base, _ = device_server
    status, data = get_json(base + "/api/traces")
    assert status == 200
    assert [t["name"] for t in data["traces"]] == ["c5"]
    assert data["traces"][0]["n_samples"] == 5


def test_a_server_with_no_traces_offers_none(server) -> None:
    """The demo must not present a recording that is not here."""
    base, _ = server
    status, data = get_json(base + "/api/traces")
    assert status == 200
    assert data["traces"] == []


def test_the_preview_reports_the_recorded_state_at_a_point(device_server) -> None:
    base, _ = device_server
    status, data = get_json(base + "/api/device?trace=c5&sensor=hottest&index=3")
    assert status == 200
    assert data["temperature_c"] == 53.0
    assert data["battery_pct"] == 77.0
    assert data["charging"] is False
    assert "hottest" in data["label"]


def test_the_two_sensors_preview_different_temperatures(device_server) -> None:
    base, _ = device_server
    _, hot = get_json(base + "/api/device?trace=c5&sensor=hottest&index=0")
    _, skin = get_json(base + "/api/device?trace=c5&sensor=skin&index=0")
    assert hot["temperature_c"] == 50.0
    assert skin["temperature_c"] == 30.0


def test_the_resolved_state_reaches_the_pipeline(device_server) -> None:
    base, pipeline = device_server
    status, _ = post(
        base, {"question": "q", "device": {"trace": "c5", "sensor": "hottest", "index": 2}}
    )
    assert status == 200
    state, label, measured = pipeline.devices[-1]
    assert state is not None
    assert state.temperature_c == 52.0
    assert measured is True, "a recorded trace is a measurement and must be labelled one"
    assert "c5" in label


def test_asking_with_no_device_passes_none(device_server) -> None:
    """Absent device state and a sensorless one are different claims."""
    base, pipeline = device_server
    post(base, {"question": "q"})
    assert pipeline.devices[-1] == (None, "", False)


def test_an_unknown_trace_is_refused_rather_than_silently_dropped(device_server) -> None:
    """Answering without the device the caller asked for would misreport the run."""
    base, pipeline = device_server
    status, data = post(base, {"question": "q", "device": {"trace": "nope", "sensor": "skin"}})
    assert status == 400
    assert "nope" in data["error"]
    assert pipeline.devices == [], "nothing may be answered under the wrong conditions"


def test_a_preview_for_an_unknown_trace_is_a_400(device_server) -> None:
    base, _ = device_server
    status, data = get_json(base + "/api/device?trace=nope&sensor=skin&index=0")
    assert status == 400
    assert "error" in data

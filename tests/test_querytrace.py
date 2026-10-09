"""The query trace: what it records, and what it refuses to claim.

The trace is what the demonstration and the dashboard both read, so a field that
lies here lies everywhere downstream. The tests that matter are therefore less
about shape than about provenance: that a hypothetical device state cannot be
reported as measured, that an absent reading stays absent instead of becoming a
zero, and that the network claim is a count rather than an assertion.
"""

from __future__ import annotations

import json
import math
import socket
import threading
from pathlib import Path

import pytest

from smartann.pipeline.ask import AskResult, RetrievedChunk, Stage
from smartann.pipeline.explain import render_story
from smartann.pipeline.querytrace import (
    NetworkWitness,
    doc_slugs,
    estimate_energy,
    from_ask_result,
)
from smartann.types import AnnConfig, ScrConfig

ANN = AnnConfig(nprobe=4, ef_centroid=128, ef_list=64, k=5, per_cluster_k=32)
SCR = ScrConfig(1, 0, 0, 1, 0.0, enabled=False)


def _chunk(rank: int = 1, chunk_id: int = 7, sim: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank,
        chunk_id=chunk_id,
        distance=1.0 - sim,
        text="a passage about heat",
        reduced_text="a passage about heat",
        origin="demo_phone_care",
        title="Why a phone gets warm",
    )


def _result(**over: object) -> AskResult:
    base: dict[str, object] = {
        "question": "why is my phone hot?",
        "answer": "because work makes heat",
        "chunks": (_chunk(),),
        "ann": ANN,
        "scr": SCR,
        "policy_name": "device",
        "policy_reasons": ("temperature 55.5C >= hot 46C",),
        "profile_entry": None,
        "table_measured_under": "a laptop",
        "budget_ms": 8.0,
        "stages": (Stage("embed", 1.0, "cache hit"), Stage("search", 2.0, "nprobe=4")),
        "controller_overhead_us": 16.0,
        "prompt_tokens": 100,
        "generated_tokens": 10,
        "truncated": False,
        "n_cluster_loads": 4,
        "n_dist_exact": 68,
        "dataset": "demo_phone_care",
        "index_root": Path("index"),
        "generator_name": "google/flan-t5-base",
    }
    base.update(over)
    return AskResult(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def test_a_hypothetical_device_state_is_never_reported_as_measured() -> None:
    """The whole point of carrying `device_is_measured` through the pipeline.

    A typed temperature produces a real controller decision, and that decision
    is worth showing -- but the state behind it must not be presentable as a
    reading the device gave.
    """
    r = _result(
        device={"temperature_c": 44.0, "battery_pct": 12.0},
        device_source="typed by hand",
        device_is_measured=False,
    )
    t = from_ask_result(r)
    assert t.resources.source == "hypothetical"
    assert t.resources.is_measured is False
    assert t.resources.measured_on == "nothing -- typed by hand"
    assert "NOT A MEASUREMENT" in render_story(t)


def test_a_replayed_state_is_labelled_as_the_handset_not_this_laptop() -> None:
    r = _result(
        device={"temperature_c": 55.5, "battery_pct": 75.0},
        device_source="measured trace c5, sample 1/237 at t+0s, hottest sensor",
        device_is_measured=True,
    )
    t = from_ask_result(r)
    assert t.resources.source == "trace-replay"
    assert t.resources.measured_on == "handset (recorded trace)"


def test_a_live_laptop_reading_is_labelled_as_this_laptop() -> None:
    r = _result(
        device={"cpu_util": 0.3, "mem_available_mb": 8000.0},
        device_source="WindowsMonitor",
        device_is_measured=True,
    )
    t = from_ask_result(r)
    assert t.resources.source == "live sensor"
    assert t.resources.measured_on == "this laptop"


def test_an_unreadable_sensor_stays_absent_rather_than_becoming_zero() -> None:
    """A laptop reports no temperature. Zero would read as a very cold phone."""
    r = _result(
        device={"cpu_util": 0.3, "mem_available_mb": 8000.0},
        device_source="WindowsMonitor",
        device_is_measured=True,
    )
    t = from_ask_result(r)
    assert math.isnan(t.resources.temperature_c)
    assert "temperature_c" in t.resources.absent_fields()
    assert "battery_pct" in t.resources.absent_fields()
    assert "not measured" in render_story(t)


def test_no_device_state_at_all_is_its_own_case() -> None:
    t = from_ask_result(_result())
    assert t.resources.source == "not measured"
    assert t.resources.is_measured is False


# ---------------------------------------------------------------------------
# The controller's one-line reason
# ---------------------------------------------------------------------------
def test_a_derate_names_both_levels_and_what_it_cost() -> None:
    r = _result(
        device={"temperature_c": 55.5},
        device_source="measured trace c5",
        device_is_measured=True,
        baseline_nprobe=6,
        baseline_recall=1.0,
    )
    t = from_ask_result(r)
    reason = t.controller.reason
    assert "6" in reason and "4" in reason  # from the baseline to the chosen level
    assert "temperature" in reason
    assert t.controller.derated is True


def test_no_derate_says_so_rather_than_staying_silent() -> None:
    """'The policy looked and nothing was wrong' and 'no policy ran' are
    different facts, and a reader must be able to tell them apart."""
    r = _result(
        device={"temperature_c": 30.0},
        device_source="measured trace c5",
        device_is_measured=True,
        baseline_nprobe=4,
        baseline_recall=1.0,
        policy_reasons=(),
    )
    t = from_ask_result(r)
    assert "no derate" in t.controller.reason
    assert t.controller.derated is False


def test_with_no_table_and_no_budget_the_reason_says_the_level_was_given() -> None:
    t = from_ask_result(_result(budget_ms=None, policy_name="static", policy_reasons=()))
    assert "fixed nprobe" in t.controller.reason


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------
def test_energy_is_the_measured_coefficient_times_the_clusters_opened(
    tmp_path: Path,
) -> None:
    model = tmp_path / "energy_model.json"
    model.write_text(json.dumps({"attribution": {"mj_per_cluster": 3.2, "r2": 0.16}}))
    e = estimate_energy(4, model)
    assert e.mj == pytest.approx(12.8)
    assert e.label == "estimate"
    assert "0.16" in e.coefficient_source  # the weak fit travels with the number
    assert "generation" in e.excludes


def test_without_an_energy_model_nothing_is_estimated(tmp_path: Path) -> None:
    """No coefficient means no number, not a default one."""
    e = estimate_energy(4, tmp_path / "absent.json")
    assert math.isnan(e.mj)
    assert e.label == "not measured"


def test_a_malformed_energy_model_does_not_produce_a_number(tmp_path: Path) -> None:
    model = tmp_path / "energy_model.json"
    model.write_text('{"attribution": {}}')
    assert math.isnan(estimate_energy(4, model).mj)


# ---------------------------------------------------------------------------
# The network witness
# ---------------------------------------------------------------------------
def test_the_witness_counts_bytes_this_process_actually_sends() -> None:
    """A loopback server, so the count is of real socket traffic rather than of
    a mock that agreed with the assertion."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    received: list[int] = []

    def serve() -> None:
        conn, _ = server.accept()
        received.append(len(conn.recv(1024)))
        conn.close()

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        with NetworkWitness() as witness:
            client = socket.socket()
            client.connect(("127.0.0.1", port))
            client.sendall(b"hello there")
            client.close()
    finally:
        thread.join(timeout=5)
        server.close()

    assert witness.bytes_sent == len(b"hello there")
    assert witness.connections == 1
    assert witness.silent is False


def test_a_query_that_sends_nothing_reports_silence() -> None:
    with NetworkWitness() as witness:
        pass
    assert witness.silent is True
    t = from_ask_result(
        _result(), network_bytes_sent=witness.bytes_sent, network_connections=witness.connections
    )
    assert t.sent_nothing is True
    assert "0 bytes sent" in render_story(t)


def test_the_witness_restores_the_socket_class_when_it_exits() -> None:
    """It patches a shared class. Leaving the patch in place would make every
    later socket in the process pay for it, including in other tests."""
    original = socket.socket.sendall
    with NetworkWitness():
        assert socket.socket.sendall is not original
    assert socket.socket.sendall is original


def test_the_witness_restores_the_socket_class_after_an_exception() -> None:
    original = socket.socket.send
    with pytest.raises(ValueError), NetworkWitness():
        raise ValueError("boom")
    assert socket.socket.send is original


def test_a_connection_with_no_payload_is_still_recorded() -> None:
    """Opening a connection tells a remote host this device exists, even if no
    application bytes follow. A byte count alone would report that as silence."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    thread = threading.Thread(target=lambda: server.accept()[0].close())
    thread.start()
    try:
        with NetworkWitness() as witness:
            client = socket.socket()
            client.connect(("127.0.0.1", port))
            client.close()
    finally:
        thread.join(timeout=5)
        server.close()

    assert witness.bytes_sent == 0
    assert witness.connections == 1
    assert witness.silent is False  # not silent, despite zero bytes


# ---------------------------------------------------------------------------
# Serialisation and document ids
# ---------------------------------------------------------------------------
def test_a_trace_round_trips_through_json(tmp_path: Path) -> None:
    t = from_ask_result(_result())
    path = t.write(tmp_path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["query"] == "why is my phone hot?"
    assert loaded["retrieval"]["clusters_probed"] == 4
    assert loaded["network"]["sent_nothing"] is True
    assert loaded["controller"]["chosen_nprobe"] == 4


def test_two_traces_of_one_question_do_not_overwrite_each_other(tmp_path: Path) -> None:
    a = from_ask_result(_result()).write(tmp_path)
    b = from_ask_result(_result()).write(tmp_path)
    assert a != b
    assert a.exists() and b.exists()


def test_doc_slugs_reads_the_mapping_from_the_corpus_database(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "chunks.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE documents (doc_id INTEGER PRIMARY KEY, file_path TEXT, title TEXT);"
        "CREATE TABLE metadata (chunk_id INTEGER PRIMARY KEY, doc_id INTEGER);"
    )
    conn.execute("INSERT INTO documents VALUES (0, 'demo_phone_care/heat-basics', 'Basics')")
    conn.execute("INSERT INTO metadata VALUES (0, 0)")
    conn.execute("INSERT INTO metadata VALUES (1, 0)")
    conn.commit()
    conn.close()

    slugs = doc_slugs(db)
    assert slugs == {0: "heat-basics", 1: "heat-basics"}


def test_a_missing_corpus_database_yields_no_doc_ids_rather_than_raising(
    tmp_path: Path,
) -> None:
    assert doc_slugs(tmp_path / "nothing.sqlite") == {}


def test_a_knowledge_base_hit_gets_no_document_slug() -> None:
    """KB chunk ids live in a different table and would collide with corpus ids.
    An empty slug is right; a slug from the wrong store would be a false
    citation."""
    kb_chunk = RetrievedChunk(
        rank=1,
        chunk_id=0,
        distance=0.1,
        text="t",
        reduced_text="t",
        origin="kb",
        title="Wikipedia thing",
    )
    t = from_ask_result(_result(chunks=(kb_chunk,)), doc_ids={0: "heat-basics"})
    assert t.chunks[0].doc_id == ""

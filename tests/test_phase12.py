"""The Phase 12 features: the guard, the cache, the stress modes, the blocker.

Each of these can fail in a way that produces a plausible answer rather than an
error, which is why they are tested on behaviour rather than on shape: a guard
that never refuses, a cache that answers one question with another's answer, a
stress mode whose typed temperature is reported as a reading, and a network
blocker that lets something through all look fine from the outside.
"""

from __future__ import annotations

import json
import math
import socket
from pathlib import Path

import numpy as np
import pytest

from smartann.pipeline.cache import SemanticCache
from smartann.pipeline.guard import REFUSAL, GroundingGuard
from smartann.pipeline.offline import NetworkBlockedError, block_network
from smartann.telemetry.stress import load_modes

# ---------------------------------------------------------------------------
# 12.1 the grounding guard
# ---------------------------------------------------------------------------
GUARD = GroundingGuard(
    threshold=0.83,
    answerable_floor=0.87,
    unanswerable_ceiling=0.78,
    calibrated_from="a test",
    dataset="demo_phone_care",
)


def test_a_score_above_the_threshold_is_answered() -> None:
    v = GUARD.verdict(0.90)
    assert v.grounded is True
    assert v.margin == pytest.approx(0.07)


def test_a_score_below_the_threshold_is_refused() -> None:
    v = GUARD.verdict(0.72)
    assert v.grounded is False
    assert v.margin < 0
    assert "below" in v.reason


def test_the_boundary_is_answered_not_refused() -> None:
    """At exactly the threshold the margin is zero, and a zero margin is not a
    failure. Stated because `>` and `>=` here decide a user-visible behaviour."""
    assert GUARD.verdict(0.83).grounded is True


def test_retrieving_nothing_at_all_is_refused() -> None:
    """An empty retrieval has no best score. Answering from no evidence is the
    one case where a refusal is unarguable."""
    v = GUARD.verdict(float("nan"))
    assert v.grounded is False
    assert "nothing was retrieved" in v.reason


def test_a_disabled_guard_answers_everything_and_says_so() -> None:
    v = GroundingGuard.off().verdict(0.1)
    assert v.grounded is True
    assert math.isnan(v.threshold)


def test_calibration_takes_the_midpoint_of_the_measured_gap(tmp_path: Path) -> None:
    p = tmp_path / "eval.json"
    p.write_text(
        json.dumps(
            {
                "dataset": "demo_phone_care",
                "separation": {"lowest_answerable": 0.90, "highest_unanswerable": 0.70},
                "paraphrases": {"n": 10},
                "off_topic_in_corpus": {"n": 5},
                "out_of_corpus": {"n": 5},
            }
        )
    )
    g = GroundingGuard.calibrate(p)
    assert g.threshold == pytest.approx(0.80)
    assert g.gap == pytest.approx(0.20)
    assert g.dataset == "demo_phone_care"


def test_calibration_refuses_when_the_two_groups_overlap(tmp_path: Path) -> None:
    """No threshold separates overlapping distributions. Returning one anyway
    would hide that behind a number that looks like every other threshold."""
    p = tmp_path / "eval.json"
    p.write_text(
        json.dumps(
            {
                "dataset": "d",
                "separation": {"lowest_answerable": 0.70, "highest_unanswerable": 0.80},
                "paraphrases": {"n": 1},
                "off_topic_in_corpus": {"n": 1},
                "out_of_corpus": {"n": 1},
            }
        )
    )
    with pytest.raises(ValueError, match="no gap"):
        GroundingGuard.calibrate(p)


def test_a_guard_calibrated_elsewhere_is_refused(tmp_path: Path) -> None:
    """Same rule as the profile table: a threshold measured on another corpus
    produces confident refusals that nothing in the output marks as borrowed."""
    p = tmp_path / "guard.json"
    GUARD.write(p)
    assert GroundingGuard.load(p, dataset="demo_phone_care").threshold == pytest.approx(0.83)
    with pytest.raises(ValueError, match="calibrated on"):
        GroundingGuard.load(p, dataset="squad")


def test_the_refusal_text_does_not_pretend_to_be_an_answer() -> None:
    assert "couldn't find" in REFUSAL.lower()


# ---------------------------------------------------------------------------
# 12.2 the semantic cache
# ---------------------------------------------------------------------------
def _unit(*values: float) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_an_identical_query_hits() -> None:
    c: SemanticCache[str] = SemanticCache(threshold=0.9)
    v = _unit(1.0, 0.0, 0.0)
    c.store(v, "why is my phone hot", "payload", cost_ms=900.0)
    hit = c.lookup(v)
    assert hit is not None
    assert hit.payload == "payload"
    assert hit.saved_ms == 900.0


def test_a_near_duplicate_above_the_threshold_hits() -> None:
    c: SemanticCache[str] = SemanticCache(threshold=0.9)
    c.store(_unit(1.0, 0.0), "a", "payload", cost_ms=1.0)
    hit = c.lookup(_unit(1.0, 0.2))  # cosine ~0.98
    assert hit is not None
    assert hit.similarity > 0.9


def test_a_different_question_misses() -> None:
    """The failure that matters. A collision does not degrade an answer, it
    replaces it with another question's answer, confidently."""
    c: SemanticCache[str] = SemanticCache(threshold=0.9)
    c.store(_unit(1.0, 0.0), "a", "payload", cost_ms=1.0)
    assert c.lookup(_unit(0.0, 1.0)) is None
    assert c.misses == 1


def test_the_nearest_entry_wins_not_the_first_stored() -> None:
    """With several near-duplicates resident, answering with whichever was
    inserted first would make the result depend on insertion order."""
    c: SemanticCache[str] = SemanticCache(threshold=0.5)
    c.store(_unit(1.0, 0.3), "near", "NEAR", cost_ms=1.0)
    c.store(_unit(1.0, 0.0), "exact", "EXACT", cost_ms=1.0)
    hit = c.lookup(_unit(1.0, 0.0))
    assert hit is not None
    assert hit.payload == "EXACT"


def test_the_cache_is_bounded() -> None:
    c: SemanticCache[int] = SemanticCache(threshold=0.99, max_entries=3)
    for i in range(10):
        v = np.zeros(10, dtype=np.float32)
        v[i] = 1.0
        c.store(v, f"q{i}", i, cost_ms=1.0)
    assert len(c) == 3


def test_eviction_is_by_least_recently_used_not_least_recently_stored() -> None:
    """The question an audience keeps asking must stay resident even as newer
    ones arrive."""
    c: SemanticCache[str] = SemanticCache(threshold=0.99, max_entries=2)
    a, b, d = np.eye(3, dtype=np.float32)
    c.store(a, "a", "A", cost_ms=1.0)
    c.store(b, "b", "B", cost_ms=1.0)
    assert c.lookup(a) is not None  # touch 'a', making 'b' the least recent
    c.store(d, "d", "D", cost_ms=1.0)
    assert c.lookup(a) is not None
    assert c.lookup(b) is None


def test_restoring_a_question_replaces_rather_than_duplicates() -> None:
    c: SemanticCache[str] = SemanticCache(threshold=0.99)
    v = _unit(1.0, 0.0)
    c.store(v, "q", "old", cost_ms=1.0)
    c.store(v, "q", "new", cost_ms=2.0)
    assert len(c) == 1
    hit = c.lookup(v)
    assert hit is not None and hit.payload == "new"


def test_vectors_are_normalised_on_the_way_in() -> None:
    """Lookup is a dot product, which is only a cosine on unit vectors. An
    unnormalised entry would score above threshold against anything long."""
    c: SemanticCache[str] = SemanticCache(threshold=0.9)
    c.store(np.array([10.0, 0.0], dtype=np.float32), "q", "p", cost_ms=1.0)
    hit = c.lookup(_unit(1.0, 0.0))
    assert hit is not None
    assert hit.similarity == pytest.approx(1.0, abs=1e-5)


def test_a_zero_vector_cannot_be_cached() -> None:
    c: SemanticCache[str] = SemanticCache()
    with pytest.raises(ValueError):
        c.store(np.zeros(4, dtype=np.float32), "q", "p", cost_ms=1.0)


def test_an_empty_cache_misses_without_raising() -> None:
    c: SemanticCache[str] = SemanticCache()
    assert c.lookup(_unit(1.0, 0.0)) is None


def test_a_threshold_outside_the_unit_interval_is_refused() -> None:
    with pytest.raises(ValueError):
        SemanticCache(threshold=0.0)
    with pytest.raises(ValueError):
        SemanticCache(threshold=1.5)


# ---------------------------------------------------------------------------
# 12.3 stress modes
# ---------------------------------------------------------------------------
DEVICE_DIR = Path(__file__).resolve().parent.parent / "results" / "device"


@pytest.mark.skipif(not DEVICE_DIR.is_dir(), reason="no recorded traces on disk")
def test_measured_modes_come_from_a_real_sample() -> None:
    modes = load_modes(DEVICE_DIR)
    hot = modes["hot"]
    assert hot.is_measured
    # The hottest sample in the C5 trace, which is far above a typed 44 C.
    assert hot.state.temperature_c > 46.0
    assert "measured" in hot.label


@pytest.mark.skipif(not DEVICE_DIR.is_dir(), reason="no recorded traces on disk")
def test_typed_modes_are_labelled_hypothetical_everywhere() -> None:
    """A typed temperature that reached a report as a reading would be the one
    thing this project does not do."""
    modes = load_modes(DEVICE_DIR)
    for name in ("hot-44", "low-battery", "battery-saver"):
        mode = modes[name]
        assert not mode.is_measured
        assert "HYPOTHETICAL" in mode.label
        assert mode.provenance == "hypothetical"


@pytest.mark.skipif(not DEVICE_DIR.is_dir(), reason="no recorded traces on disk")
def test_the_low_battery_mode_leaves_temperature_absent() -> None:
    """So the derate it shows is the battery tier alone, not a typed temperature
    quietly doing half the work."""
    mode = load_modes(DEVICE_DIR)["low-battery"]
    assert math.isnan(mode.state.temperature_c)
    assert mode.state.battery_pct == 12.0
    assert mode.state.charging is False


def test_modes_still_exist_when_no_trace_is_on_disk(tmp_path: Path) -> None:
    """The hypothetical modes do not need a trace. The measured ones must simply
    be absent rather than filled in with plausible numbers."""
    modes = load_modes(tmp_path)
    assert "hot-44" in modes
    assert "low-battery" in modes
    assert "normal" not in modes
    assert "hot" not in modes


# ---------------------------------------------------------------------------
# 12.5 the network blocker
# ---------------------------------------------------------------------------
def test_an_outbound_connection_raises() -> None:
    with block_network() as report, pytest.raises(NetworkBlockedError):
        socket.create_connection(("example.com", 80), timeout=1)
    assert report.attempts
    assert not report.clean


def test_name_resolution_is_blocked_too() -> None:
    """A DNS lookup is itself traffic leaving the device: a pipeline that
    resolves a hostname and then fails has still told a resolver what it wanted."""
    with block_network(), pytest.raises(NetworkBlockedError):
        socket.getaddrinfo("example.com", 80)


def test_loopback_is_blocked_as_well() -> None:
    """Allowing 127.0.0.1 would be convenient and wrong: a local service is
    still a service the pipeline depends on being up."""
    with block_network(), pytest.raises(NetworkBlockedError):
        socket.create_connection(("127.0.0.1", 9), timeout=1)


def test_code_that_touches_no_network_runs_untouched() -> None:
    with block_network() as report:
        total = sum(range(1000))
    assert total == 499500
    assert report.clean


def test_the_socket_module_is_restored_afterwards() -> None:
    original_connect = socket.socket.connect
    original_getaddrinfo = socket.getaddrinfo
    with block_network():
        pass
    assert socket.socket.connect is original_connect
    assert socket.getaddrinfo is original_getaddrinfo


def test_the_socket_module_is_restored_after_an_exception() -> None:
    original = socket.getaddrinfo
    with pytest.raises(NetworkBlockedError), block_network():
        socket.getaddrinfo("example.com", 80)
    assert socket.getaddrinfo is original


# ---------------------------------------------------------------------------
# the recorded Phase 12 results, when they are on disk
# ---------------------------------------------------------------------------
RESULTS = Path(__file__).resolve().parent.parent / "results"


@pytest.mark.skipif(
    not (RESULTS / "offline_proof.json").exists(), reason="offline proof has not been run"
)
def test_the_recorded_offline_proof_passed() -> None:
    d = json.loads((RESULTS / "offline_proof.json").read_text(encoding="utf-8"))
    assert d["passed"] is True
    assert d["runs"]["with_offline_env"]["network_attempts"] == []
    assert d["n_chunks_retrieved"] > 0


@pytest.mark.skipif(
    not (RESULTS / "demo_profile_table.json").exists(), reason="profile table has not been built"
)
def test_the_profile_table_recall_ladder_rises_with_effort() -> None:
    """Recall must not fall as more clusters are opened. It can plateau; a drop
    means the table is measuring something other than the index."""
    d = json.loads((RESULTS / "demo_profile_table.json").read_text(encoding="utf-8"))
    recalls = [e["recall"] for e in d["entries"]]
    assert recalls == sorted(recalls)
    assert recalls[0] < recalls[-1]

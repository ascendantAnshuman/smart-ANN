"""The single-query path, with the two ways it can lie held down.

The first is a decision that is printed but not carried out. The probe runs at
the widest level any decision could need, and the search must then visit only
the clusters the controller actually chose. Skip that narrowing and the report
says "nprobe=8" while thirty-two clusters are opened: the controller looks
adaptive, the work is not, and every cost in the run belongs to a configuration
nobody selected. `test_search_opens_only_the_chosen_clusters` is the guard.

The second is a profile table from the wrong corpus. Latencies and recalls from
another dataset are entirely plausible numbers, which is what makes them
dangerous, so the table's own `dataset` field is checked rather than trusted.

Everything here runs against stubs. The real index, corpus and generator are
exercised by Phase 7; what needs pinning at this level is the wiring between
them, and stubs make the cluster count observable in a way a real search does
not.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from smartann.controller.profile import ProfileEntry, ProfileTable
from smartann.generate.base import GenerationResult
from smartann.index.ecovector import ProbeResult, SearchResult
from smartann.pipeline.ask import (
    SCR_OFF,
    AskResult,
    RagPipeline,
    RetrievedChunk,
    load_profile_table,
)
from smartann.pipeline.ask_report import render
from smartann.telemetry.replay import what_if_state
from smartann.types import DeviceState, QueryFeatures, ScrConfig

FEATURES = QueryFeatures(
    centroid_top1_dist=0.1,
    centroid_dist_gap=1.4,
    centroid_dist_entropy=0.5,
    query_norm=1.0,
    query_token_len=5,
)


class StubIndex:
    """Records the probe width it was asked for and what `finish` then received."""

    def __init__(self, n_clusters: int = 64) -> None:
        self.n_clusters = n_clusters
        self.probed_nprobe: int | None = None
        self.finished_cluster_count: int | None = None
        self.sizes = np.full(n_clusters, 100, dtype=np.int64)

    def probe(self, query: np.ndarray, cfg, token_len: int = 0) -> ProbeResult:
        self.probed_nprobe = cfg.nprobe
        n = min(cfg.nprobe, self.n_clusters)
        return ProbeResult(
            cluster_ids=np.arange(n, dtype=np.int64),
            centroid_dists=np.linspace(0.1, 0.9, n),
            features=FEATURES,
            t_centroid_ms=0.08,
            n_dist_modeled=n,
        )

    def finish(self, query: np.ndarray, probe: ProbeResult, cfg) -> SearchResult:
        self.finished_cluster_count = int(probe.cluster_ids.size)
        k = cfg.k
        return SearchResult(
            ids=np.arange(k, dtype=np.int64),
            # Distances in hnswlib's ip convention: 1 - cosine similarity.
            dists=np.linspace(0.05, 0.25, k),
            features=probe.features,
            n_cluster_loads=probe.cluster_ids.size,
            n_dist_exact=probe.cluster_ids.size * 100,
        )


class StubCorpus:
    def __init__(self, text: str = "A passage about something. And a second sentence.") -> None:
        self.text = text
        self.closed = False

    def chunk_texts(self, ids) -> list[str]:
        return [f"{self.text} [{i}]" for i in ids]

    def close(self) -> None:
        self.closed = True


class StubEmbedder:
    model_name = "stub/embedder"
    cache = None

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls = 0

    def encode(self, texts, **_: object) -> np.ndarray:
        self.calls += 1
        v = np.ones((len(texts), self.dim), dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


class StubGenerator:
    """Scripted answers, with the grounding verdict and memory answer settable."""

    def __init__(
        self,
        text: str = "a short answer",
        model_id: str = "stub/gen",
        *,
        grounded: bool = True,
        memory: str = "a remembered answer",
    ) -> None:
        self.text = text
        self.cfg = type("C", (), {"model_id": model_id})()
        self.last_context: str | None = None
        self.grounded = grounded
        self.memory = memory
        self.verify_calls = 0
        self.closed_book_calls = 0

    def generate(self, question: str, context: str) -> GenerationResult:
        self.last_context = context
        return GenerationResult(
            text=self.text,
            prompt_tokens=123,
            generated_tokens=4,
            t_generate_ms=42.0,
            truncated=False,
        )

    def context_answers(self, question: str, context: str) -> tuple[bool, float, float]:
        self.verify_calls += 1
        return self.grounded, (0.9 if self.grounded else 0.1), 7.0

    def closed_book(self, question: str) -> GenerationResult:
        self.closed_book_calls += 1
        return GenerationResult(
            text=self.memory,
            prompt_tokens=10,
            generated_tokens=3,
            t_generate_ms=30.0,
            truncated=False,
        )


def table(dataset: str = "squad") -> ProfileTable:
    """A table whose levels are far enough apart that a budget picks exactly one."""
    return ProfileTable(
        entries=(
            ProfileEntry(1, 1000, 5.0, 10.0, 0.60, 100.0),
            ProfileEntry(8, 1000, 50.0, 100.0, 0.90, 100.0),
            ProfileEntry(32, 1000, 180.0, 350.0, 0.98, 100.0),
        ),
        dataset=dataset,
        measured_under="a stub",
        min_samples=100,
    )


def wired(
    *,
    tbl: ProfileTable | None = None,
    answer: str = "a short answer",
    index: StubIndex | None = None,
    grounded: bool = True,
    memory: str = "a remembered answer",
) -> RagPipeline:
    """A pipeline with every heavyweight part replaced, ready to `ask`."""
    p = RagPipeline(dataset="squad", k=5)
    p._index = index or StubIndex()
    p._corpus = StubCorpus()
    p._embedder = StubEmbedder()
    p._generator = StubGenerator(answer, grounded=grounded, memory=memory)
    p._table = tbl
    p.index_root = Path("stub")
    return p


def test_search_opens_only_the_chosen_clusters() -> None:
    """The probe goes wide; the search must not follow it there.

    Without narrowing this opens 32 clusters while reporting nprobe=8, so the
    controller's decision costs nothing and changes nothing.
    """
    idx = StubIndex()
    p = wired(tbl=table(), index=idx)

    r = p.ask("a question", budget_ms=120.0)

    assert r.ann.nprobe == 8, "budget 120 with headroom 0.9 admits the p95=100 level"
    assert idx.probed_nprobe == 32, "the probe must cover the widest level available"
    assert idx.finished_cluster_count == 8, "the search must visit only the chosen clusters"
    assert r.n_cluster_loads == 8


def test_a_tighter_budget_buys_less_effort() -> None:
    """The whole point of the budget policy, on one query."""
    p = wired(tbl=table())
    assert p.ask("q", budget_ms=120.0).ann.nprobe == 8
    assert p.ask("q", budget_ms=15.0).ann.nprobe == 1


def test_without_a_table_the_given_nprobe_is_used() -> None:
    """No measured table means nothing to choose from, so nothing is chosen."""
    idx = StubIndex()
    p = wired(tbl=None, index=idx)
    r = p.ask("q", budget_ms=120.0, nprobe=4)
    assert r.policy_name == "static"
    assert r.ann.nprobe == 4
    assert idx.finished_cluster_count == 4
    assert r.profile_entry is None


def test_a_table_from_another_dataset_is_refused(tmp_path: Path) -> None:
    """sift1m latencies must not choose effort for a squad query."""
    path = tmp_path / "t.json"
    table("sift1m").write(path)
    with pytest.raises(ValueError, match="measured on 'sift1m'"):
        load_profile_table(path, "squad")


def test_a_table_for_the_same_dataset_loads(tmp_path: Path) -> None:
    """The check must admit the matching case, not refuse everything."""
    path = tmp_path / "t.json"
    table("squad").write(path)
    assert load_profile_table(path, "squad").dataset == "squad"


def test_an_empty_question_is_refused() -> None:
    p = wired()
    with pytest.raises(ValueError, match="empty question"):
        p.ask("   ")


def test_ask_before_load_is_refused() -> None:
    with pytest.raises(RuntimeError, match="load"):
        RagPipeline().ask("q")


def test_similarity_inverts_the_ip_distance() -> None:
    """The index stores 1 - cosine; the report shows cosine."""
    c = RetrievedChunk(rank=1, chunk_id=7, distance=0.25, text="abc", reduced_text="ab")
    assert c.similarity == pytest.approx(0.75)
    assert (c.chars_in, c.chars_out) == (3, 2)


def test_a_declined_answer_is_detected_and_explained() -> None:
    """'unanswerable' is a correct reply, and the report must say so."""
    p = wired(answer="unanswerable")
    r = p.ask("something outside the corpus")
    assert r.declined is True
    assert "declined" in render(r).lower()


def test_an_ordinary_answer_is_not_flagged_as_declined() -> None:
    r = wired(answer="Paris").ask("q")
    assert r.declined is False
    assert "The model declined" not in render(r)


def test_reduction_shrinks_the_context_and_is_reported() -> None:
    """With SCR on, the chars-in/chars-out pair must reflect real removal."""
    p = wired()
    scr = ScrConfig(1, 0, 0, 1, 0.0, enabled=True)
    r = p.ask("q", scr=scr)
    assert r.chars_out < r.chars_in
    assert "top_windows=1" in render(r)


def test_passthrough_keeps_every_character() -> None:
    r = wired().ask("q", scr=SCR_OFF)
    assert r.chars_out == r.chars_in
    assert "byte-identical passthrough" in render(r)


def test_only_reduced_text_reaches_the_generator() -> None:
    """The model must see what SCR produced, not the original chunks."""
    p = wired()
    gen: StubGenerator = p._generator
    p.ask("q", scr=ScrConfig(1, 0, 0, 1, 0.0, enabled=True))
    context = gen.last_context or ""
    assert context
    assert len(context) < sum(len(t) for t in StubCorpus().chunk_texts(range(5)))


def test_stage_costs_are_recorded_for_every_stage() -> None:
    r = wired().ask("q")
    names = [s.name for s in r.stages]
    assert names == ["embed", "probe", "search", "reduce", "generate", "verify"]
    assert r.total_ms == pytest.approx(sum(s.ms for s in r.stages))
    assert all(s.ms >= 0 for s in r.stages)


def test_as_dict_round_trips_through_json() -> None:
    r = wired(tbl=table()).ask("q", budget_ms=120.0)
    back = json.loads(json.dumps(r.as_dict()))
    assert back["policy"]["chosen_nprobe"] == 8
    assert back["policy"]["table_entry"]["nprobe"] == 8
    assert len(back["retrieval"]) == 5


def test_the_report_labels_table_recall_as_the_levels_not_the_querys() -> None:
    """A typed question has no ground truth, so its recall must not be implied."""
    text = render(wired(tbl=table()).ask("q", budget_ms=120.0))
    assert "measured recall" in text
    assert "not this question's recall" in text


def test_the_report_names_the_conditions_the_table_was_measured_under() -> None:
    """Phone p95 choosing effort for a laptop query must be visible, not silent."""
    assert "a stub" in render(wired(tbl=table()).ask("q", budget_ms=120.0))


def test_truncation_is_surfaced() -> None:
    """A truncated prompt means retrieved context never reached the model."""
    r = wired().ask("q")
    truncated = AskResult(**{**r.__dict__, "truncated": True})
    assert "truncated" in render(truncated)


def test_close_releases_the_corpus() -> None:
    p = wired()
    corpus: StubCorpus = p._corpus
    p.close()
    assert corpus.closed is True


def test_missing_index_names_the_build_command() -> None:
    """An absent index must say how to make one, not fail obscurely."""
    p = RagPipeline(dataset="nosuchdataset")
    with pytest.raises(FileNotFoundError, match="index build"):
        p.load(load_generator=False)


# ---------------------------------------------------------------------------
# Grounding: an extracted span is only shown as a finding if the passages
# actually contain the answer.
# ---------------------------------------------------------------------------


def test_a_span_the_grounding_check_rejects_is_withheld() -> None:
    """The Google-CEO case: a plausible name from passages that lack the fact."""
    r = wired(answer="Brian McClendon", grounded=False).ask("who is the ceo of google")
    assert r.source == "not_in_corpus"
    assert r.declined is True
    assert r.overruled is True
    assert r.extracted_answer == "Brian McClendon"
    assert r.answer != "Brian McClendon", "a rejected span must not be shown as the answer"
    text = render(r)
    assert "withheld" in text
    assert "Brian McClendon" in text, "what was withheld must still be visible, and say so"


def test_a_span_the_grounding_check_accepts_is_shown() -> None:
    r = wired(answer="Einstein", grounded=True).ask("who developed relativity")
    assert r.source == "corpus"
    assert r.answer == "Einstein"
    assert r.declined is False
    assert "grounding check" in render(r)


def test_an_unanswerable_reply_skips_the_check() -> None:
    """Nothing to verify when the model already declined; the check would be wasted work."""
    p = wired(answer="unanswerable")
    r = p.ask("q")
    gen: StubGenerator = p._generator
    assert gen.verify_calls == 0
    assert r.declined is True
    assert r.overruled is False


def test_verify_can_be_switched_off() -> None:
    """Off, the original extractive-only behaviour returns unchanged."""
    p = wired(answer="Brian McClendon", grounded=False)
    r = p.ask("q", verify=False)
    gen: StubGenerator = p._generator
    assert gen.verify_calls == 0
    assert r.source == "corpus"
    assert r.answer == "Brian McClendon"


def test_memory_is_asked_only_when_the_corpus_fails_and_only_when_requested() -> None:
    p = wired(answer="Einstein", grounded=True)
    p.ask("q", fallback=True)
    gen: StubGenerator = p._generator
    assert gen.closed_book_calls == 0, "a grounded answer must not be replaced by memory"

    p = wired(answer="Brian McClendon", grounded=False)
    p.ask("q", fallback=False)
    gen = p._generator
    assert gen.closed_book_calls == 0, "memory must be opt-in"


def test_a_memory_answer_is_labelled_as_memory() -> None:
    """The one outcome that must never be mistaken for a retrieved finding."""
    r = wired(answer="unanswerable", memory="Sundar Pichai").ask("q", fallback=True)
    assert r.source == "model_memory"
    assert r.from_model_memory is True
    assert r.answer == "Sundar Pichai"
    assert r.declined is True, "the corpus still did not answer"
    text = render(r)
    assert "NOT FROM THE CORPUS" in text
    assert "recall" in [s.name for s in r.stages]


def test_source_reaches_the_json() -> None:
    back = json.loads(json.dumps(wired(answer="x", grounded=False).ask("q").as_dict()))
    assert back["source"] == "not_in_corpus"
    assert back["extracted_answer"] == "x"


# ---------------------------------------------------------------------------
# Answering under a device condition
# ---------------------------------------------------------------------------
# The claim these pin is the project's headline one: that effort follows the
# device's state. It is easy to write a controller that prints a device reading
# and then chooses the same level regardless, so what is checked here is the
# level actually searched, not the prose beside it.


def hot(temp_c: float = 50.0, battery: float = 80.0, charging: bool = False) -> DeviceState:
    """A device state with only the fields a policy reads set; the rest absent."""
    return what_if_state(temperature_c=temp_c, battery_pct=battery, charging=charging)


def test_a_hot_device_buys_less_effort_than_the_budget_allowed() -> None:
    p = wired(tbl=table())
    cool = p.ask("q", budget_ms=120.0, device=hot(temp_c=25.0))
    burning = p.ask("q", budget_ms=120.0, device=hot(temp_c=50.0))
    assert cool.ann.nprobe == 8
    assert burning.ann.nprobe < cool.ann.nprobe, "a hot device must cost effort"


def test_the_derate_is_carried_out_not_just_reported() -> None:
    """The defect this guards: printing nprobe=1 while searching eight clusters."""
    idx = StubIndex()
    p = wired(tbl=table(), index=idx)
    r = p.ask("q", budget_ms=120.0, device=hot(temp_c=50.0))
    assert idx.finished_cluster_count == r.ann.nprobe


def test_the_baseline_records_what_the_budget_alone_would_have_chosen() -> None:
    """A derate must be a difference between two real levels, not an assertion."""
    r = wired(tbl=table()).ask("q", budget_ms=120.0, device=hot(temp_c=50.0))
    assert r.baseline_nprobe == 8
    assert r.derated is True
    assert r.recall_given_up > 0, "giving up effort gives up measured recall"


def test_a_device_that_crosses_no_threshold_is_not_a_derate() -> None:
    r = wired(tbl=table()).ask("q", budget_ms=120.0, device=hot(temp_c=25.0))
    assert r.derated is False
    assert r.baseline_nprobe == r.ann.nprobe


def test_the_reported_level_is_the_one_actually_searched() -> None:
    """Printing the budget's row beside a derated search would describe a run nobody made."""
    r = wired(tbl=table()).ask("q", budget_ms=120.0, device=hot(temp_c=50.0))
    assert r.profile_entry is not None
    assert r.profile_entry.nprobe == r.ann.nprobe


def test_a_device_of_all_nan_derates_nothing_and_says_so() -> None:
    """A sensorless platform must read as absent, never as a cool, healthy device."""
    r = wired(tbl=table()).ask("q", budget_ms=120.0, device=what_if_state())
    assert r.ann.nprobe == 8
    assert r.policy_reasons == ()
    assert r.device_signals_absent is True
    assert "acted on nothing" in render(r)


def test_no_device_means_no_device_policy_ran() -> None:
    """Absent device state and an all-NaN one are different claims."""
    r = wired(tbl=table()).ask("q", budget_ms=120.0)
    assert r.device is None
    assert r.baseline_nprobe is None
    assert r.derated is False


def test_a_hypothetical_state_is_labelled_as_one_in_the_report() -> None:
    """Rule 1: a typed number must never be presentable as a measurement."""
    text = render(
        wired(tbl=table()).ask("q", budget_ms=120.0, device=hot(), device_is_measured=False)
    )
    assert "HYPOTHETICAL" in text


def test_a_measured_state_is_not_labelled_hypothetical() -> None:
    text = render(
        wired(tbl=table()).ask(
            "q", budget_ms=120.0, device=hot(), device_source="trace c5", device_is_measured=True
        )
    )
    assert "HYPOTHETICAL" not in text
    assert "trace c5" in text


def test_a_low_battery_on_mains_power_does_not_derate() -> None:
    """Charging is a real signal, used only to decline to derate."""
    p = wired(tbl=table())
    on_battery = p.ask("q", budget_ms=120.0, device=hot(temp_c=25.0, battery=5.0, charging=False))
    plugged_in = p.ask("q", budget_ms=120.0, device=hot(temp_c=25.0, battery=5.0, charging=True))
    assert on_battery.ann.nprobe < plugged_in.ann.nprobe


def test_the_device_reaches_the_json() -> None:
    back = json.loads(
        json.dumps(wired(tbl=table()).ask("q", budget_ms=120.0, device=hot(temp_c=50.0)).as_dict())
    )
    assert back["device"]["state"]["temperature_c"] == 50.0
    assert back["device"]["measured"] is False
    assert back["policy"]["derated"] is True
    assert back["policy"]["baseline_nprobe"] == 8


def test_a_cool_device_is_not_reported_as_an_unmeasured_one() -> None:
    """Two opposite situations both produce an empty reason list.

    A phone at 27 C crosses no threshold; a laptop with no thermometer crosses
    none either. Rendering both as "the policy acted on nothing" would report a
    measurement as an absence, which is the failure Section 3 rule 1 names.
    """
    cool = wired(tbl=table()).ask("q", budget_ms=120.0, device=hot(temp_c=27.0, battery=75.0))
    assert cool.policy_reasons == ()
    assert cool.device_signals_absent is False
    text = render(cool)
    assert "acted on nothing" not in text
    assert "none crossed a threshold" in text


def test_a_sensorless_platform_is_reported_as_absence() -> None:
    blind = wired(tbl=table()).ask("q", budget_ms=120.0, device=what_if_state())
    assert blind.device_signals_absent is True
    assert "acted on nothing" in render(blind)


def test_signals_absent_reaches_the_json() -> None:
    back = json.loads(
        json.dumps(wired(tbl=table()).ask("q", budget_ms=120.0, device=what_if_state()).as_dict())
    )
    assert back["device"]["signals_absent"] is True


def test_a_query_signal_is_not_reported_as_a_device_signal() -> None:
    """The label must follow the policy: a gap is not a thermometer reading."""
    from smartann.controller.policies.query import QueryEffortFit

    f = QueryEffortFit(
        signal="centroid_dist_gap",
        edges=(0.01,),
        assignment=(1, 8),
        budget_ms=100.0,
        dataset="squad",
        delta_recall=0.01,
        cost_saving=0.0,
        n_held_out=500,
    )
    r = wired(tbl=table()).ask("q", budget_ms=120.0, query_fit=f)
    text = render(r)
    assert "device signals used" not in text
    assert "centroid_dist_gap" in text
    assert r.policy_name == "query-adaptive"


def test_the_query_fit_can_lower_effort_below_the_budgets_level() -> None:
    from smartann.controller.policies.query import QueryEffortFit

    f = QueryEffortFit(
        signal="centroid_dist_gap",
        edges=(),
        assignment=(1,),  # every query to the cheapest level
        budget_ms=100.0,
        dataset="squad",
        delta_recall=0.0,
        cost_saving=0.2,
        n_held_out=500,
    )
    r = wired(tbl=table()).ask("q", budget_ms=120.0, query_fit=f)
    assert r.ann.nprobe == 1
    assert r.baseline_nprobe == 8, "the budget alone would have spent eight"
    assert r.derated is True

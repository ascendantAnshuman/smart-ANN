"""The demonstration corpus and its hand-written labels.

The corpus is committed source rather than generated data, so nothing regenerates
it and nothing else checks it. These are the invariants the rest of Phase 11
assumes: that every document can be cited, that every label names a document
that exists, and that the distractors are really unrelated.

A broken invariant here does not crash anything. It produces an index that
builds, a demo that answers, and an evaluation whose numbers are quietly
measuring the wrong thing -- which is why it is worth a test file of its own.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

CORPUS = Path(__file__).resolve().parent.parent / "data" / "demo_phone_care"
DOCS_PATH = CORPUS / "documents.json"
QUERIES_PATH = CORPUS / "eval_queries.json"

REQUIRED_FIELDS = {"id", "title", "text", "source"}
DISTRACTOR_PREFIX = "distract-"
# The window the corpus was written to. Short enough that a chunk is a coherent
# passage, long enough that a document says something.
MIN_WORDS = 100
MAX_WORDS = 250


@pytest.fixture(scope="module")
def docs() -> list[dict[str, str]]:
    return json.loads(DOCS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def spec() -> dict[str, object]:
    return json.loads(QUERIES_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The documents
# ---------------------------------------------------------------------------
def test_the_corpus_is_the_size_the_demo_claims(docs: list[dict[str, str]]) -> None:
    topical = [d for d in docs if not d["id"].startswith(DISTRACTOR_PREFIX)]
    distractors = [d for d in docs if d["id"].startswith(DISTRACTOR_PREFIX)]
    assert len(docs) == 71
    assert len(topical) == 47
    assert len(distractors) == 24


def test_every_document_has_every_field(docs: list[dict[str, str]]) -> None:
    for d in docs:
        assert set(d) == REQUIRED_FIELDS, d.get("id")
        for key in REQUIRED_FIELDS:
            assert str(d[key]).strip(), f"{d.get('id')} has an empty {key}"


def test_every_document_declares_that_it_was_authored_here(
    docs: list[dict[str, str]],
) -> None:
    """The provenance label travels with the document into the index, the trace
    and the interface. A document without it could be shown as a source."""
    assert {d["source"] for d in docs} == {"authored for demo"}


def test_document_ids_are_unique(docs: list[dict[str, str]]) -> None:
    """Two documents sharing an id would make a citation ambiguous, and the
    evaluation labels address documents by id."""
    ids = [d["id"] for d in docs]
    assert len(set(ids)) == len(ids)


def test_titles_are_unique(docs: list[dict[str, str]]) -> None:
    """The interface cites by title. Two identical titles would show a reader
    two different passages under one name."""
    titles = [d["title"] for d in docs]
    assert len(set(titles)) == len(titles)


def test_every_document_is_within_the_written_length_window(
    docs: list[dict[str, str]],
) -> None:
    wrong = [
        (d["id"], len(d["text"].split()))
        for d in docs
        if not MIN_WORDS <= len(d["text"].split()) <= MAX_WORDS
    ]
    assert not wrong, f"outside {MIN_WORDS}-{MAX_WORDS} words: {wrong}"


def test_no_document_text_is_duplicated(docs: list[dict[str, str]]) -> None:
    """Two copies of one passage would take two of the five retrieval slots with
    the same evidence, which is the failure the knowledge base guards against
    for the same reason."""
    bodies = [" ".join(d["text"].split()) for d in docs]
    assert len(set(bodies)) == len(bodies)


def test_ids_are_lowercase_slugs(docs: list[dict[str, str]]) -> None:
    """Ids become document slugs in `chunks.sqlite` via `file_path`, and the
    trace splits that field on '/' to recover them."""
    for d in docs:
        assert d["id"] == d["id"].lower()
        assert "/" not in d["id"]
        assert all(ch.isalnum() or ch == "-" for ch in d["id"]), d["id"]


# ---------------------------------------------------------------------------
# The labels
# ---------------------------------------------------------------------------
def test_every_gold_label_names_a_document_that_exists(
    docs: list[dict[str, str]], spec: dict[str, object]
) -> None:
    """A label naming a document that was renamed or removed scores as a
    permanent miss, and nothing else would report it."""
    ids = {d["id"] for d in docs}
    for key in (
        "gold_primary",
        "gold_relevant",
        "gold_primary_frozen",
        "gold_primary_extended",
    ):
        named = set(spec[key])  # type: ignore[arg-type]
        assert named <= ids, f"{key} names documents that do not exist: {named - ids}"
    for item in spec["off_topic"]:  # type: ignore[index]
        named = set(item["gold"])
        assert named <= ids, f"off_topic gold names missing documents: {named - ids}"


def test_the_primary_set_is_a_subset_of_the_relevant_set(spec: dict[str, object]) -> None:
    assert set(spec["gold_primary"]) <= set(spec["gold_relevant"])  # type: ignore[arg-type]


def test_the_frozen_primary_set_is_inside_the_extended_one(
    spec: dict[str, object],
) -> None:
    """The extended set may only add. If it ever dropped a document, a later run
    would not be comparable with the number the frozen set was recorded at."""
    frozen = set(spec["gold_primary_frozen"])  # type: ignore[arg-type]
    extended = set(spec["gold_primary_extended"])  # type: ignore[arg-type]
    assert frozen <= extended
    assert frozen == set(spec["gold_primary"])  # type: ignore[arg-type]


def test_no_distractor_is_labelled_relevant_to_the_heating_question(
    spec: dict[str, object],
) -> None:
    """The distractors exist to be wrong answers for this question. One of them
    in the relevant set would make the discrimination test vacuous."""
    relevant = set(spec["gold_relevant"])  # type: ignore[arg-type]
    assert not {s for s in relevant if s.startswith(DISTRACTOR_PREFIX)}


def test_the_primary_documents_are_not_distractors(spec: dict[str, object]) -> None:
    primary = set(spec["gold_primary_extended"])  # type: ignore[arg-type]
    assert not {s for s in primary if s.startswith(DISTRACTOR_PREFIX)}


def test_there_are_ten_paraphrases_and_ten_off_topic_queries(
    spec: dict[str, object],
) -> None:
    assert len(spec["paraphrases"]) == 10  # type: ignore[arg-type]
    assert len(spec["off_topic"]) == 10  # type: ignore[arg-type]


def test_the_off_topic_queries_are_split_between_both_kinds(
    spec: dict[str, object],
) -> None:
    """Phase 12.1's refusal threshold is the gap between these two groups. With
    only one kind present there is no gap to measure."""
    kinds = [item["kind"] for item in spec["off_topic"]]  # type: ignore[index]
    assert set(kinds) == {"off_topic_in_corpus", "out_of_corpus"}
    assert kinds.count("off_topic_in_corpus") == 5
    assert kinds.count("out_of_corpus") == 5


def test_an_answerable_off_topic_query_names_its_gold_and_an_unanswerable_one_does_not(
    spec: dict[str, object],
) -> None:
    for item in spec["off_topic"]:  # type: ignore[index]
        if item["kind"] == "off_topic_in_corpus":
            assert item["gold"], f"{item['query']} is marked answerable but names no gold"
        else:
            assert item["gold"] == [], f"{item['query']} is marked unanswerable but names gold"


def test_the_paraphrases_are_distinct_wordings_of_one_question(
    spec: dict[str, object],
) -> None:
    """Ten copies of one sentence would measure nothing about robustness to
    phrasing, which is what the paraphrase set is for."""
    paraphrases = [str(q) for q in spec["paraphrases"]]  # type: ignore[arg-type]
    assert len(set(paraphrases)) == len(paraphrases)
    # Every one of them asks about heat, and none of them mentions a specific
    # cause -- these are paraphrases of the general question, not variants.
    for q in paraphrases:
        low = q.lower()
        assert any(w in low for w in ("hot", "heat", "warm"))


def test_every_query_is_a_question_a_person_would_type(spec: dict[str, object]) -> None:
    queries = [str(q) for q in spec["paraphrases"]] + [  # type: ignore[arg-type]
        str(item["query"])
        for item in spec["off_topic"]  # type: ignore[index]
    ]
    for q in queries:
        assert q.strip() == q
        assert len(q.split()) >= 4
        assert q.endswith("?")


# ---------------------------------------------------------------------------
# The build config agrees with the corpus
# ---------------------------------------------------------------------------
def test_the_index_config_partitions_the_corpus_the_registry_rule_would(
    docs: list[dict[str, str]],
) -> None:
    """`nc` is written out in the config because this dataset is not in the
    registry. It must still be what the registry's auto rule would produce, or
    the demo index is partitioned differently from every other index here.
    """
    import math

    import yaml

    from smartann.text import sentence_windows, split_sentences

    cfg = yaml.safe_load(
        (CORPUS.parent.parent / "configs" / "index" / "demo_phone_care_ecovector.yaml").read_text(
            encoding="utf-8"
        )
    )
    n_chunks = sum(len(list(sentence_windows(split_sentences(d["text"]), 3, 1))) for d in docs)
    assert n_chunks == 255
    assert cfg["nc"] == round(math.sqrt(n_chunks))
    assert cfg["dataset"] == "demo_phone_care"
    # Default min_graph_size, i.e. every cluster flat and mmap-read (ADR-0028).
    assert "min_graph_size" not in cfg

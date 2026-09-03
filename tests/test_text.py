"""Phase 1: the shared sentence splitter (ADR-0012).

Chunking and SCR both depend on this, so its span contract is tested hard: a
splitter that silently rewrites text would make every stored ``char_offset``
wrong, and the damage would only surface in Phase 5.
"""

from __future__ import annotations

import itertools

import pytest

from smartann.text import Sentence, sentence_windows, split_sentences


def test_basic_split() -> None:
    got = split_sentences("One. Two. Three.")
    assert [s.text for s in got] == ["One.", "Two.", "Three."]


@pytest.mark.parametrize(
    "text",
    [
        "Dr. Smith went home. He slept.",
        "It cost $3.14 in total. That is cheap.",
        "The U.S. economy grew. Analysts agreed.",
        "See No. 5 on the list. It is highlighted.",
    ],
)
def test_abbreviations_and_decimals_do_not_split(text: str) -> None:
    """The cases a naive regex splitter mangles; all appear in SQuAD contexts."""
    assert len(split_sentences(text)) == 2, text


def test_spans_index_back_into_the_source() -> None:
    text = "Alpha beta. Gamma delta! Epsilon?  Zeta."
    for s in split_sentences(text):
        assert text[s.start : s.end] == s.text


def test_spans_are_ordered_and_non_overlapping() -> None:
    text = "One sentence here. Another one follows. And a third."
    got = split_sentences(text)
    assert len(got) == 3
    for a, b in itertools.pairwise(got):
        assert a.end <= b.start


def test_spans_exclude_surrounding_whitespace() -> None:
    for s in split_sentences("  Leading space. Trailing space.   "):
        assert s.text == s.text.strip()
        assert s.text != ""


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_empty_input_yields_nothing(text: str) -> None:
    assert split_sentences(text) == []


def test_single_sentence_without_terminator() -> None:
    got = split_sentences("no terminator here")
    assert len(got) == 1
    assert got[0].text == "no terminator here"


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------
def _fake(n: int) -> list[Sentence]:
    return [Sentence(text=f"s{i}", start=i, end=i + 1) for i in range(n)]


def test_windows_step_is_size_minus_overlap() -> None:
    assert sentence_windows(_fake(9), 3, 1) == [(0, 3), (2, 5), (4, 7), (6, 9)]


def test_windows_cover_every_sentence() -> None:
    for n in range(1, 30):
        spans = sentence_windows(_fake(n), 3, 1)
        covered = {i for lo, hi in spans for i in range(lo, hi)}
        assert covered == set(range(n)), f"n={n} left sentences uncovered"


def test_short_document_yields_one_window() -> None:
    """Dropping short paragraphs would silently shrink the corpus."""
    assert sentence_windows(_fake(1), 3, 1) == [(0, 1)]
    assert sentence_windows(_fake(2), 3, 1) == [(0, 2)]


def test_zero_overlap_is_disjoint() -> None:
    spans = sentence_windows(_fake(8), 4, 0)
    assert spans == [(0, 4), (4, 8)]


def test_empty_document_yields_no_windows() -> None:
    assert sentence_windows([], 3, 1) == []


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (3, 3), (3, 4), (3, -1)])
def test_invalid_window_parameters_raise(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        sentence_windows(_fake(10), size, overlap)


def test_windows_terminate_when_overlap_is_large() -> None:
    """overlap = size - 1 gives step 1; it must still terminate and cover."""
    spans = sentence_windows(_fake(6), 3, 2)
    assert spans[0] == (0, 3)
    assert spans[-1][1] == 6
    assert len(spans) == 4

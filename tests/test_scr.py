"""Selective Content Reduction. No model is loaded: `encode` is injected.

The scorer is a stub whose behaviour is known exactly, so these tests check the
REDUCTION LOGIC rather than an embedding model's opinions. A test that depended
on gte-small's similarities would be testing the model.
"""

from __future__ import annotations

import numpy as np
import pytest

from smartann.scr import reduce_context, score_windows
from smartann.types import ScrConfig

# 10 sentences; sentence i contains the token "S<i>".
TEXT = " ".join(f"This is sentence S{i} of the document." for i in range(10))
QUERY = np.array([1.0, 0.0], dtype=np.float32)


def _encoder(target: str):
    """Scores a window 1.0 if it contains `target`, else 0.0.

    Deterministic and inspectable, so a failing test points at the reduction
    logic rather than at a model's similarity judgement.
    """

    def encode(texts: list[str]) -> np.ndarray:
        return np.array(
            [[1.0, 0.0] if target in t else [0.0, 1.0] for t in texts], dtype=np.float32
        )

    return encode


def _cfg(**kw) -> ScrConfig:  # type: ignore[no-untyped-def]
    base = {
        "sliding_window_size": 3,
        "overlap_size": 1,
        "context_extension_size": 0,
        "top_windows": 1,
        "similarity_threshold": 0.0,
        "enabled": True,
    }
    return ScrConfig(**{**base, **kw})


# ---------------------------------------------------------------------------
# the ablation baseline
# ---------------------------------------------------------------------------
def test_disabled_returns_the_input_byte_for_byte() -> None:
    """C2's ablation compares against this; any normalisation would be measured."""
    r = reduce_context(TEXT, QUERY, _encoder("S5"), _cfg(enabled=False))
    assert r.text == TEXT
    assert r.chars_out == r.chars_in
    assert r.reduction_ratio == 0.0
    assert r.enabled is False


def test_disabled_does_not_call_the_encoder() -> None:
    """Turning SCR off must not pay the embedding cost it exists to justify."""

    def explode(texts: list[str]) -> np.ndarray:
        raise AssertionError("encoder called with SCR disabled")

    reduce_context(TEXT, QUERY, explode, _cfg(enabled=False))


# ---------------------------------------------------------------------------
# reduction
# ---------------------------------------------------------------------------
def test_reduction_keeps_the_matching_window() -> None:
    r = reduce_context(TEXT, QUERY, _encoder("S5"), _cfg())
    assert "S5" in r.text
    assert r.chars_out < r.chars_in
    assert 0.0 < r.reduction_ratio < 1.0


def test_more_windows_kept_means_less_reduction() -> None:
    """Monotone in top_windows -- the knob the controller will turn for C2."""
    sizes = [
        reduce_context(TEXT, QUERY, _encoder("sentence"), _cfg(top_windows=k)).chars_out
        for k in (1, 2, 3)
    ]
    assert sizes == sorted(sizes), f"kept text must not shrink as top_windows rises: {sizes}"


def test_context_extension_widens_the_kept_span() -> None:
    narrow = reduce_context(TEXT, QUERY, _encoder("S5"), _cfg(context_extension_size=0))
    wide = reduce_context(TEXT, QUERY, _encoder("S5"), _cfg(context_extension_size=2))
    assert wide.n_sentences_kept > narrow.n_sentences_kept
    assert wide.chars_out > narrow.chars_out


def test_overlapping_kept_windows_are_merged_not_duplicated() -> None:
    """Windows overlap by construction; emitting a sentence twice would inflate
    the token count SCR exists to reduce."""
    r = reduce_context(
        TEXT, QUERY, _encoder("sentence"), _cfg(top_windows=5, context_extension_size=2)
    )
    for a, b in zip(r.kept_windows, r.kept_windows[1:], strict=False):
        assert a[1] < b[0], f"spans {a} and {b} overlap or touch and should have merged"
    assert r.text.count("S5 ") <= 1


def test_a_threshold_excluding_everything_still_returns_context() -> None:
    """An empty context would surface as a generation failure, not a retrieval miss."""
    r = reduce_context(TEXT, QUERY, _encoder("ABSENT"), _cfg(similarity_threshold=0.9))
    assert r.text.strip()
    assert len(r.kept_windows) == 1


def test_threshold_drops_windows_below_it() -> None:
    loose = reduce_context(
        TEXT, QUERY, _encoder("sentence"), _cfg(top_windows=4, similarity_threshold=0.0)
    )
    strict = reduce_context(
        TEXT, QUERY, _encoder("S5"), _cfg(top_windows=4, similarity_threshold=0.5)
    )
    assert strict.chars_out < loose.chars_out


# ---------------------------------------------------------------------------
# reproducibility from the record
# ---------------------------------------------------------------------------
def test_kept_windows_index_back_into_the_source() -> None:
    """The decision must be reconstructable from stored data alone."""
    from smartann.text import split_sentences

    r = reduce_context(TEXT, QUERY, _encoder("S5"), _cfg(top_windows=2))
    sentences = split_sentences(TEXT)
    for i, j in r.kept_windows:
        assert 0 <= i < j <= len(sentences)
        assert TEXT[sentences[i].start : sentences[j - 1].end] in r.text


def test_reduction_is_deterministic() -> None:
    a = reduce_context(TEXT, QUERY, _encoder("S5"), _cfg(top_windows=2))
    b = reduce_context(TEXT, QUERY, _encoder("S5"), _cfg(top_windows=2))
    assert a == b


def test_scores_are_similarities_not_distances() -> None:
    """Larger is better here; the index stores the opposite convention
    (ADR-0018), and conflating them ranks every window backwards."""
    _, scored = score_windows(TEXT, QUERY, _encoder("S5"), _cfg())
    hit = [w for w in scored if "S5" in TEXT[w.char_start : w.char_end]]
    miss = [w for w in scored if "S5" not in TEXT[w.char_start : w.char_end]]
    assert hit and miss
    assert min(w.score for w in hit) > max(w.score for w in miss)


def test_empty_and_whitespace_input_is_handled() -> None:
    for text in ("", "   \n  "):
        r = reduce_context(text, QUERY, _encoder("x"), _cfg())
        assert r.text == text
        assert r.n_sentences_total == 0


def test_token_estimate_is_labelled_as_an_estimate() -> None:
    """Phase 7 is blocked, so no real tokenizer exists to count with."""
    r = reduce_context(TEXT, QUERY, _encoder("S5"), _cfg())
    assert r.approx_tokens() == pytest.approx(r.chars_out / 4.0, abs=1)
    assert r.approx_tokens(chars_per_token=2.0) > r.approx_tokens(chars_per_token=8.0)

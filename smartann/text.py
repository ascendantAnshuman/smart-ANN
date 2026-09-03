"""Sentence segmentation, shared by dataset chunking and SCR.

Phase 1 chunking builds 3-sentence windows over corpus paragraphs; Phase 5's
Selective Content Reduction builds its own sentence windows over retrieved text.
If those two used different splitters, SCR's windows would not align with chunk
boundaries -- a silent correctness bug that would surface only as inexplicably
poor SCR performance, several phases later. So there is exactly one splitter and
both import it (ADR-0012).

``pysbd`` is the backend: it is pure Python (no compiler, which matters on this
machine), rule-based and therefore deterministic, and it handles the
abbreviation and decimal cases that a naive ``re.split(r'[.!?]')`` mangles --
"No. 5", "Dr. Smith", "3.14", "U.S." all appear in SQuAD contexts.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import pysbd

__all__ = ["Sentence", "sentence_windows", "split_sentences"]

# pysbd.Segmenter is not documented as thread-safe and holds per-instance state,
# so it is created once per thread rather than shared or rebuilt per call
# (rebuilding costs ~10 ms, which is intolerable inside the SCR path).
_local = threading.local()


@dataclass(frozen=True)
class Sentence:
    """One sentence, with its span in the *original* text.

    The offsets are what make chunk provenance auditable: ``text[start:end]``
    must reproduce the sentence exactly, so a chunk stored in SQLite can always
    be traced back to its source paragraph.
    """

    text: str
    start: int
    end: int


def _segmenter() -> pysbd.Segmenter:
    seg = getattr(_local, "seg", None)
    if seg is None:
        seg = pysbd.Segmenter(language="en", clean=False, char_span=True)
        _local.seg = seg
    return seg


def split_sentences(text: str) -> list[Sentence]:
    """Split *text* into sentences carrying their character spans.

    Returns an empty list for empty or whitespace-only input. Spans are always
    valid indices into *text*; this is asserted, because a splitter that
    silently rewrites the text (normalising quotes, say) would make every stored
    ``char_offset`` wrong.
    """
    if not text or not text.strip():
        return []

    out: list[Sentence] = []
    for span in _segmenter().segment(text):
        start, end = int(span.start), int(span.end)
        # pysbd's char_span keeps trailing whitespace inside the span; trim it
        # so offsets point at real sentence content.
        raw = text[start:end]
        lstrip = len(raw) - len(raw.lstrip())
        rstrip = len(raw) - len(raw.rstrip())
        start += lstrip
        end -= rstrip
        if end <= start:
            continue
        out.append(Sentence(text=text[start:end], start=start, end=end))

    for s in out:
        assert text[s.start : s.end] == s.text, "sentence span does not match source text"
    return out


def sentence_windows(
    sentences: list[Sentence], window_size: int, overlap: int
) -> list[tuple[int, int]]:
    """Index ranges ``[i, j)`` of a sliding window over *sentences*.

    ``overlap`` is the number of sentences shared between consecutive windows,
    so the step is ``window_size - overlap``. A document shorter than one window
    yields a single window covering all of it, rather than nothing -- dropping
    short paragraphs would silently shrink the corpus.
    """
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    if not 0 <= overlap < window_size:
        raise ValueError(f"overlap must be in [0, {window_size}), got {overlap}")

    n = len(sentences)
    if n == 0:
        return []
    if n <= window_size:
        return [(0, n)]

    step = window_size - overlap
    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + window_size, n)
        spans.append((start, end))
        if end == n:
            break
        start += step
    return spans

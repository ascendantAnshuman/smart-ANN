"""Selective Content Reduction: keep the windows a query actually needs.

MobileRAG's SCR trims retrieved chunks before generation. Each chunk is split
into overlapping sentence windows, every window is scored against the query, the
best few are kept, and their neighbours are re-attached so the kept text is not
severed mid-argument.

The point is generation cost, not retrieval cost: retrieval has already run and
its results are unchanged. SCR decides how much of what was retrieved is worth
paying a language model to read.

**Three properties this implementation is built around.**

*Reduction must be exactly reproducible from a `QueryRecord`.* Every plot and
statistic derives from the parquet (Section 3, rule 1), so the kept-window
indices and the resulting token count are recorded rather than recomputed.

*Disabled must mean untouched.* `ScrConfig(enabled=False)` returns the input
byte-for-byte. An "off" path that still normalises whitespace would make the
C2 ablation measure the normaliser as well as the reduction.

*The sentence splitter is shared with chunking* (ADR-0012). Chunk boundaries and
window boundaries come from one `pysbd` segmenter, so a chunk cannot be split one
way at index time and another way at query time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smartann.text import Sentence, sentence_windows, split_sentences
from smartann.types import ScrConfig

__all__ = ["ReductionResult", "WindowScore", "reduce_context", "score_windows"]


@dataclass(frozen=True)
class WindowScore:
    """One candidate window: where it is, and how well it matches the query."""

    start_sentence: int
    end_sentence: int  # exclusive
    char_start: int
    char_end: int
    score: float

    @property
    def n_sentences(self) -> int:
        return self.end_sentence - self.start_sentence


@dataclass(frozen=True)
class ReductionResult:
    """What SCR kept, what it dropped, and enough to reproduce the decision."""

    text: str
    kept_windows: tuple[tuple[int, int], ...]  # sentence ranges, after extension
    scores: tuple[float, ...]  # score of each kept window, same order
    n_windows_total: int
    n_sentences_total: int
    n_sentences_kept: int
    chars_in: int
    chars_out: int
    enabled: bool

    @property
    def reduction_ratio(self) -> float:
        """Fraction of characters removed. 0.0 when SCR is off."""
        return 1.0 - (self.chars_out / self.chars_in) if self.chars_in else 0.0

    def approx_tokens(self, chars_per_token: float = 4.0) -> int:
        """Token estimate for `QueryRecord.context_tokens`.

        Characters over a constant, not a real tokenizer: the generator is not
        built (Phase 7 is blocked on MSVC), so a true count is unavailable. The
        divisor is a documented approximation and the field it feeds is named
        `approx_` so nothing downstream reads it as a measurement.
        """
        return round(self.chars_out / chars_per_token)


def score_windows(
    text: str,
    query_vec: np.ndarray,
    encode: object,
    cfg: ScrConfig,
) -> tuple[list[Sentence], list[WindowScore]]:
    """Split *text* into windows and score each against *query_vec*.

    `encode` is any callable taking `list[str]` and returning unit-norm float32
    rows -- normally `Embedder.encode`. It is injected rather than constructed
    here so this module never loads a model, which keeps it importable in tests
    and keeps the embedding cost visible to the caller that must time it.
    """
    sentences = split_sentences(text)
    if not sentences:
        return [], []

    ranges = sentence_windows(sentences, cfg.sliding_window_size, cfg.overlap_size)
    texts = [text[sentences[i].start : sentences[j - 1].end] for i, j in ranges]
    vectors = np.asarray(encode(texts), dtype=np.float32)  # type: ignore[operator]
    q = np.asarray(query_vec, dtype=np.float32).ravel()

    # Both sides are unit-norm at production (ADR-0009), so the inner product is
    # cosine similarity. Larger is better here -- this is a SIMILARITY, not the
    # distance convention the index stores (ADR-0018), and conflating the two
    # would rank every window backwards.
    sims = vectors @ q
    return sentences, [
        WindowScore(
            start_sentence=i,
            end_sentence=j,
            char_start=sentences[i].start,
            char_end=sentences[j - 1].end,
            score=float(s),
        )
        for (i, j), s in zip(ranges, sims, strict=True)
    ]


def reduce_context(
    text: str,
    query_vec: np.ndarray,
    encode: object,
    cfg: ScrConfig,
) -> ReductionResult:
    """Reduce *text* to the windows most relevant to the query."""
    chars_in = len(text)

    if not cfg.enabled:
        # Byte-identical passthrough. The C2 ablation compares against this, so
        # any normalisation here would be measured as part of the reduction.
        sentences = split_sentences(text)
        return ReductionResult(
            text=text,
            kept_windows=((0, len(sentences)),) if sentences else (),
            scores=(),
            n_windows_total=0,
            n_sentences_total=len(sentences),
            n_sentences_kept=len(sentences),
            chars_in=chars_in,
            chars_out=chars_in,
            enabled=False,
        )

    sentences, scored = score_windows(text, query_vec, encode, cfg)
    if not scored:
        return ReductionResult(
            text=text,
            kept_windows=(),
            scores=(),
            n_windows_total=0,
            n_sentences_total=len(sentences),
            n_sentences_kept=len(sentences),
            chars_in=chars_in,
            chars_out=chars_in,
            enabled=True,
        )

    ranked = sorted(scored, key=lambda w: (-w.score, w.start_sentence))
    keep = [w for w in ranked[: cfg.top_windows] if w.score >= cfg.similarity_threshold]
    if not keep:
        # Never return nothing. A threshold that excludes every window means the
        # chunk is a poor match, not that the generator should be handed an empty
        # context -- and an empty context would show up as a generation failure
        # rather than as the retrieval miss it actually is.
        keep = [ranked[0]]

    # Re-attach neighbouring sentences so kept text is not severed mid-argument,
    # then merge overlaps: windows overlap by construction, and emitting a
    # sentence twice would inflate the token count SCR exists to reduce.
    spans: list[tuple[int, int]] = []
    for w in sorted(keep, key=lambda w: w.start_sentence):
        lo = max(0, w.start_sentence - cfg.context_extension_size)
        hi = min(len(sentences), w.end_sentence + cfg.context_extension_size)
        if spans and lo <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], hi))
        else:
            spans.append((lo, hi))

    pieces = [text[sentences[i].start : sentences[j - 1].end] for i, j in spans]
    out = " ".join(pieces)
    return ReductionResult(
        text=out,
        kept_windows=tuple(spans),
        scores=tuple(w.score for w in sorted(keep, key=lambda w: w.start_sentence)),
        n_windows_total=len(scored),
        n_sentences_total=len(sentences),
        n_sentences_kept=sum(j - i for i, j in spans),
        chars_in=chars_in,
        chars_out=len(out),
        enabled=True,
    )

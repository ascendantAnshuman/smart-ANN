"""Exact match and token F1, as SQuAD v1.1 defines them.

Transcribed from the official evaluation script rather than reimplemented from
the paper's prose, because the normalisation is where every reimplementation
diverges: articles are stripped, punctuation is removed, whitespace is
collapsed and case is folded, **in that order**, and F1 is computed over
multisets of whitespace tokens so a repeated token counts twice.

A question carries several gold answers and scores as the **maximum** over
them. That is not leniency -- SQuAD's gold answers are different spans written
by different annotators for the same question, so any one of them is correct
and scoring against only the first would understate every system equally but
not identically.

These are the metrics C2 needs. Until Phase 7 there was no generator, so
context reduction could only be priced in tokens and latency; a reduction that
halves the context and destroys the answer looks identical to a free one on
those axes.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence

__all__ = ["exact_match", "f1", "normalize_answer", "score_answer"]


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 over multisets.

    ``Counter & Counter`` is a multiset intersection, so a token appearing
    twice in both counts twice. Using sets here is the classic error and
    inflates F1 on answers with repeated words.
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    # An empty gold or prediction scores 1.0 only if BOTH are empty. The
    # official script special-cases this; without it, predicting nothing
    # against an empty gold would score 0 and drag the mean down.
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_tokens)
    recall = n_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_answer(prediction: str, gold_answers: Sequence[str]) -> tuple[float, float]:
    """``(exact_match, f1)``, each maximised over the gold answers.

    Raises on an empty gold list rather than returning zero: a question with no
    reference answer cannot be scored, and silently recording 0.0 would report
    a model failure where there is a data failure (Section 3, rule 2).
    """
    if not gold_answers:
        raise ValueError(
            "no gold answers to score against; a question with no reference "
            "cannot yield a metric, and 0.0 would misattribute a data gap to "
            "the model"
        )
    em = max(exact_match(prediction, g) for g in gold_answers)
    token_f1 = max(f1(prediction, g) for g in gold_answers)
    return em, token_f1

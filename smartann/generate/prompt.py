"""Prompt construction, kept in one place so C2's ablation is honest.

Context reduction changes the prompt. If the template also changed between the
reduced and unreduced arms -- a different instruction, a different separator --
the measured quality difference would confound the two. One template, one
join, used by every arm.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["CLOSED_BOOK_TEMPLATE", "TEMPLATE", "VERIFY_TEMPLATE", "build_prompt", "join_chunks"]

# Asked separately from the extractive prompt, and never used by Phase 7, so
# adding it changes no recorded number. Yes/no over a passage and a question is
# a task FLAN-T5 was instruction-tuned on, which is why a small model answers it
# more reliably than it declines inside the extractive prompt.
VERIFY_TEMPLATE = (
    "Read the passages and decide whether they contain the answer to the question.\n\n"
    "Passages:\n{context}\n\n"
    "Question: {question}\n\n"
    "Do the passages contain the answer to the question? Answer yes or no."
)

# No context at all. Whatever comes back is the model's recollection, and the
# only honest use of it is labelled as exactly that.
CLOSED_BOOK_TEMPLATE = "Answer the following question.\n\nQuestion: {question}\nAnswer:"

# Extractive instruction. The model is asked for a span, because SQuAD's gold
# answers ARE spans and EM against a chatty answer measures verbosity rather
# than correctness. "unanswerable" is offered so a model given bad context can
# say so instead of hallucinating -- which is the behaviour C2 must be able to
# detect when reduction removes the answer.
TEMPLATE = (
    "Answer the question using only the context below. "
    "Reply with the shortest exact span from the context that answers it. "
    'If the context does not contain the answer, reply "unanswerable".\n\n'
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)


def join_chunks(chunks: Sequence[str]) -> str:
    """Retrieved chunks as one numbered context block, in rank order.

    Numbering is part of the template for every arm, reduced or not, so a later
    analysis can ask which rank the answer came from without the numbering
    itself becoming a difference between arms.
    """
    # Filter BEFORE numbering. Enumerating the raw list numbers by original
    # position, so a chunk that SCR emptied would leave a gap -- the model
    # would receive "[1] ... [3] ..." and be told, implicitly, that something
    # it cannot see was withheld. Aggressive reduction can empty a chunk, so
    # this is reachable rather than theoretical.
    kept = [c.strip() for c in chunks if c.strip()]
    return "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(kept))


def build_prompt(question: str, chunks: Sequence[str]) -> str:
    """One prompt from a question and the retrieved chunks, in rank order."""
    return TEMPLATE.format(context=join_chunks(chunks), question=question.strip())

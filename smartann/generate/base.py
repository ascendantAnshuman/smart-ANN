"""What a generator returns, and the contract it must satisfy.

Deliberately narrow: the controller never sees the model, only the cost and the
text. Everything measured here is measured the same way as the retrieval
stages, so a generation latency and a search latency are commensurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

__all__ = ["GenerationResult", "Generator"]


@dataclass(frozen=True)
class GenerationResult:
    """One generated answer with its measured cost.

    ``prompt_tokens`` is the tokenised length actually fed to the model, not an
    estimate from character counts. C2 claims that context reduction saves
    generation cost; that claim is only checkable against the number the model
    really consumed, and a whitespace-based estimate diverges from it by the
    tokeniser's compression ratio, which itself varies with the text.
    """

    text: str
    prompt_tokens: int
    generated_tokens: int
    t_generate_ms: float
    truncated: bool


class Generator(Protocol):
    """Anything that turns a question plus context into an answer."""

    name: str

    def generate(self, question: str, context: str) -> GenerationResult: ...

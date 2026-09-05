"""A CPU generator built on `transformers`, which is already a dependency.

**Why this exists at all.** Phase 7 was recorded as blocked on the MSVC C++
Build Tools, because `llama-cpp-python` has no usable wheel on CPython 3.12 /
win-amd64 and must be built from source. That finding is correct and still
stands. The conclusion drawn from it did not: the project needs *a generator*,
not llama.cpp. `sentence-transformers` has been a declared dependency since
Phase 1 and pulls in `torch` and `transformers`, so a generator was installed
and available the whole time. No new dependency is added here (Section 4,
rule 6 is not engaged). See ADR-0043.

**Why seq2seq by default.** FLAN-T5 is instruction-tuned, runs at usable speed
on CPU, and answers extractively without a chat scaffold. A causal model of
comparable size spends its budget on conversational framing that EM punishes,
which would measure the prompt rather than the retrieval.

**Determinism.** Greedy decoding, `do_sample=False`. C2 compares answer quality
across context-reduction settings; sampling noise would sit inside exactly the
difference being measured. This costs diversity, which is not a quantity this
project reports.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from smartann.generate.base import GenerationResult

__all__ = ["GeneratorConfig", "HFGenerator"]


@dataclass(frozen=True)
class GeneratorConfig:
    """Everything that changes an answer, so the config hash covers it."""

    model_id: str = "google/flan-t5-base"
    max_input_tokens: int = 512
    max_new_tokens: int = 32
    num_threads: int = 4
    seed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "max_input_tokens": self.max_input_tokens,
            "max_new_tokens": self.max_new_tokens,
            "num_threads": self.num_threads,
            "seed": self.seed,
            "decoding": "greedy (do_sample=False)",
        }


class HFGenerator:
    """Seq2seq generation on CPU, with the prompt truncation reported.

    Truncation is returned rather than swallowed. A context longer than
    `max_input_tokens` is silently cut by the tokeniser, and if that went
    unrecorded, C2 would read "reduction did not hurt quality" from arms whose
    unreduced context had been truncated to the same length anyway -- the
    reduction would appear free because the comparison never happened.
    """

    def __init__(self, cfg: GeneratorConfig | None = None) -> None:
        self.cfg = cfg or GeneratorConfig()
        self.name = self.cfg.model_id
        self._tokenizer: Any = None
        self._model: Any = None

    def load(self) -> None:
        """Load the model. Separate from ``__init__`` so timing excludes it."""
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        torch.manual_seed(self.cfg.seed)
        torch.set_num_threads(self.cfg.num_threads)
        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.model_id)
        self._model.eval()

    def generate(self, question: str, context: str) -> GenerationResult:
        from smartann.generate.prompt import TEMPLATE

        if self._model is None:
            raise RuntimeError("call load() before generate(); loading is timed separately")
        import torch

        prompt = TEMPLATE.format(context=context, question=question.strip())

        # Tokenise twice: once unbounded to learn the true length, once
        # truncated to feed the model. The difference is what `truncated`
        # reports, and it cannot be recovered afterwards.
        full = self._tokenizer(prompt, return_tensors="pt", truncation=False)
        n_full = int(full["input_ids"].shape[1])
        enc = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_input_tokens,
        )
        n_used = int(enc["input_ids"].shape[1])

        t0 = time.perf_counter()
        with torch.no_grad():
            out = self._model.generate(
                **enc,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        t_ms = (time.perf_counter() - t0) * 1000.0

        text = self._tokenizer.decode(out[0], skip_special_tokens=True).strip()
        return GenerationResult(
            text=text,
            prompt_tokens=n_used,
            generated_tokens=int(out.shape[1]),
            t_generate_ms=t_ms,
            truncated=n_full > n_used,
        )

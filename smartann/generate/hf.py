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
    # Forbid the tokens that make up the "[1]" chunk labels. The context is
    # numbered so an analysis can ask which rank an answer came from, and the
    # model sometimes returns the label itself as the "shortest exact span":
    # 22 of the 4,000 recorded Phase 7 answers were a bare marker, every one
    # scoring zero. Seven vocabulary tokens contain a bracket, and 1 of the
    # 1,000 gold answers contains one (a quotation edit, "wid[en]"), so
    # blocking them can cost at most one question. Off by default so every
    # recorded experiment reproduces byte-for-byte under its recorded config.
    block_markers: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "max_input_tokens": self.max_input_tokens,
            "max_new_tokens": self.max_new_tokens,
            "num_threads": self.num_threads,
            "seed": self.seed,
            "decoding": "greedy (do_sample=False)",
            "block_markers": self.block_markers,
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
        self._blocked: list[list[int]] = []

    def load(self) -> None:
        """Load the model. Separate from ``__init__`` so timing excludes it."""
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        torch.manual_seed(self.cfg.seed)
        torch.set_num_threads(self.cfg.num_threads)
        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.model_id)
        self._model.eval()

        # Read from the vocabulary rather than listed by hand, so a different
        # tokenizer cannot leave a marker token unblocked.
        self._blocked = []
        if self.cfg.block_markers:
            self._blocked = [
                [idx]
                for tok, idx in self._tokenizer.get_vocab().items()
                if "[" in tok or "]" in tok
            ]

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

        extra: dict[str, Any] = {"bad_words_ids": self._blocked} if self._blocked else {}
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self._model.generate(
                **enc,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
                num_beams=1,
                **extra,
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

    def context_answers(self, question: str, context: str) -> tuple[bool, float, float]:
        """Does the model judge that this context answers this question?

        A second, separate question put to the same model, because the
        extractive prompt has a blind spot: asked for "the shortest exact
        span", a small model given passages that mention the right entity but
        not the fact will return the nearest plausible span rather than
        decline. Asked "who is the CEO of Google" over alumni lists, it names a
        Google vice-president.

        Decided by comparing the model's own scores for "yes" and "no" as the
        first output token, so the verdict is the model's argmax and no
        threshold is chosen here. Returns the verdict, the probability the
        model put on "yes" of the two, and the time taken.
        """
        from smartann.generate.prompt import VERIFY_TEMPLATE

        if self._model is None:
            raise RuntimeError("call load() before context_answers()")
        import torch

        prompt = VERIFY_TEMPLATE.format(context=context, question=question.strip())
        enc = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.cfg.max_input_tokens
        )
        yes = self._tokenizer("yes", add_special_tokens=False)["input_ids"][0]
        no = self._tokenizer("no", add_special_tokens=False)["input_ids"][0]
        start = torch.full((1, 1), self._model.config.decoder_start_token_id, dtype=torch.long)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self._model(**enc, decoder_input_ids=start).logits[0, -1]
        t_ms = (time.perf_counter() - t0) * 1000.0

        pair = torch.softmax(torch.stack([logits[yes], logits[no]]), dim=0)
        p_yes = float(pair[0])
        return p_yes > 0.5, p_yes, t_ms

    def closed_book(self, question: str) -> GenerationResult:
        """Answer from the model's own weights, with no retrieved context.

        This is not retrieval-augmented and nothing here checks it. It exists
        for questions the corpus cannot answer, and every caller must label the
        result as the model's recollection: FLAN-T5 was trained on a fixed
        snapshot, so an answer about anything that changes can be stale, and a
        small model's recollection of a fact it rarely saw can be simply wrong.
        """
        from smartann.generate.prompt import CLOSED_BOOK_TEMPLATE

        if self._model is None:
            raise RuntimeError("call load() before closed_book()")
        import torch

        prompt = CLOSED_BOOK_TEMPLATE.format(question=question.strip())
        enc = self._tokenizer(prompt, return_tensors="pt")
        extra: dict[str, Any] = {"bad_words_ids": self._blocked} if self._blocked else {}

        t0 = time.perf_counter()
        with torch.no_grad():
            out = self._model.generate(
                **enc,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
                num_beams=1,
                **extra,
            )
        t_ms = (time.perf_counter() - t0) * 1000.0
        return GenerationResult(
            text=self._tokenizer.decode(out[0], skip_special_tokens=True).strip(),
            prompt_tokens=int(enc["input_ids"].shape[1]),
            generated_tokens=int(out.shape[1]),
            t_generate_ms=t_ms,
            truncated=False,
        )

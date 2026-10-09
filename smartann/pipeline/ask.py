"""One question in, one answer out, with every stage priced.

This is the whole system on a single query: embed, probe, let the controller
choose an effort level, search, reduce the context, generate. Phase 7 already
runs this over 1,000 SQuAD questions and scores them; what was missing was a
way to run it once, on a question someone actually typed, and watch each stage
decide.

**Nothing is reimplemented here.** Retrieval, reduction, prompting and
generation are the same objects Phase 7 drives, so an answer produced by this
path and an answer counted in the C2 results come from the same code. A second
implementation of the query path would be a second system, and only one of them
would have produced the recorded numbers.

**Recall is not reported for a typed question, and the omission is deliberate.**
Recall needs a ground-truth neighbour list, which exists for the 1,000 benchmark
questions and does not exist for anything a user invents. The profile table
carries a recall per effort level, and that number describes the level as
measured over the benchmark set, not the query just run. It is labelled that
way wherever it is shown, because quoting it as this query's recall would be
inventing a measurement (Section 3, rule 1).

**The model can and does say "unanswerable".** The prompt offers that reply so
a generator handed context without the answer can decline instead of
fabricating one, which is the behaviour C2 has to be able to detect when
reduction cuts too deep. A question outside the corpus should land there, and
that is a correct answer rather than a failure of the pipeline.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from smartann.controller.base import Policy
from smartann.controller.policies.budget import BudgetPolicy
from smartann.controller.policies.device import DeviceStatePolicy
from smartann.controller.policies.static import StaticPolicy
from smartann.controller.profile import ProfileEntry, ProfileTable
from smartann.types import (
    AnnConfig,
    Constraints,
    ControllerState,
    DeviceState,
    PerfHistory,
    ScrConfig,
)

__all__ = [
    "CORPUS_LABELS",
    "ORIGIN_KB",
    "ORIGIN_SQUAD",
    "SCR_OFF",
    "SOURCE_CORPUS",
    "SOURCE_MODEL",
    "SOURCE_NONE",
    "AskResult",
    "RagPipeline",
    "RetrievedChunk",
    "Stage",
    "load_profile_table",
]

# Passthrough, byte-identical (ADR-0035). The default for a single typed query:
# reduction is a measured quality cost, and a demonstration should not pay it
# unless it is the thing being demonstrated.
SCR_OFF = ScrConfig(1, 0, 0, 3, 0.0, enabled=False)

# Where a displayed answer came from. Three values, because "the corpus did
# not answer" and "the model then guessed from memory" are different claims
# and a reader must never have to infer which one they are looking at.
SOURCE_CORPUS = "corpus"
SOURCE_NONE = "not_in_corpus"
SOURCE_MODEL = "model_memory"

# Which store a retrieved passage came from. The benchmark index is fixed;
# the knowledge base is what a user adds.
ORIGIN_SQUAD = "squad"
ORIGIN_KB = "kb"

# How each corpus is named to a reader. A dataset with no entry is named after
# itself, which is a dull label but never a false one.
CORPUS_LABELS = {
    ORIGIN_SQUAD: "SQuAD benchmark corpus",
    "demo_phone_care": "phone-care demo corpus",
}

# No history exists on a one-shot query, and the fields say so rather than
# reading as a quiet, healthy machine. BudgetPolicy checks `queries_in_window`
# against its own minimum before trusting any of this, so a zeroed window makes
# it fall back to the table's own p95 instead of scaling by a fabricated drift.
COLD_HISTORY = PerfHistory(
    ewma_latency_ms=float("nan"),
    p50_latency_ms=float("nan"),
    p95_latency_ms=float("nan"),
    p99_latency_ms=float("nan"),
    ewma_energy_mj=float("nan"),
    slo_violation_rate=0.0,
    queries_in_window=0,
    interarrival_ewma_s=float("nan"),
)


def _device_dict(d: DeviceState) -> dict[str, Any]:
    """The device snapshot a report shows, with absence kept as absence.

    Only the fields a policy in this project acts on. The rest of `DeviceState`
    is NaN on every platform measured so far, and listing a column of "n/a"
    beside the ones that matter buries the signal.
    """
    return {
        "temperature_c": d.temperature_c,
        "temperature_slope_c_per_s": d.temperature_slope_c_per_s,
        "battery_pct": d.battery_pct,
        "charging": d.charging,
        "discharge_rate_ma": d.discharge_rate_ma,
        "throttled": d.throttled,
    }


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _find_citation(answer: str, chunks: tuple[RetrievedChunk, ...]) -> int | None:
    """The rank of the first passage that contains the answer verbatim.

    Verbatim, because the prompt asks for "the shortest exact span from the
    context", so a correct answer is literally present in one of the passages.
    Found by string match rather than asked of the model: a model asked which
    passage it used can name the wrong one, and a substring either is there or
    is not. An answer found in no passage gets no citation, which is itself a
    warning worth showing.
    """
    needle = _norm(answer)
    if not needle or needle.startswith("unanswerable"):
        return None
    for c in chunks:
        if needle in _norm(c.reduced_text):
            return c.rank
    return None


def load_profile_table(path: Path, dataset: str) -> ProfileTable:
    """Read a measured profile table, refusing one built on another corpus.

    A table from a different dataset carries latencies and recalls that are
    entirely plausible for this one, which is exactly what makes it dangerous:
    the controller would choose effort from measurements of a different index
    over a different corpus and nothing in the output would look wrong. The
    table names the dataset it was measured on, so that field is checked rather
    than trusted to match.

    Standalone rather than a step inside `RagPipeline.load` so the check can be
    exercised without a built index present.
    """
    table = ProfileTable.read(path)
    if table.dataset != dataset:
        raise ValueError(
            f"{path} was measured on '{table.dataset}' but this pipeline answers from "
            f"'{dataset}'. Effort chosen from another corpus's measurements is not a "
            f"measured choice. Pass a table built for '{dataset}', or omit it to run at "
            f"a fixed nprobe."
        )
    return table


@dataclass(frozen=True)
class Stage:
    """One step of the pipeline and what it cost."""

    name: str
    ms: float
    detail: str = ""


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk the search returned, before and after reduction."""

    rank: int
    chunk_id: int
    distance: float
    text: str
    reduced_text: str
    origin: str = "squad"
    title: str = ""
    url: str = ""
    added_at: str = ""

    @property
    def similarity(self) -> float:
        """Cosine similarity, recovered from hnswlib's convention.

        The squad index is built in `ip` space over unit-norm vectors, where the
        stored distance is `1 - q.v` (`_brute_force` in `index/ecovector.py`).
        Subtracting is exact rather than approximate, but it is only valid for
        this space, which is why it is derived here and not stored as if it were
        a separate measurement. Knowledge-base hits are stored the same way, as
        `1 - cosine`, so one property serves both stores.
        """
        return 1.0 - self.distance

    @property
    def kind_label(self) -> str:
        """How to name this passage's store to a reader, not to the code.

        `origin` carries the DATASET a chunk came from, so a corpus this mapping
        has no entry for is named after itself rather than after SQuAD. The
        label used to be the constant string "SQuAD benchmark corpus" for
        everything that was not the knowledge base, which was true while SQuAD
        was the only index and became a false provenance claim the moment the
        Phase 11 demonstration corpus was added: every authored passage was
        presented to the reader as a benchmark passage.
        """
        if self.origin == ORIGIN_KB:
            return "your knowledge base" if not self.url else "Wikipedia, added by you"
        return CORPUS_LABELS.get(self.origin, f"{self.origin} corpus")

    @property
    def chars_in(self) -> int:
        return len(self.text)

    @property
    def chars_out(self) -> int:
        return len(self.reduced_text)


@dataclass(frozen=True)
class AskResult:
    """An answer, the evidence behind it, and the cost of each stage."""

    question: str
    answer: str
    chunks: tuple[RetrievedChunk, ...]
    ann: AnnConfig
    scr: ScrConfig
    policy_name: str
    policy_reasons: tuple[str, ...]
    profile_entry: ProfileEntry | None
    table_measured_under: str
    budget_ms: float | None
    stages: tuple[Stage, ...]
    controller_overhead_us: float
    prompt_tokens: int
    generated_tokens: int
    truncated: bool
    n_cluster_loads: int
    n_dist_exact: int
    dataset: str
    index_root: Path
    generator_name: str
    # Where the displayed answer came from. Defaults describe the original
    # behaviour, extractive only and unchecked, so a result built before the
    # grounding check existed still reads correctly.
    source: str = "corpus"
    extracted_answer: str = ""
    verify_p_yes: float = float("nan")
    cited_rank: int | None = None
    # The device condition the controller read, and what it cost. `device` is
    # None when no device state was supplied at all, which is different from a
    # state whose every field was NaN: the first means nobody asked, the second
    # means the platform could not answer.
    device: dict[str, Any] | None = None
    device_source: str = ""
    device_is_measured: bool = False
    # What the latency budget alone would have chosen, so the derate is visible
    # as a difference rather than asserted. NaN/None when no comparison ran.
    baseline_nprobe: int | None = None
    baseline_recall: float = float("nan")
    # 12.1: what the grounding guard decided, and the threshold it decided on.
    # None when no guard ran, which is different from a guard that passed.
    guard: dict[str, Any] | None = None
    # 12.2: set when this answer was reused from the semantic cache rather than
    # computed. The stages then describe the lookup, not the original run.
    cache: dict[str, Any] | None = None

    @property
    def derated(self) -> bool:
        """The device state lowered effort below what the budget allowed."""
        return self.baseline_nprobe is not None and self.ann.nprobe < self.baseline_nprobe

    @property
    def device_signals_absent(self) -> bool:
        """The policy could read nothing, as opposed to reading a healthy device.

        An empty reason list means only that no threshold was crossed, which
        happens both on a cool phone and on a laptop with no thermometer at all.
        Those are opposite situations -- one is a measurement, the other is the
        absence of one -- and a report that renders them identically is the
        failure mode Section 3 rule 1 names. This distinguishes them by asking
        whether any signal the policy acts on was finite.
        """
        if self.device is None:
            return False
        return not any(
            isinstance(self.device.get(k), (int, float))
            and math.isfinite(float(self.device.get(k, float("nan"))))
            for k in ("temperature_c", "battery_pct", "temperature_slope_c_per_s")
        )

    @property
    def recall_given_up(self) -> float:
        """Measured recall the derate cost, at the level the table recorded.

        This is the two levels' recalls differenced, both measured over the
        benchmark set. It is not this query's recall, which has no ground truth
        (Section 3, rule 1).
        """
        if self.profile_entry is None or not math.isfinite(self.baseline_recall):
            return float("nan")
        return self.baseline_recall - self.profile_entry.recall

    @property
    def cited(self) -> RetrievedChunk | None:
        """The passage the answer was found in, when it was found verbatim."""
        if self.cited_rank is None:
            return None
        for c in self.chunks:
            if c.rank == self.cited_rank:
                return c
        return None

    @property
    def declined(self) -> bool:
        """The corpus did not answer, whether or not the model then guessed.

        Covers both the extractive prompt replying "unanswerable" and the
        grounding check overruling a span it produced anyway.
        """
        if self.refused_ungrounded:
            return True
        if self.source != SOURCE_CORPUS:
            return True
        return self.answer.strip().lower().startswith("unanswerable")

    @property
    def from_model_memory(self) -> bool:
        """The answer shown is the model's recollection, not a retrieved span."""
        return self.source == SOURCE_MODEL

    @property
    def overruled(self) -> bool:
        """The extractive prompt produced a span that the grounding check rejected."""
        return (
            self.source != SOURCE_CORPUS
            and bool(self.extracted_answer)
            and not self.extracted_answer.lower().startswith("unanswerable")
        )

    @property
    def refused_ungrounded(self) -> bool:
        """The guard refused before the model was ever asked."""
        return self.guard is not None and not bool(self.guard.get("grounded", True))

    @property
    def cache_hit(self) -> bool:
        return self.cache is not None and bool(self.cache.get("hit", False))

    @property
    def best_score(self) -> float:
        return max((c.similarity for c in self.chunks), default=float("nan"))

    @property
    def total_ms(self) -> float:
        return sum(s.ms for s in self.stages)

    @property
    def chars_in(self) -> int:
        return sum(c.chars_in for c in self.chunks)

    @property
    def chars_out(self) -> int:
        return sum(c.chars_out for c in self.chunks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "declined": self.declined,
            "source": self.source,
            "extracted_answer": self.extracted_answer,
            "verify_p_yes": self.verify_p_yes,
            "cited_rank": self.cited_rank,
            "dataset": self.dataset,
            "index_root": str(self.index_root),
            "generator": self.generator_name,
            "policy": {
                "name": self.policy_name,
                "reasons": list(self.policy_reasons),
                "budget_ms": self.budget_ms,
                "overhead_us": self.controller_overhead_us,
                "chosen_nprobe": self.ann.nprobe,
                "table_entry": self.profile_entry.as_dict() if self.profile_entry else None,
                "table_measured_under": self.table_measured_under,
                "baseline_nprobe": self.baseline_nprobe,
                "baseline_recall": self.baseline_recall,
                "derated": self.derated,
                "recall_given_up": self.recall_given_up,
            },
            "device": {
                "state": self.device,
                "source": self.device_source,
                "measured": self.device_is_measured,
                "signals_absent": self.device_signals_absent,
            }
            if self.device is not None
            else None,
            "retrieval": [
                {
                    "rank": c.rank,
                    "chunk_id": c.chunk_id,
                    "similarity": c.similarity,
                    "chars_in": c.chars_in,
                    "chars_out": c.chars_out,
                    "origin": c.origin,
                    "title": c.title,
                    "url": c.url,
                    "added_at": c.added_at,
                    "text": c.reduced_text,
                }
                for c in self.chunks
            ],
            "work": {
                "n_cluster_loads": self.n_cluster_loads,
                "n_dist_exact": self.n_dist_exact,
                "prompt_tokens": self.prompt_tokens,
                "generated_tokens": self.generated_tokens,
                "truncated": self.truncated,
            },
            "timing_ms": {s.name: s.ms for s in self.stages} | {"total": self.total_ms},
        }


class RagPipeline:
    """Loads the index, corpus, embedder and generator once; answers many times.

    Loading is explicit and separate from construction because the generator
    costs seconds to bring up. A caller that wants to time the query path must
    be able to pay that cost outside the measured region, exactly as
    `phase7_qa.py` does.
    """

    def __init__(
        self,
        dataset: str = "squad",
        variant: str = "ecovector",
        k: int = 5,
        profile_table_path: Path | None = None,
        max_input_tokens: int = 1024,
        model_id: str = "google/flan-t5-base",
        use_kb: bool = True,
    ) -> None:
        self.dataset = dataset
        self.variant = variant
        self.k = k
        self.profile_table_path = profile_table_path
        self.max_input_tokens = max_input_tokens
        self.model_id = model_id
        self.use_kb = use_kb
        self._kb: Any = None

        self._index: Any = None
        self._corpus: Any = None
        self._embedder: Any = None
        self._generator: Any = None
        self._table: ProfileTable | None = None
        self.index_root: Path | None = None

    def load(self, load_generator: bool = True) -> RagPipeline:
        from smartann.config import resolve_paths
        from smartann.generate import GeneratorConfig, HFGenerator
        from smartann.generate.corpus import QaCorpus
        from smartann.index.ecovector import EcoVectorIndex
        from smartann.index.storage import resolve_index_root
        from smartann.pipeline.embedder import Embedder

        ip = resolve_index_root(self.dataset, self.variant)
        if ip is None:
            raise FileNotFoundError(
                f"no {self.variant} index for {self.dataset}. Build one with "
                f"`smartann index build configs/index/{self.dataset}_{self.variant}.yaml`; "
                f"this will not answer from an index that does not exist."
            )
        self.index_root = ip.root
        self._index = EcoVectorIndex.open(ip.root)

        paths = resolve_paths()
        proc = Path(paths.data_root) / "processed" / self.dataset
        chunks_db = proc / "chunks.sqlite"
        if not chunks_db.exists():
            raise FileNotFoundError(
                f"{chunks_db} is missing, so retrieved ids cannot be turned back into "
                f"text. Run `smartann data prepare {self.dataset}` first."
            )
        self._corpus = QaCorpus(chunks_db)
        self._embedder = Embedder(cache_path=Path(paths.data_root) / "cache" / "ask.sqlite")
        if self.use_kb:
            from smartann.kb import KnowledgeBase

            self._kb = KnowledgeBase(Path(paths.data_root) / "kb")
        # Force the sentence model up now, outside anything this reports as a
        # cost. Lazily loaded, it lands inside the first `encode` and the embed
        # stage reports ~5,500 ms for work that takes single-digit milliseconds.
        # Same lesson as ADR-0030, and the same one Phase 7 learned when its
        # first reducing arm billed itself for every cache miss in the run.
        #
        # Touching `.model` rather than calling `encode` on a throwaway string:
        # encode consults the cache first, so once that string is cached the
        # warm-up stops loading anything and the load silently moves back into
        # the measured region. A warm-up that quietly stops warming is worse
        # than none, because the number it protects still looks plausible.
        _ = self._embedder.model

        if self.profile_table_path is not None and self.profile_table_path.exists():
            self._table = load_profile_table(self.profile_table_path, self.dataset)

        if load_generator:
            # 1024 rather than the model's nominal 512, for the reason Phase 7
            # records: at k=5 an unreduced context reaches ~590 tokens, and at
            # 512 two thirds of it was silently truncated.
            # Markers blocked here and not in Phase 7's recorded runs, so those
            # still reproduce under their recorded config.
            self._generator = HFGenerator(
                GeneratorConfig(
                    model_id=self.model_id,
                    max_input_tokens=self.max_input_tokens,
                    block_markers=True,
                )
            )
            self._generator.load()
        return self

    @property
    def table(self) -> ProfileTable | None:
        return self._table

    def _policy(
        self,
        budget_ms: float | None,
        nprobe: int,
        scr: ScrConfig,
        device_aware: bool = False,
        query_fit: Any = None,
    ) -> Policy:
        """A budget policy when a budget and a measured table are both present.

        Without the table there is nothing to choose from that was ever
        measured, so the honest fallback is the static configuration the caller
        asked for rather than a guess at what would fit.

        `device_aware` swaps in `DeviceStatePolicy`, which composes over the
        same table and can only ever lower the level the budget allowed. It is
        not the default: on a host with no thermometer it would read NaN and
        decide nothing, and a policy that silently does nothing is worse than
        one the caller chose not to run.
        """
        base = AnnConfig(nprobe=nprobe, ef_centroid=128, ef_list=64, k=self.k, per_cluster_k=32)
        if budget_ms is not None and self._table is not None:
            # Device state takes precedence: a shield that a difficulty signal
            # could override is not a shield. The two are not combined here
            # because no experiment has measured them together, and composing
            # two policies produces a third whose behaviour nobody has scored.
            if device_aware:
                return DeviceStatePolicy(self._table, base, scr)
            if query_fit is not None:
                from smartann.controller.policies.query import QueryAdaptivePolicy

                return QueryAdaptivePolicy(self._table, base, scr, query_fit)
            return BudgetPolicy(self._table, base, scr)
        return StaticPolicy(base, scr, tuning_note=f"nprobe={nprobe} as given")

    def ask(
        self,
        question: str,
        budget_ms: float | None = None,
        nprobe: int = 8,
        scr: ScrConfig | None = None,
        verify: bool = True,
        fallback: bool = False,
        device: DeviceState | None = None,
        device_source: str = "",
        device_is_measured: bool = False,
        query_fit: Any = None,
        guard: Any = None,
        cache: Any = None,
    ) -> AskResult:
        """Answer one question, recording what each stage cost.

        Passing `device` switches the controller to `DeviceStatePolicy`, so the
        effort level follows the handset's condition as well as the latency
        budget. `device_is_measured` must be True only when that state came from
        a recorded trace or a live sensor; a hypothetical must be carried
        through as a hypothetical so no report can present it as a measurement.
        """
        if self._index is None or self._corpus is None:
            raise RuntimeError("call load() before ask()")
        if not question.strip():
            raise ValueError("an empty question has no answer to retrieve")

        from smartann.generate.prompt import join_chunks
        from smartann.scr.reduce import reduce_context

        scr = scr or SCR_OFF
        stages: list[Stage] = []

        # Whether this question was already embedded decides whether the stage
        # below measures inference or a SQLite read, and those differ by two
        # orders of magnitude. Asked before reading, so the answer is about the
        # state going in rather than the state the read leaves behind.
        cached = self._question_is_cached(question)

        t0 = time.perf_counter()
        vec = self._embedder.encode([question])[0].astype(np.float32)
        stages.append(
            Stage(
                "embed",
                (time.perf_counter() - t0) * 1000.0,
                f"{self._embedder.model_name}, {'cache hit' if cached else 'fresh inference'}",
            )
        )

        # 12.2: a question already answered in other words. Checked after the
        # embedding, because the embedding IS the key, and before the controller,
        # because a reused answer needs no effort decision at all.
        if cache is not None:
            t0 = time.perf_counter()
            hit = cache.lookup(vec)
            lookup_ms = (time.perf_counter() - t0) * 1000.0
            if hit is not None:
                stages.append(
                    Stage(
                        "cache",
                        lookup_ms,
                        f"hit at similarity {hit.similarity:.3f} on {hit.question!r}",
                    )
                )
                # `replace` on a frozen dataclass; typed through a local so the
                # cache can stay generic over what it stores.
                reused: AskResult = hit.payload
                return replace(
                    reused,
                    question=question.strip(),
                    stages=tuple(stages),
                    cache=hit.as_dict(),
                    # The device state belongs to THIS query, not to the run
                    # that filled the cache. The answer is reused; the
                    # conditions it is being served under are not.
                    device=_device_dict(device) if device is not None else None,
                    device_source=device_source,
                    device_is_measured=device_is_measured,
                )
            stages.append(Stage("cache", lookup_ms, "miss"))

        policy = self._policy(
            budget_ms, nprobe, scr, device_aware=device is not None, query_fit=query_fit
        )
        # The probe must be wide enough for any level the policy might pick, or
        # a decision to raise effort could not be carried out with the clusters
        # already in hand. This mirrors `_widest_ann` in the bench runner.
        widest = max(
            (e.nprobe for e in self._table.entries) if self._table else (nprobe,),
            default=nprobe,
        )
        probe_ann = AnnConfig(
            nprobe=max(widest, nprobe),
            ef_centroid=128,
            ef_list=64,
            k=self.k,
            per_cluster_k=32,
        )

        t0 = time.perf_counter()
        probe = self._index.probe(vec, probe_ann, token_len=len(question.split()))
        stages.append(
            Stage(
                "probe",
                (time.perf_counter() - t0) * 1000.0,
                f"{probe_ann.nprobe} centroids, gap {probe.features.centroid_dist_gap:.3f}",
            )
        )

        state = ControllerState(
            device=device if device is not None else self._device_state(),
            query=probe.features,
            history=COLD_HISTORY,
        )
        constraints = Constraints(
            latency_target_ms=budget_ms if budget_ms is not None else float("inf"),
            latency_hard_ms=budget_ms if budget_ms is not None else float("inf"),
            energy_budget_mj=float("inf"),
            min_recall=None,
        )
        decision = policy.decide_timed(state, constraints)
        ann = decision.config.ann

        # What the budget alone would have picked, from the same measured table.
        # Recorded so a derate is shown as a difference between two real levels
        # rather than asserted in prose, and so "the device changed nothing"
        # is distinguishable from "no device policy ran".
        entry = self._table.best_within(budget_ms) if (self._table and budget_ms) else None
        baseline_nprobe: int | None = None
        baseline_recall = float("nan")
        if (device is not None or query_fit is not None) and entry is not None:
            baseline_nprobe = entry.nprobe
            baseline_recall = entry.recall
            # The report must describe the level actually searched, not the one
            # the budget would have allowed, so the entry follows the decision.
            if self._table is not None:
                entry = self._table.entry_for(ann.nprobe) or entry

        # Visit only the nearest `nprobe` of the clusters the probe returned.
        # Without this the search opens every cluster the WIDE probe found while
        # the report says the controller chose eight, so the decision is printed
        # but never acted on -- an adaptive system that is only adaptive on
        # paper. `run_queries` narrows the same way and for the same reason.
        from smartann.index.ecovector import ProbeResult

        want = min(ann.nprobe, probe.cluster_ids.size)
        narrowed = ProbeResult(
            cluster_ids=probe.cluster_ids[:want],
            centroid_dists=probe.centroid_dists[:want],
            features=probe.features,
            t_centroid_ms=probe.t_centroid_ms,
            n_dist_modeled=probe.n_dist_modeled,
        )

        t0 = time.perf_counter()
        res = self._index.finish(vec, narrowed, ann)
        stages.append(
            Stage(
                "search",
                (time.perf_counter() - t0) * 1000.0,
                f"nprobe={ann.nprobe}, {res.n_cluster_loads} cluster loads",
            )
        )

        ids = res.ids[: self.k].tolist()
        dists = res.dists[: self.k].tolist()
        texts = self._corpus.chunk_texts(ids)
        titles_fn = getattr(self._corpus, "chunk_titles", None)
        titles = titles_fn(ids) if titles_fn is not None else [""] * len(ids)

        # Candidates from both stores, ranked together by cosine similarity.
        # Both sides are gte-small on unit vectors, so the scores are the same
        # quantity and a merge by score is a fair comparison, not a heuristic.
        candidates: list[tuple[float, str, int, str, str, str, str]] = [
            # The dataset name, not a constant: a chunk's origin must say which
            # corpus it came from, and this path serves more than one now.
            (1.0 - float(d), self.dataset, int(cid), t, title, "", "")
            for cid, d, t, title in zip(ids, dists, texts, titles, strict=True)
        ]
        if self._kb is not None and len(self._kb):
            t0 = time.perf_counter()
            hits = self._kb.search(vec, self.k)
            stages.append(
                Stage(
                    "kb",
                    (time.perf_counter() - t0) * 1000.0,
                    f"exact search over {len(self._kb):,} added chunks",
                )
            )
            candidates += [
                (h.similarity, ORIGIN_KB, h.chunk_id, h.text, h.title, h.url, h.added_at)
                for h in hits
            ]
        candidates.sort(key=lambda c: -c[0])
        candidates = candidates[: self.k]

        t0 = time.perf_counter()
        reduced = [
            reduce_context(c[3], vec, self._embedder.encode, scr).text if scr.enabled else c[3]
            for c in candidates
        ]
        stages.append(
            Stage(
                "reduce",
                (time.perf_counter() - t0) * 1000.0,
                "passthrough (SCR off)" if not scr.enabled else f"top_windows={scr.top_windows}",
            )
        )

        chunks = tuple(
            RetrievedChunk(
                rank=i + 1,
                chunk_id=cid,
                distance=1.0 - sim,
                text=text,
                reduced_text=r,
                origin=origin,
                title=title,
                url=url,
                added_at=added,
            )
            for i, ((sim, origin, cid, text, title, url, added), r) in enumerate(
                zip(candidates, reduced, strict=True)
            )
        )

        # 12.1: decide whether the corpus answered at all, BEFORE paying for
        # generation. A refusal reached after the model has written something
        # costs the most expensive stage and then discards it.
        guard_dict: dict[str, Any] | None = None
        if guard is not None:
            best = max((c.similarity for c in chunks), default=float("nan"))
            verdict = guard.verdict(best)
            guard_dict = verdict.as_dict()
            if not verdict.grounded:
                from smartann.pipeline.guard import REFUSAL

                stages.append(Stage("guard", 0.0, verdict.reason))
                return AskResult(
                    question=question.strip(),
                    answer=REFUSAL,
                    chunks=chunks,
                    ann=ann,
                    scr=scr,
                    policy_name=policy.name,
                    policy_reasons=tuple(getattr(policy, "reasons", lambda: [])()),
                    profile_entry=entry,
                    table_measured_under=self._table.measured_under if self._table else "",
                    budget_ms=budget_ms,
                    stages=tuple(stages),
                    controller_overhead_us=decision.overhead_us,
                    prompt_tokens=0,
                    generated_tokens=0,
                    truncated=False,
                    n_cluster_loads=int(res.n_cluster_loads),
                    n_dist_exact=int(res.n_dist_exact),
                    dataset=self.dataset,
                    index_root=self.index_root or Path(),
                    generator_name=(
                        self._generator.cfg.model_id if self._generator is not None else ""
                    ),
                    source=SOURCE_NONE,
                    extracted_answer="",
                    verify_p_yes=float("nan"),
                    cited_rank=None,
                    device=_device_dict(device) if device is not None else None,
                    device_source=device_source,
                    device_is_measured=device_is_measured,
                    baseline_nprobe=baseline_nprobe,
                    baseline_recall=baseline_recall,
                    guard=guard_dict,
                )

        if self._generator is None:
            raise RuntimeError("load(load_generator=True) is required before generating an answer")
        context = join_chunks([c.reduced_text for c in chunks])
        g = self._generator.generate(question, context)
        stages.append(Stage("generate", g.t_generate_ms, f"{g.prompt_tokens} prompt tokens"))
        extracted = g.text.strip()

        # The extractive prompt returns a span whenever it can find one, so a
        # passage that mentions the right entity but not the fact still yields
        # a confident-looking answer. The grounding check asks the model the
        # question it is better at: do these passages contain the answer at all.
        p_yes = float("nan")
        grounded = not extracted.lower().startswith("unanswerable")
        if verify and grounded:
            ok, p_yes, t_verify = self._generator.context_answers(question, context)
            stages.append(Stage("verify", t_verify, f"p(contains answer) = {p_yes:.2f}"))
            grounded = ok

        answer, source, fallback_text = extracted, SOURCE_CORPUS, ""
        if not grounded:
            answer, source = "not found in the corpus", SOURCE_NONE
            if fallback:
                fb = self._generator.closed_book(question)
                stages.append(Stage("recall", fb.t_generate_ms, "closed book, no passages"))
                fallback_text = fb.text.strip()
                answer, source = fallback_text, SOURCE_MODEL

        result = AskResult(
            question=question.strip(),
            answer=answer,
            chunks=chunks,
            ann=ann,
            scr=scr,
            policy_name=policy.name,
            policy_reasons=tuple(getattr(policy, "reasons", lambda: [])()),
            profile_entry=entry,
            table_measured_under=self._table.measured_under if self._table else "",
            budget_ms=budget_ms,
            stages=tuple(stages),
            controller_overhead_us=decision.overhead_us,
            prompt_tokens=g.prompt_tokens,
            generated_tokens=g.generated_tokens,
            truncated=g.truncated,
            n_cluster_loads=int(res.n_cluster_loads),
            n_dist_exact=int(res.n_dist_exact),
            dataset=self.dataset,
            index_root=self.index_root or Path(),
            generator_name=self._generator.cfg.model_id,
            source=source,
            extracted_answer=extracted,
            verify_p_yes=p_yes,
            cited_rank=_find_citation(answer, chunks) if source == SOURCE_CORPUS else None,
            device=_device_dict(device) if device is not None else None,
            device_source=device_source,
            device_is_measured=device_is_measured,
            baseline_nprobe=baseline_nprobe,
            baseline_recall=baseline_recall,
            guard=guard_dict,
        )
        if cache is not None:
            # Stored with what this run cost, so a later hit can report what it
            # avoided rather than estimating it.
            cache.store(vec, question.strip(), result, result.total_ms)
        return result

    def _question_is_cached(self, question: str) -> bool:
        """Is this question's vector already on disk?

        Reported rather than inferred from the elapsed time, because a fast
        embed and a cached embed look identical in a timing column and only one
        of them means the model did any work.
        """
        from smartann.pipeline.embedder import _cache_key

        cache = getattr(self._embedder, "cache", None)
        if cache is None:
            return False
        key = _cache_key(self._embedder.model_name, question)
        return bool(cache.get_many([key]))

    def _device_state(self) -> Any:
        """Live telemetry where the platform provides it, NaN where it does not.

        A failure to read the device is not a cold device. `unavailable_state`
        exists so the difference stays visible, and it is what a policy needs in
        order to decline to act rather than act on a zero.
        """
        from smartann.telemetry import unavailable_state

        try:
            from smartann.telemetry.windows import WindowsMonitor

            return WindowsMonitor().sample()
        except Exception:
            return unavailable_state()

    def close(self) -> None:
        if self._corpus is not None:
            self._corpus.close()

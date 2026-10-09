"""12.2 -- reuse the answer to a question that was already asked, in other words.

An exact-match cache on the question string catches almost nothing, because
people do not retype a question the same way twice. A semantic cache compares
the query *embedding* instead, so "why does my phone get hot when it's not
charging" and "what makes my phone warm while unplugged" hit the same entry.

**The threshold is calibrated, not guessed.** `scripts/phase12_calibrate_cache.py`
measures two distributions over the Phase 11.4 query set: the similarity between
queries that are paraphrases of each other, and the similarity between queries
that are not. A threshold between the two reuses an answer only for questions
that genuinely mean the same thing. The number and both distributions are
written to `results/cache_threshold.json`, so the constant in a config file is
traceable to the measurement that produced it.

**Why too low a threshold is dangerous rather than merely wrong.** A cache hit
does not degrade an answer, it replaces it. Two questions that are similar but
not the same -- "why is my phone hot" and "why is my phone NOT charging" --
would be answered identically, confidently, with a citation, and nothing in the
output would look wrong. That is worse than a miss, which costs only time.

**LRU and bounded.** A demonstration runs for minutes and a phone has finite
memory; an unbounded cache is a leak with good manners. Eviction is by least
recent *use* rather than least recent insertion, so the question an audience
keeps asking stays resident.

**What is reused, and what is not.** A hit returns the stored answer and
passages. It does not return the stored device state, latency or energy, because
those described the run that filled the cache and not this one. The saving is
reported as what the earlier run cost, which is what was avoided.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np

__all__ = ["CacheHit", "SemanticCache"]

T = TypeVar("T")


@dataclass(frozen=True)
class CacheHit(Generic[T]):
    """A reused answer, and what reusing it avoided."""

    payload: T
    question: str
    similarity: float
    threshold: float
    saved_ms: float
    saved_energy_mj: float
    age_queries: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "hit": True,
            "matched_question": self.question,
            "similarity": self.similarity,
            "threshold": self.threshold,
            "saved_ms": self.saved_ms,
            "saved_energy_mj": self.saved_energy_mj,
            "age_queries": self.age_queries,
            "note": (
                "latency and energy here are what the ORIGINAL run of this question "
                "cost; they are what was avoided, not what this query spent."
            ),
        }


@dataclass
class _Entry(Generic[T]):
    vector: np.ndarray
    question: str
    payload: T
    cost_ms: float
    energy_mj: float
    stored_at: int


class SemanticCache(Generic[T]):
    """Bounded, LRU, keyed by embedding similarity rather than by string."""

    def __init__(self, threshold: float = 0.95, max_entries: int = 64) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        if max_entries < 1:
            raise ValueError(f"max_entries must be at least 1, got {max_entries}")
        self.threshold = threshold
        self.max_entries = max_entries
        self._entries: OrderedDict[str, _Entry[T]] = OrderedDict()
        self._clock = 0
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else float("nan")

    def lookup(self, vector: np.ndarray) -> CacheHit[T] | None:
        """The stored answer for the nearest question above threshold, or None.

        Nearest rather than first-above-threshold: with several near-duplicates
        resident, answering with whichever happened to be inserted first would
        make the result depend on insertion order.
        """
        self._clock += 1
        if not self._entries:
            self.misses += 1
            return None

        q = np.asarray(vector, dtype=np.float32).reshape(-1)
        best_key, best_sim = "", -math.inf
        for key, entry in self._entries.items():
            sim = float(entry.vector @ q)
            if sim > best_sim:
                best_key, best_sim = key, sim

        if best_sim < self.threshold:
            self.misses += 1
            return None

        entry = self._entries[best_key]
        self._entries.move_to_end(best_key)  # least-recently-USED, not inserted
        self.hits += 1
        return CacheHit(
            payload=entry.payload,
            question=entry.question,
            similarity=best_sim,
            threshold=self.threshold,
            saved_ms=entry.cost_ms,
            saved_energy_mj=entry.energy_mj,
            age_queries=self._clock - entry.stored_at,
        )

    def store(
        self,
        vector: np.ndarray,
        question: str,
        payload: T,
        cost_ms: float,
        energy_mj: float = float("nan"),
    ) -> None:
        """Remember this answer. Vectors are normalised so lookup is a dot product."""
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(v))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("a zero or non-finite query vector cannot be cached")
        v = v / norm

        key = question.strip().lower()
        if key in self._entries:
            del self._entries[key]
        self._entries[key] = _Entry(v, question, payload, cost_ms, energy_mj, self._clock)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)  # evict the least recently used

    def clear(self) -> None:
        self._entries.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict[str, Any]:
        return {
            "entries": len(self._entries),
            "max_entries": self.max_entries,
            "threshold": self.threshold,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
        }

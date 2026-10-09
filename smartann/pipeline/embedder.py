"""Sentence embedding with an on-disk cache.

Pulled forward from Phase 7 because the QA chunk index cannot be prepared
without it. The interface is deliberately minimal for now -- ONNX export and the
batching strategy stay in Phase 7.

gte-small is a **cosine** model, so every vector this returns is L2-normalised
at the point of production. Normalising at query time instead would let the ANN
index and the ground truth disagree about the metric, which is the failure mode
ADR-0010 exists to prevent.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["DEFAULT_MODEL", "Embedder", "EmbeddingCache"]

DEFAULT_MODEL = "thenlper/gte-small"
DEFAULT_DIM = 384


def _cache_key(model_name: str, text: str) -> str:
    """SHA-256 over model name and text, so caches cannot collide across models."""
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


class EmbeddingCache:
    """SQLite-backed vector cache.

    One database rather than one file per vector: a 50k-chunk corpus would
    otherwise create 50k tiny files, which NTFS handles poorly and OneDrive
    handles worse.
    """

    def __init__(self, path: str | Path, dim: int) -> None:
        self.path = Path(path)
        self.dim = dim
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Shared across threads, and that needs BOTH of these.
        #
        # check_same_thread=False, because this connection outlives the thread that
        # opened it: a Streamlit run, or any server reusing a loaded pipeline, hands
        # the same object to a different worker thread on the next interaction, and
        # SQLite's guard then raises on the SECOND question rather than the first.
        #
        # And a lock, because disabling that guard is not on its own sufficient.
        # sqlite3.threadsafety is 3, which makes the *library* serialised; it does
        # not make one Connection safe to drive from two Python threads at once.
        # Four concurrent readers against this cache raised "not enough values to
        # unpack (expected 2, got 0)" -- a fetch landing on another thread's
        # statement. Every use of the connection therefore takes `self._lock`,
        # which is reentrant because the write paths call read helpers that
        # take it again.
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vec BLOB)")
        self._conn.commit()

    def get_many(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        chunk = 900  # stay under SQLite's default variable limit
        with self._lock:
            for i in range(0, len(keys), chunk):
                block = keys[i : i + chunk]
                placeholders = ",".join("?" * len(block))
                rows = self._conn.execute(
                    f"SELECT key, vec FROM vectors WHERE key IN ({placeholders})", tuple(block)
                ).fetchall()
                for key, blob in rows:
                    found[key] = np.frombuffer(blob, dtype=np.float32)
        return found

    def put_many(self, items: Sequence[tuple[str, np.ndarray]]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO vectors (key, vec) VALUES (?, ?)",
                [(k, np.ascontiguousarray(v, dtype=np.float32).tobytes()) for k, v in items],
            )
            self._conn.commit()

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0])

    def close(self) -> None:
        self._conn.close()


class Embedder:
    """Batched, cached sentence embedding.

    The model is loaded lazily so that constructing an ``Embedder`` in a test or
    a ``data status`` call does not pay a multi-second import and model load.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_path: str | Path | None = None,
        device: str = "cpu",
        dim: int = DEFAULT_DIM,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.dim = dim
        self._model: Any = None
        self.cache = EmbeddingCache(cache_path, dim) if cache_path is not None else None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            actual = int(self._model.get_sentence_embedding_dimension())
            if actual != self.dim:
                raise ValueError(
                    f"{self.model_name} produces {actual}-d embeddings but the registry "
                    f"expects {self.dim}-d"
                )
        return self._model

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Embed *texts*, returning ``(len(texts), dim)`` float32, L2-normalised.

        Cache hits are served without loading the model at all, so re-running a
        preparation step costs no GPU/CPU inference.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        keys = [_cache_key(self.model_name, t) for t in texts]
        cached = self.cache.get_many(keys) if self.cache is not None else {}

        missing_idx = [i for i, k in enumerate(keys) if k not in cached]
        if missing_idx:
            fresh = self.model.encode(
                [texts[i] for i in missing_idx],
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
            ).astype(np.float32)
            if self.cache is not None:
                self.cache.put_many([(keys[i], fresh[j]) for j, i in enumerate(missing_idx)])
            for j, i in enumerate(missing_idx):
                cached[keys[i]] = fresh[j]

        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, k in enumerate(keys):
            out[i] = cached[k]

        # Cached vectors were normalised when produced, but a cache written by an
        # older build might not have been; verify rather than assume.
        norms = np.linalg.norm(out, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            worst = float(np.max(np.abs(norms - 1.0)))
            raise ValueError(
                f"embeddings are not unit-norm (max deviation {worst:.2e}); "
                "the cache may predate normalisation being enforced"
            )
        return out

    def close(self) -> None:
        if self.cache is not None:
            self.cache.close()

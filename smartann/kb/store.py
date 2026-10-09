"""A knowledge base the user can grow, searched beside the SQuAD index.

**Why a second store rather than new rows in the SQuAD index.** Every recorded
result in this project -- recall per effort level, the Phase 7 quality ladder,
the device profile table -- was measured against one fixed index. Adding
documents to it would change the recall of every level and quietly invalidate
all of them. A separate store keeps the benchmark index byte-identical, so its
numbers stay reproducible, while `ask` can still answer from new material.

**Why exact search here.** The knowledge base is small: a dozen Wikipedia
articles is a few thousand chunks, and a brute-force dot product over that many
384-dimensional vectors takes well under a millisecond. An approximate index
would add a recall below 1.0 to the one store where there is no ground truth to
measure it against. Exact search has recall 1.0 by construction, so there is
nothing to report and nothing to get wrong.

**Comparable scores by construction.** The same embedder (gte-small) produces
both the SQuAD vectors and these, both are L2-normalised, and both indexes rank
by cosine. A similarity from one is therefore directly comparable with a
similarity from the other, which is what lets the two result lists be merged by
score instead of interleaved by rule.

**Chunked exactly like SQuAD.** Three-sentence windows with one sentence of
overlap, through the same `sentence_windows` function, so a passage from a new
article and a passage from the benchmark are the same shape of evidence.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from smartann.text import sentence_windows, split_sentences

__all__ = ["KbHit", "KnowledgeBase", "SourceInfo", "chunk_paragraphs"]

WINDOW = 3
OVERLAP = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    source_id  INTEGER PRIMARY KEY,
    title      TEXT NOT NULL,
    url        TEXT NOT NULL,
    kind       TEXT NOT NULL,
    added_at   TEXT NOT NULL,
    n_chunks   INTEGER NOT NULL,
    n_chars    INTEGER NOT NULL,
    text_sha   TEXT NOT NULL,
    UNIQUE (title, url)
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   INTEGER PRIMARY KEY,
    source_id  INTEGER NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
    text       TEXT NOT NULL,
    vector     BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_by_source ON chunks(source_id);
"""


@dataclass(frozen=True)
class SourceInfo:
    """One document that was added, and when."""

    source_id: int
    title: str
    url: str
    kind: str
    added_at: str
    n_chunks: int
    n_chars: int


@dataclass(frozen=True)
class KbHit:
    """One retrieved chunk, with where it came from."""

    chunk_id: int
    similarity: float
    text: str
    title: str
    url: str
    kind: str
    added_at: str


def chunk_paragraphs(paragraphs: Sequence[str]) -> list[str]:
    """Sentence windows within each paragraph, never across two.

    A window spanning a paragraph break would glue the end of one topic to the
    start of the next, and the embedding of that chunk would describe neither.
    SQuAD contexts are single paragraphs for the same reason.
    """
    out: list[str] = []
    for para in paragraphs:
        text = " ".join(para.split())
        if not text:
            continue
        sents = split_sentences(text)
        for i, j in sentence_windows(sents, WINDOW, OVERLAP):
            chunk = " ".join(s.text.strip() for s in sents[i:j]).strip()
            if chunk:
                out.append(chunk)
    return out


class KnowledgeBase:
    """SQLite-backed chunks and vectors, loaded into memory for exact search."""

    def __init__(self, root: Path, dim: int = 384) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dim = dim
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
        # Concurrent readers against a shared connection have been observed raising
        # "not enough values to unpack" -- a fetch landing on another thread's
        # statement. Every use of the connection therefore takes `self._lock`,
        # which is reentrant because the write paths call read helpers that
        # take it again.
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.root / "kb.sqlite"), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._ids = np.zeros(0, dtype=np.int64)
        self._matrix = np.zeros((0, dim), dtype=np.float32)
        self._reload()

    def _reload(self) -> None:
        """Pull every vector into one matrix; search is then a single matmul."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT chunk_id, vector FROM chunks ORDER BY chunk_id"
            ).fetchall()
        if not rows:
            self._ids = np.zeros(0, dtype=np.int64)
            self._matrix = np.zeros((0, self.dim), dtype=np.float32)
            return
        self._ids = np.array([int(r["chunk_id"]) for r in rows], dtype=np.int64)
        self._matrix = np.stack(
            [np.frombuffer(r["vector"], dtype=np.float32) for r in rows]
        ).astype(np.float32)

    def __len__(self) -> int:
        return int(self._ids.size)

    def add(
        self,
        title: str,
        paragraphs: Sequence[str],
        url: str,
        kind: str,
        encode: Callable[[list[str]], np.ndarray],
    ) -> SourceInfo:
        """Chunk, embed and store one document, replacing any earlier copy.

        Re-adding the same title and URL replaces it rather than duplicating
        it. Two copies of one article would take two of the five retrieval
        slots with the same passage, and crowd out the evidence a question
        actually needed.
        """
        chunks = chunk_paragraphs(paragraphs)
        if not chunks:
            raise ValueError(f"'{title}' produced no text to index")

        vecs = np.asarray(encode(chunks), dtype=np.float32)
        if vecs.shape != (len(chunks), self.dim):
            raise ValueError(
                f"embedder returned {vecs.shape} for {len(chunks)} chunks; expected "
                f"({len(chunks)}, {self.dim})"
            )
        # Normalise defensively: the merge with the SQuAD index is only a fair
        # comparison if both sides are cosine on unit vectors.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.maximum(norms, 1e-12)

        body = "\n".join(chunks)
        sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
        added = datetime.now(UTC).astimezone().strftime("%Y-%m-%d %H:%M")
        n_chars = sum(len(p) for p in paragraphs)

        with self._lock, self._conn:
            self._conn.execute("DELETE FROM sources WHERE title = ? AND url = ?", (title, url))
            cur = self._conn.execute(
                "INSERT INTO sources (title, url, kind, added_at, n_chunks, n_chars, text_sha) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (title, url, kind, added, len(chunks), n_chars, sha),
            )
            source_id = int(cur.lastrowid or 0)
            self._conn.executemany(
                "INSERT INTO chunks (source_id, text, vector) VALUES (?, ?, ?)",
                [
                    (source_id, c, np.ascontiguousarray(v, dtype=np.float32).tobytes())
                    for c, v in zip(chunks, vecs, strict=True)
                ],
            )
        self._reload()
        return SourceInfo(source_id, title, url, kind, added, len(chunks), n_chars)

    def search(self, query_vec: np.ndarray, k: int) -> list[KbHit]:
        """The k most similar chunks, exactly. Empty when nothing has been added."""
        if not len(self) or k <= 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        sims = self._matrix @ q
        top = np.argsort(-sims)[: min(k, sims.size)]
        ids = [int(self._ids[i]) for i in top]
        by_id = self._rows(ids)
        return [
            KbHit(
                chunk_id=cid,
                similarity=float(sims[i]),
                text=by_id[cid]["text"],
                title=by_id[cid]["title"],
                url=by_id[cid]["url"],
                kind=by_id[cid]["kind"],
                added_at=by_id[cid]["added_at"],
            )
            for i, cid in zip(top, ids, strict=True)
        ]

    def _rows(self, ids: list[int]) -> dict[int, sqlite3.Row]:
        placeholders = ",".join("?" * len(ids))
        with self._lock:
            rows = self._conn.execute(
                "SELECT c.chunk_id, c.text, s.title, s.url, s.kind, s.added_at FROM chunks c "
                "JOIN sources s ON s.source_id = c.source_id "
                f"WHERE c.chunk_id IN ({placeholders})",
                tuple(ids),
            ).fetchall()
        return {int(r["chunk_id"]): r for r in rows}

    def sources(self) -> list[SourceInfo]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT source_id, title, url, kind, added_at, n_chunks, n_chars FROM sources "
                "ORDER BY added_at DESC, title"
            ).fetchall()
        return [
            SourceInfo(
                int(r["source_id"]),
                str(r["title"]),
                str(r["url"]),
                str(r["kind"]),
                str(r["added_at"]),
                int(r["n_chunks"]),
                int(r["n_chars"]),
            )
            for r in rows
        ]

    def remove(self, title: str) -> int:
        """Delete every source with this title. Returns how many were removed."""
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM sources WHERE title = ?", (title,))
        self._reload()
        return int(cur.rowcount)

    def close(self) -> None:
        self._conn.close()

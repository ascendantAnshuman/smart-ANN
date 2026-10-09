"""Reading the SQuAD side of ``chunks.sqlite``: questions, golds, chunk text.

Retrieval returns integer ids. Generation needs the text behind them and the
reference answers to score against, and both were written by Phase 1 -- the
`queries` table has carried a JSON list of gold answer strings since the corpus
was built. Nothing new is prepared here.

The id mapping is the part worth stating. `base.npy` row *i* is the embedding
of `metadata.chunk_id = i`, so a retrieved id indexes both directly. That
identity is asserted on open rather than assumed, because a mismatch would
silently pair every answer with the wrong context and show up only as
uniformly bad quality -- indistinguishable from a weak model.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = ["QaCorpus", "QaQuestion"]


@dataclass(frozen=True)
class QaQuestion:
    query_id: int
    squad_id: str
    question: str
    answers: tuple[str, ...]
    gold_doc_id: int


class QaCorpus:
    """Chunk text and gold answers for a QA dataset."""

    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} does not exist; run `smartann data prepare squad` first"
            )
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
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._assert_contiguous_chunk_ids()

    def _assert_contiguous_chunk_ids(self) -> None:
        """Chunk ids must be ``0..n-1`` so a retrieved row index IS a chunk id.

        Checked, not trusted. If ids were sparse or 1-based, every chunk looked
        up would be off by a constant and the resulting EM/F1 would read as a
        bad model rather than a broken join.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n, MIN(chunk_id) AS lo, MAX(chunk_id) AS hi FROM metadata"
            ).fetchone()
        n, lo, hi = int(row["n"]), int(row["lo"]), int(row["hi"])
        if (lo, hi) != (0, n - 1):
            raise ValueError(
                f"chunk ids are not contiguous 0..{n - 1} (got {lo}..{hi}); a retrieved "
                f"row index would not be a chunk id, and every context would be wrong"
            )
        self.n_chunks = n

    def chunk_texts(self, chunk_ids: Sequence[int]) -> list[str]:
        """Text for each id, **in the order given** -- rank order matters."""
        ids = [int(c) for c in chunk_ids if int(c) >= 0]
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT chunk_id, text FROM metadata WHERE chunk_id IN ({placeholders})",
                tuple(ids),
            ).fetchall()
        by_id = {int(r["chunk_id"]): str(r["text"]) for r in rows}
        # SQL returns no particular order; restore retrieval rank explicitly.
        return [by_id[i] for i in ids if i in by_id]

    def chunk_titles(self, chunk_ids: Sequence[int]) -> list[str]:
        """The article each chunk came from, in the order given.

        Needed to cite an answer: a passage number means nothing to a reader,
        and "Albert Einstein, SQuAD/Wikipedia" says where to go and check. A
        chunk with no document row gets an empty title rather than a guess.
        """
        ids = [int(c) for c in chunk_ids if int(c) >= 0]
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        with self._lock:
            rows = self._conn.execute(
                "SELECT m.chunk_id, COALESCE(d.title, '') AS title FROM metadata m "
                "LEFT JOIN documents d ON d.doc_id = m.doc_id "
                f"WHERE m.chunk_id IN ({placeholders})",
                tuple(ids),
            ).fetchall()
        by_id = {int(r["chunk_id"]): str(r["title"]) for r in rows}
        return [by_id.get(i, "") for i in ids]

    def questions(self, limit: int | None = None) -> list[QaQuestion]:
        sql = "SELECT query_id, squad_id, question, answers, gold_doc_id FROM queries ORDER BY query_id"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        out = []
        with self._lock:
            fetched = self._conn.execute(sql).fetchall()
        for r in fetched:
            answers = tuple(json.loads(r["answers"]))
            if not answers:
                raise ValueError(
                    f"query {r['query_id']} has no gold answers; it cannot be scored "
                    f"and must not be silently counted as a miss"
                )
            out.append(
                QaQuestion(
                    query_id=int(r["query_id"]),
                    squad_id=str(r["squad_id"]),
                    question=str(r["question"]),
                    answers=answers,
                    gold_doc_id=int(r["gold_doc_id"]),
                )
            )
        return out

    def close(self) -> None:
        self._conn.close()

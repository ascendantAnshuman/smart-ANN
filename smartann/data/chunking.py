"""SQuAD -> sentence-window chunks -> SQLite, mirroring MobileRAG's schema.

The three-table layout (``embeddings`` / ``documents`` / ``metadata``) is
MobileRAG's, so Phase 2's DB construction is a direct port rather than a
translation. A fourth table, ``queries``, holds the evaluation set; MobileRAG
does not specify one.

Two decisions here are load-bearing and documented in the ADR log:

- The corpus is the union of unique **train and dev** contexts. SQuAD v1.1's dev
  contexts are disjoint from its train contexts, so indexing train alone while
  querying dev questions would give identically zero gold-document recall and
  quietly poison every downstream retrieval metric.
- The chunk count will **not** match MobileRAG's reported 87,599 (ADR-0009).
  That figure is SQuAD's question count, and the paper does not state its
  chunking policy. Ours is stated, and whatever count it produces is recorded.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from smartann.data.registry import DatasetSpec
from smartann.pipeline.embedder import Embedder
from smartann.text import sentence_windows, split_sentences

__all__ = ["ChunkStats", "build_qa_sqlite", "load_squad_contexts", "open_chunk_db"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id     INTEGER PRIMARY KEY,
    file_path  TEXT NOT NULL,
    title      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS metadata (
    chunk_id    INTEGER PRIMARY KEY,
    doc_id      INTEGER NOT NULL,
    char_offset INTEGER NOT NULL,
    char_len    INTEGER NOT NULL,
    text        TEXT NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
);
CREATE TABLE IF NOT EXISTS embeddings (
    emb_id  INTEGER PRIMARY KEY,
    doc_id  INTEGER NOT NULL,
    vector  BLOB NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
);
CREATE TABLE IF NOT EXISTS queries (
    query_id     INTEGER PRIMARY KEY,
    squad_id     TEXT NOT NULL,
    question     TEXT NOT NULL,
    answers      TEXT NOT NULL,   -- JSON list of gold answer strings
    gold_doc_id  INTEGER NOT NULL,
    split        TEXT NOT NULL,
    FOREIGN KEY (gold_doc_id) REFERENCES documents (doc_id)
);
CREATE TABLE IF NOT EXISTS corpus_info (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metadata_doc ON metadata (doc_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_doc ON embeddings (doc_id);
"""


@dataclass(frozen=True)
class ChunkStats:
    """What preparation actually produced. Recorded, never predicted."""

    n_documents: int
    n_chunks: int
    n_queries: int
    n_train_contexts: int
    n_dev_contexts: int
    n_shared_contexts: int
    mean_sentences_per_doc: float
    mean_chunk_chars: float
    dim: int


@dataclass(frozen=True)
class SquadContext:
    title: str
    context: str
    split: str


def load_squad_contexts(path: Path, split: str) -> tuple[list[SquadContext], list[dict[str, object]]]:
    """Parse a SQuAD v1.1 JSON file into unique contexts and question records."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "data" not in payload:
        raise ValueError(f"{path}: not a SQuAD file (no 'data' key)")

    contexts: list[SquadContext] = []
    seen: set[str] = set()
    questions: list[dict[str, object]] = []

    for article in payload["data"]:
        title = article.get("title", "")
        for para in article.get("paragraphs", []):
            ctx = para["context"]
            if ctx not in seen:
                seen.add(ctx)
                contexts.append(SquadContext(title=title, context=ctx, split=split))
            for qa in para.get("qas", []):
                questions.append(
                    {
                        "squad_id": qa["id"],
                        "question": qa["question"],
                        "answers": sorted({a["text"] for a in qa.get("answers", [])}),
                        "context": ctx,
                        "split": split,
                    }
                )
    return contexts, questions


def open_chunk_db(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def build_qa_sqlite(
    spec: DatasetSpec,
    raw_paths: dict[str, Path],
    out_dir: Path,
    embedder: Embedder,
    quiet: bool = False,
) -> ChunkStats:
    """Build ``chunks.sqlite`` plus ``base.npy`` / ``queries.npy`` for a QA set.

    ``base.npy`` and ``queries.npy`` are written alongside the database so that
    QA datasets load through exactly the same reader path as the ANN datasets.
    Row *i* of ``base.npy`` corresponds to ``chunk_id = i``, which is the
    contract Phase 2's index relies on to map a retrieved vector back to text.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "chunks.sqlite"
    if db_path.exists():
        db_path.unlink()

    window = int(spec.extra.get("chunk_window_sentences", 3))
    overlap = int(spec.extra.get("chunk_overlap_sentences", 1))
    seed = int(spec.extra.get("query_sample_seed", 0))
    query_split = str(spec.extra.get("query_split", "dev"))

    train_ctxs, train_qs = load_squad_contexts(raw_paths["train"], "train")
    dev_ctxs, dev_qs = load_squad_contexts(raw_paths["dev"], "dev")

    train_texts = {c.context for c in train_ctxs}
    dev_texts = {c.context for c in dev_ctxs}
    n_shared = len(train_texts & dev_texts)

    # Union of both splits, deterministically ordered so doc_ids are stable
    # across runs and machines (sorted by text, not by dict iteration order).
    merged: dict[str, SquadContext] = {}
    for c in train_ctxs + dev_ctxs:
        merged.setdefault(c.context, c)
    documents = sorted(merged.values(), key=lambda c: (c.context, c.title))

    conn = open_chunk_db(db_path)
    conn.executescript(SCHEMA)

    doc_id_of: dict[str, int] = {}
    doc_rows = []
    for doc_id, doc in enumerate(documents):
        doc_id_of[doc.context] = doc_id
        doc_rows.append((doc_id, f"squad/{doc.split}/{doc.title}", doc.title))
    conn.executemany("INSERT INTO documents VALUES (?, ?, ?)", doc_rows)

    # --- chunks -----------------------------------------------------------
    chunk_rows: list[tuple[int, int, int, int, str]] = []
    chunk_texts: list[str] = []
    sentence_counts: list[int] = []
    chunk_id = 0
    for doc in tqdm(documents, desc="chunking", disable=quiet or not sys.stderr.isatty(), leave=False):
        sentences = split_sentences(doc.context)
        sentence_counts.append(len(sentences))
        doc_id = doc_id_of[doc.context]
        for i, j in sentence_windows(sentences, window, overlap):
            start = sentences[i].start
            end = sentences[j - 1].end
            text = doc.context[start:end]
            chunk_rows.append((chunk_id, doc_id, start, end - start, text))
            chunk_texts.append(text)
            chunk_id += 1
    conn.executemany("INSERT INTO metadata VALUES (?, ?, ?, ?, ?)", chunk_rows)

    # --- embeddings -------------------------------------------------------
    vectors = embedder.encode(chunk_texts, batch_size=64, show_progress=not (quiet or not sys.stderr.isatty()))
    if vectors.shape != (len(chunk_texts), spec.dim):
        raise ValueError(f"embedder returned {vectors.shape}, expected {(len(chunk_texts), spec.dim)}")
    conn.executemany(
        "INSERT INTO embeddings VALUES (?, ?, ?)",
        [
            (cid, did, np.ascontiguousarray(vectors[cid], dtype=np.float32).tobytes())
            for cid, did, _, _, _ in chunk_rows
        ],
    )

    # --- queries ----------------------------------------------------------
    pool = [q for q in (dev_qs if query_split == "dev" else train_qs) if q["answers"]]
    pool.sort(key=lambda q: str(q["squad_id"]))  # deterministic before sampling
    rng = np.random.default_rng(seed)
    n_take = min(spec.expected_n_queries, len(pool))
    picked = [pool[i] for i in rng.choice(len(pool), size=n_take, replace=False)]

    query_rows = []
    for qid, q in enumerate(picked):
        gold = doc_id_of.get(str(q["context"]))
        if gold is None:
            raise ValueError(
                f"query {q['squad_id']} has a context absent from the corpus; "
                "this should be impossible now that train and dev are unioned"
            )
        query_rows.append(
            (qid, str(q["squad_id"]), str(q["question"]), json.dumps(q["answers"]), gold, query_split)
        )
    conn.executemany("INSERT INTO queries VALUES (?, ?, ?, ?, ?, ?)", query_rows)

    query_vectors = embedder.encode([str(q["question"]) for q in picked], batch_size=64)

    stats = ChunkStats(
        n_documents=len(documents),
        n_chunks=len(chunk_rows),
        n_queries=len(query_rows),
        n_train_contexts=len(train_texts),
        n_dev_contexts=len(dev_texts),
        n_shared_contexts=n_shared,
        mean_sentences_per_doc=float(np.mean(sentence_counts)) if sentence_counts else 0.0,
        mean_chunk_chars=float(np.mean([len(t) for t in chunk_texts])) if chunk_texts else 0.0,
        dim=spec.dim,
    )
    conn.executemany(
        "INSERT OR REPLACE INTO corpus_info VALUES (?, ?)",
        [
            ("chunk_window_sentences", str(window)),
            ("chunk_overlap_sentences", str(overlap)),
            ("query_sample_seed", str(seed)),
            ("query_split", query_split),
            ("embedding_model", str(spec.embedding_model)),
            ("space", spec.space),
            ("search_space", spec.search_space),
            ("normalized", str(spec.normalized)),
            ("n_documents", str(stats.n_documents)),
            ("n_chunks", str(stats.n_chunks)),
            ("n_queries", str(stats.n_queries)),
            ("resolved_nc", str(spec.resolve_nc(stats.n_chunks))),
        ],
    )
    conn.commit()
    conn.close()

    np.save(out_dir / "base.npy", vectors)
    np.save(out_dir / "queries.npy", query_vectors)
    return stats

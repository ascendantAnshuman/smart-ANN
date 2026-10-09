#!/usr/bin/env python
"""Show the input in text form, at every stage it exists in.

Written for the question "where is your input, and what does it look like?".
Answering it by opening a 30 MB JSON in an editor shows the format but not the
transformation; answering it with `base.npy` shows the transformation but not
the text. This walks ONE passage from the raw SQuAD file to the row of
``base.npy`` that the index actually searches, so the two ends are visibly the
same piece of writing.

Read-only: it opens the corpus, prints, and exits. The SQLite handle is opened
``mode=ro`` so running this during a demo cannot damage a prepared dataset.

    python scripts/show_input.py                  # the default passage
    python scripts/show_input.py --doc-id 1844    # a chosen document
    python scripts/show_input.py --query 7        # a chosen question
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import textwrap
from pathlib import Path

import numpy as np

from smartann.config import resolve_paths

WIDTH = 88
RULE = "=" * WIDTH

# SQuAD v1.1 train's first article, so it is also what an examiner sees if they
# open the raw file themselves rather than trusting this script.
DEFAULT_TITLE = "University_of_Notre_Dame"


def rule(stage: str, where: Path | str) -> None:
    print(f"\n{RULE}\n{stage}\n{where}\n{RULE}")


def wrap(text: str, indent: str = "  ") -> str:
    return textwrap.fill(text, WIDTH, initial_indent=indent, subsequent_indent=indent)


def stage1_raw(raw_dir: Path, title: str) -> str | None:
    """The input as published: SQuAD v1.1 JSON, byte-for-byte unmodified."""
    train = raw_dir / "train-v1.1.json"
    rule("STAGE 1 -- RAW INPUT (text, exactly as downloaded)", train)
    if not train.exists():
        print(f"  MISSING: {train}")
        print("  Run: smartann data prepare squad")
        return None

    print(f"  {train.stat().st_size / 1e6:.1f} MB of JSON. Its shape:\n")
    print("    {'data': [{'title': ..., 'paragraphs': [")
    print("        {'context': <the passage>,")
    print("         'qas': [{'question': ..., 'answers': [{'text': ...}]}]}")
    print("    ]}]}\n")

    payload = json.loads(train.read_text(encoding="utf-8"))
    article = next((a for a in payload["data"] if a.get("title") == title), payload["data"][0])
    para = article["paragraphs"][0]
    qa = para["qas"][0]

    print(f"  title    : {article['title']}")
    print(f"  context  : {len(para['context'])} characters --")
    print(wrap(para["context"], "    "))
    print(f"\n  question : {qa['question']}")
    print(
        f"  answer   : {qa['answers'][0]['text']!r} at character {qa['answers'][0]['answer_start']}"
    )
    n_paras = sum(len(a["paragraphs"]) for a in payload["data"])
    print(f"\n  This article: {len(article['paragraphs'])} paragraphs.")
    print(f"  The whole file: {len(payload['data'])} articles, {n_paras:,} paragraphs.")
    return str(para["context"])


def stage2_chunks(db: sqlite3.Connection, context: str | None, doc_id: int | None) -> int:
    """The input after chunking: the same prose, cut into overlapping windows."""
    if doc_id is None:
        if context is None:
            raise SystemExit("no passage to trace; pass --doc-id")
        row = db.execute(
            "SELECT doc_id FROM metadata WHERE text LIKE ? LIMIT 1", (context[:60] + "%",)
        ).fetchone()
        if row is None:
            raise SystemExit("that passage is not in the corpus")
        doc_id = int(row["doc_id"])

    doc = db.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    chunks = db.execute(
        "SELECT chunk_id, char_offset, char_len, text FROM metadata "
        "WHERE doc_id = ? ORDER BY chunk_id",
        (doc_id,),
    ).fetchall()

    print(f"\n  doc_id {doc_id}   title {doc['title']}   file_path {doc['file_path']}")
    print(f"  That one passage became {len(chunks)} chunks")
    print("  (3-sentence windows, 1 sentence of overlap):\n")
    for c in chunks:
        print(
            f"  chunk_id {c['chunk_id']:<7} char_offset {c['char_offset']:<5} char_len {c['char_len']}"
        )
        print(wrap(c["text"], "      "))
        print()
    if len(chunks) > 1:
        print("  Note the overlap: each chunk's last sentence is the next one's first, so")
        print("  char_offset advances by less than char_len. A sentence that answers a")
        print("  question is therefore never split across a boundary and lost.")
    return int(chunks[0]["chunk_id"])


def stage3_vector(db: sqlite3.Connection, processed: Path, chunk_id: int) -> None:
    """The input after embedding: 384 float32s, which is what the index sees."""
    base_path = processed / "base.npy"
    rule("STAGE 3 -- THE SAME CHUNK AS A VECTOR (what the index searches)", base_path)

    base = np.load(base_path, mmap_mode="r")
    vec = np.asarray(base[chunk_id])
    print(f"  base.npy is {base.shape[0]:,} x {base.shape[1]} {base.dtype} -- one row per chunk.")
    print(f"  Row {chunk_id} IS chunk_id {chunk_id}: ids are 0..n-1 and contiguous, asserted on")
    print("  open in smartann/generate/corpus.py, so a retrieved row index is a chunk id")
    print("  with no lookup table in between.\n")
    head = ", ".join(f"{x:+.4f}" for x in vec[:8])
    print(f"  first 8 of 384 dims : [{head}, ...]")
    print(f"  L2 norm             : {float(np.linalg.norm(vec)):.6f}")
    print("                        unit length, so cosine and inner product rank the")
    print("                        same and the index searches with 'ip'.")

    blob = db.execute("SELECT vector FROM embeddings WHERE emb_id = ?", (chunk_id,)).fetchone()
    if blob is not None:
        stored = np.frombuffer(blob["vector"], dtype=np.float32)
        same = bool(np.allclose(stored, vec, atol=1e-6))
        print(
            f"\n  chunks.sqlite holds the same {len(blob['vector'])}-byte blob; identical = {same}"
        )


def stage4_queries(db: sqlite3.Connection, processed: Path, query_id: int) -> None:
    """The other input: the questions, and the answers they are scored against."""
    rule("THE QUERY SIDE -- questions are text too", f"{processed / 'chunks.sqlite'} :: queries")
    q = db.execute("SELECT * FROM queries WHERE query_id = ?", (query_id,)).fetchone()
    if q is None:
        print(f"  no query_id {query_id}")
        return
    gold = db.execute(
        "SELECT title FROM documents WHERE doc_id = ?", (q["gold_doc_id"],)
    ).fetchone()
    print(f"  query_id     : {q['query_id']}    squad_id {q['squad_id']}    split {q['split']}")
    print(f"  question     : {q['question']}")
    print(f"  gold answers : {json.loads(q['answers'])}")
    print(f"  gold_doc_id  : {q['gold_doc_id']}  ({gold['title'] if gold else '?'})")

    queries = np.load(processed / "queries.npy", mmap_mode="r")
    print(f"\n  queries.npy is {queries.shape[0]:,} x {queries.shape[1]} {queries.dtype};")
    print(f"  row {query_id} is that question embedded by the SAME model as the chunks.")
    print("  Retrieval is one inner product between this row and the corpus.")
    print("\n  These questions carry gold answer strings, which is the only reason EM and")
    print("  F1 can be computed at all -- SIFT and NYTimes are bare vectors with no text.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Show the SQuAD input at every stage.")
    ap.add_argument("--doc-id", type=int, default=None, help="Trace this document instead.")
    ap.add_argument("--title", default=DEFAULT_TITLE, help="Trace this article title.")
    ap.add_argument("--query", type=int, default=0, help="Query id to display.")
    args = ap.parse_args()

    paths = resolve_paths()
    processed = paths.processed / "squad"
    db_path = processed / "chunks.sqlite"
    print(f"\ndata_root = {paths.data_root}    (override with SMARTANN_DATA_ROOT)")

    context = stage1_raw(paths.raw / "squad", args.title)

    rule("STAGE 2 -- THE PROJECT'S OWN FORMAT (still text)", db_path)
    if not db_path.exists():
        print(f"  MISSING: {db_path}")
        print("  Run: smartann data prepare squad")
        return 1

    db = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    print("  Four tables: documents = articles, metadata = chunks AND THEIR TEXT,")
    print("  embeddings = vectors, queries = questions + gold answers.\n")
    info = {r["key"]: r["value"] for r in db.execute("SELECT key, value FROM corpus_info")}
    for key in (
        "n_documents",
        "n_chunks",
        "n_queries",
        "embedding_model",
        "chunk_window_sentences",
        "chunk_overlap_sentences",
        "space",
        "search_space",
        "normalized",
    ):
        if key in info:
            print(f"    {key:<24} {info[key]}")

    chunk_id = stage2_chunks(db, context, args.doc_id)
    stage3_vector(db, processed, chunk_id)
    stage4_queries(db, processed, args.query)

    print(f"\n{RULE}")
    print("Same words, three representations: published JSON -> chunk rows that keep the")
    print("text -> unit vectors. Nothing is discarded; the text stays queryable in")
    print("chunks.sqlite, which is what makes a generated answer gradeable against a gold.")
    print(f"{RULE}\n")
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

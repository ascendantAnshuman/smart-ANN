#!/usr/bin/env python
"""11.2 -- prepare and index the demonstration corpus.

The authored documents in `data/demo_phone_care/documents.json` go through the
**same** chunker, the same embedder and the same index builder as SQuAD, and
land in the same on-disk layout. That is the point: an answer the demonstration
gives and a number the benchmark reports then come from one implementation. A
second, simpler path for the demo would be a second system, and only one of them
would have produced the recorded results.

Reused verbatim, not reimplemented:

* `smartann.text.split_sentences` / `sentence_windows` -- 3-sentence windows
  with 1 sentence of overlap, the parameters recorded in the SQuAD corpus_info
* `smartann.data.chunking.SCHEMA` and `open_chunk_db` -- the same tables, so
  `QaCorpus` reads this corpus with no special case
* `smartann.pipeline.embedder.Embedder` -- gte-small, L2-normalised at write
* `smartann.index.builder.build_ecovector` -- the same partition and build

**Why this dataset is not in the registry.** `smartann/data/registry.py`
describes datasets that are *downloaded*: each carries source URLs and a
recorded SHA-256, and `data prepare` / `data verify` exist to fetch and check
them. This corpus is authored in this repository and committed to it, so there
is nothing to download and nothing to checksum against an upstream. Adding a
registry entry with no artifacts would make `data status` report a dataset that
can never be verified. The index config carries everything the build needs
instead, and `resolve_index_root("demo_phone_care")` finds the result because it
resolves through that same config file.

**The space is asserted, not declared.** gte-small is a cosine model and the
embedder returns unit vectors, so the search space is inner product -- the same
derivation the registry applies to SQuAD. The vectors are checked for unit norm
before the build rather than trusted, because a metric mismatch is the bug that
silently halves recall (ADR-0010).
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smartann.config import config_hash, load_yaml, resolve_paths
from smartann.data.chunking import SCHEMA, open_chunk_db
from smartann.index.builder import build_ecovector
from smartann.index.storage import IndexBuildConfig, IndexPaths
from smartann.pipeline.embedder import DEFAULT_MODEL, Embedder
from smartann.text import sentence_windows, split_sentences

DATASET = "demo_phone_care"
WINDOW = 3  # sentences per chunk, as recorded in the SQuAD corpus
OVERLAP = 1
DIM = 384
SPACE = "ip"  # gte-small is cosine; unit vectors make that inner product
REQUIRED_FIELDS = {"id", "title", "text", "source"}


def load_documents(path: Path) -> list[dict[str, str]]:
    """Read the authored corpus, refusing anything malformed.

    Every field is required on every document. A document with no id cannot be
    cited by an answer, and a document with no source cannot be labelled in the
    interface, so neither is allowed through to be noticed later.
    """
    docs = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(docs, list) or not docs:
        raise ValueError(f"{path} does not hold a non-empty list of documents")
    seen: set[str] = set()
    for i, d in enumerate(docs):
        missing = REQUIRED_FIELDS - set(d)
        if missing:
            raise ValueError(f"document {i} is missing {sorted(missing)}")
        if not str(d["text"]).strip():
            raise ValueError(f"document {d['id']} has no text")
        if d["id"] in seen:
            raise ValueError(f"duplicate document id {d['id']!r}")
        seen.add(d["id"])
    return docs


def chunk_documents(docs: list[dict[str, str]]) -> list[tuple[int, int, int, int, str]]:
    """Sentence-window every document, returning metadata rows.

    Offsets are into the document's own text, which is what the SQuAD rows
    carry, so a chunk can always be located back in the document it came from.
    """
    rows: list[tuple[int, int, int, int, str]] = []
    chunk_id = 0
    for doc_id, doc in enumerate(docs):
        text = str(doc["text"])
        sentences = split_sentences(text)
        for i, j in sentence_windows(sentences, WINDOW, OVERLAP):
            start, end = sentences[i].start, sentences[j - 1].end
            rows.append((chunk_id, doc_id, start, end - start, text[start:end]))
            chunk_id += 1
    if not rows:
        raise ValueError("the corpus produced no chunks")
    return rows


def write_corpus_db(
    db_path: Path, docs: list[dict[str, str]], rows: list[tuple[int, int, int, int, str]]
) -> None:
    if db_path.exists():
        db_path.unlink()
    conn: sqlite3.Connection = open_chunk_db(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT INTO documents VALUES (?, ?, ?)",
            [(i, f"{DATASET}/{d['id']}", str(d["title"])) for i, d in enumerate(docs)],
        )
        conn.executemany("INSERT INTO metadata VALUES (?, ?, ?, ?, ?)", rows)
        conn.executemany(
            "INSERT INTO corpus_info VALUES (?, ?)",
            [
                ("dataset", DATASET),
                ("n_documents", str(len(docs))),
                ("n_chunks", str(len(rows))),
                ("chunk_window_sentences", str(WINDOW)),
                ("chunk_overlap_sentences", str(OVERLAP)),
                ("embedding_model", DEFAULT_MODEL),
                ("source", "authored for demo; see data/demo_phone_care/documents.json"),
                (
                    "provenance",
                    "ORIGINAL text written for this demonstration. Not scraped, not "
                    "copied, and not a measurement of anything.",
                ),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docs", type=Path, default=Path("data/demo_phone_care/documents.json"))
    ap.add_argument(
        "--config", type=Path, default=Path("configs/index/demo_phone_care_ecovector.yaml")
    )
    ap.add_argument("--force", action="store_true", help="Rebuild even if the index exists.")
    args = ap.parse_args()

    if not args.docs.exists():
        print(f"missing {args.docs}")
        return 2

    docs = load_documents(args.docs)
    rows = chunk_documents(docs)
    texts = [r[4] for r in rows]
    print(
        f"{len(docs)} documents -> {len(rows)} chunks ({WINDOW}-sentence windows, {OVERLAP} overlap)"
    )

    paths = resolve_paths()
    out_dir = Path(paths.data_root) / "processed" / DATASET
    out_dir.mkdir(parents=True, exist_ok=True)

    embedder = Embedder(cache_path=Path(paths.data_root) / "cache" / "demo.sqlite")
    vectors = np.asarray(embedder.encode(texts, batch_size=64), dtype=np.float32)
    if vectors.shape != (len(texts), DIM):
        raise ValueError(f"embedder returned {vectors.shape}, expected ({len(texts)}, {DIM})")
    # Same tolerance Embedder itself enforces on every batch it returns: float32
    # accumulation leaves norms a few parts in 1e-4 from exactly 1.0, and a
    # tighter bound here would reject vectors the shipping contract accepts.
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(
            f"vectors are not unit norm (min {norms.min():.6f}, max {norms.max():.6f}); "
            f"space={SPACE!r} would then be the wrong metric for this corpus"
        )

    write_corpus_db(out_dir / "chunks.sqlite", docs, rows)
    np.save(out_dir / "base.npy", vectors)
    print(f"wrote {out_dir / 'chunks.sqlite'} and {out_dir / 'base.npy'}")

    cfg = load_yaml(args.config, IndexBuildConfig)
    if cfg.dataset != DATASET:
        raise ValueError(f"{args.config} builds '{cfg.dataset}', not '{DATASET}'")
    nc = int(cfg.nc) if cfg.nc != "auto" else max(1, round(math.sqrt(len(rows))))
    chash = config_hash(cfg)
    out = IndexPaths.for_build(Path(paths.data_root) / "index", DATASET, chash)
    if out.meta.exists() and not args.force:
        print(f"{out.root} exists; pass --force to rebuild.")
        return 0

    print(f"building ecovector: {len(rows)}x{DIM} space={SPACE} nc={nc} hash={chash}")
    meta = build_ecovector(
        vectors,
        dataset=DATASET,
        space=SPACE,
        nc=nc,
        m=cfg.m,
        ef_construction=cfg.ef_construction,
        min_graph_size=cfg.min_graph_size,
        seed=cfg.seed,
        config_hash=chash,
        out=out,
        index_variant=cfg.index_variant,
        kmeans_batch_size=cfg.kmeans_batch_size,
        kmeans_n_init=cfg.kmeans_n_init,
    ).meta
    print(
        f"  clusters: {meta.n_graph_clusters} graph, {meta.n_flat_clusters} flat, "
        f"{meta.n_empty_clusters} empty; largest {meta.largest_cluster}, "
        f"smallest non-empty {meta.smallest_nonempty_cluster}"
    )
    print(f"  {out.root}")

    # Resolve it the way every caller will, so a build that cannot be found
    # fails here rather than inside the demonstration.
    from smartann.index.storage import resolve_index_root

    found = resolve_index_root(DATASET, "ecovector")
    if found is None or found.root != out.root:
        raise RuntimeError(
            f"the index was built at {out.root} but resolve_index_root returned "
            f"{found.root if found else None}. The demo would not find it."
        )
    print(f"resolve_index_root agrees: {found.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

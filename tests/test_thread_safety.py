"""A loaded pipeline must survive being used from a different thread.

Found the hard way. The dashboard caches the loaded pipeline so that a 21-second
model load is paid once, and Streamlit runs each interaction on a worker thread.
SQLite's default `check_same_thread` guard then raises on the **second**
question and not the first, because the first question happens to run on the
thread that opened the connection.

That is the worst shape a bug can have: invisible in every quick test, invisible
in the first demonstration question, and fatal in front of an audience on the
second. These tests open each long-lived store on one thread and read it from
another, which is exactly what a server does and what no single-threaded test
would have caught.

`sqlite3.threadsafety` is 3 (serialized) on this build, so the module serialises
access itself and disabling the guard is safe. The first test asserts that,
because if a future interpreter ships a less thread-safe sqlite3 the fix in
these stores stops being sound and someone needs to know.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from smartann.generate.corpus import QaCorpus
from smartann.pipeline.embedder import EmbeddingCache


def _on_another_thread(fn: Any) -> Any:
    """Run `fn` on a fresh thread and return its result, re-raising any error."""
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:
            box["error"] = exc

    t = threading.Thread(target=run)
    t.start()
    t.join(timeout=30)
    if "error" in box:
        raise box["error"]
    return box.get("value")


def test_the_sqlite_module_is_serialized() -> None:
    """The premise the fix rests on. If this ever fails, the stores need a lock
    rather than a disabled guard."""
    assert sqlite3.threadsafety == 3


def test_the_embedding_cache_survives_a_thread_change(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "cache.sqlite", dim=4)
    cache.put_many([("k", np.ones(4, dtype=np.float32))])

    found = _on_another_thread(lambda: cache.get_many(["k"]))
    assert "k" in found
    assert found["k"].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_the_embedding_cache_survives_a_write_from_another_thread(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "cache.sqlite", dim=4)
    _on_another_thread(lambda: cache.put_many([("later", np.zeros(4, dtype=np.float32))]))
    assert "later" in cache.get_many(["later"])


def _tiny_corpus(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE documents (doc_id INTEGER PRIMARY KEY, file_path TEXT, title TEXT);"
        "CREATE TABLE metadata (chunk_id INTEGER PRIMARY KEY, doc_id INTEGER, text TEXT);"
    )
    conn.execute("INSERT INTO documents VALUES (0, 'demo/a', 'A title')")
    conn.execute("INSERT INTO metadata VALUES (0, 0, 'a passage')")
    conn.commit()
    conn.close()


def test_the_corpus_survives_a_thread_change(tmp_path: Path) -> None:
    db = tmp_path / "chunks.sqlite"
    _tiny_corpus(db)
    corpus = QaCorpus(db)
    try:
        assert _on_another_thread(lambda: corpus.chunk_texts([0])) == ["a passage"]
        assert _on_another_thread(lambda: corpus.chunk_titles([0])) == ["A title"]
    finally:
        corpus.close()


def test_the_knowledge_base_survives_a_thread_change(tmp_path: Path) -> None:
    from smartann.kb import KnowledgeBase

    kb = KnowledgeBase(tmp_path / "kb", dim=4)
    try:
        assert _on_another_thread(lambda: len(kb)) == 0
        assert _on_another_thread(lambda: kb.sources()) == []
    finally:
        kb.close()


def test_concurrent_readers_do_not_corrupt_the_cache(tmp_path: Path) -> None:
    """The one that found the real bug, and only found it under full-suite load.

    A barrier makes every reader hit the connection in the same instant instead
    of hoping the scheduler interleaves them, so this fails reliably when the
    lock is removed rather than once in a few runs. Without the lock it raises
    "not enough values to unpack (expected 2, got 0)" -- one thread fetching
    rows from another thread's statement.
    """
    cache = EmbeddingCache(tmp_path / "cache.sqlite", dim=4)
    cache.put_many([(f"k{i}", np.full(4, i, dtype=np.float32)) for i in range(50)])

    n_threads, rounds = 8, 40
    start = threading.Barrier(n_threads)
    results: list[int] = []
    errors: list[BaseException] = []

    def reader() -> None:
        try:
            start.wait(timeout=30)
            for _ in range(rounds):
                results.append(len(cache.get_many([f"k{i}" for i in range(50)])))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not errors, errors
    assert len(results) == n_threads * rounds
    assert set(results) == {50}


def test_concurrent_readers_do_not_corrupt_the_corpus(tmp_path: Path) -> None:
    """Same shape, against the chunk store the answer path reads on every query."""
    db = tmp_path / "chunks.sqlite"
    _tiny_corpus(db)
    corpus = QaCorpus(db)
    n_threads, rounds = 8, 40
    start = threading.Barrier(n_threads)
    errors: list[BaseException] = []
    texts: list[str] = []

    def reader() -> None:
        try:
            start.wait(timeout=30)
            for _ in range(rounds):
                texts.extend(corpus.chunk_texts([0]))
                corpus.chunk_titles([0])
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    corpus.close()

    assert not errors, errors
    assert set(texts) == {"a passage"}


@pytest.mark.parametrize(
    "module",
    ["smartann/generate/corpus.py", "smartann/pipeline/embedder.py", "smartann/kb/store.py"],
)
def test_every_long_lived_store_is_both_shareable_and_locked(module: str) -> None:
    """A new long-lived SQLite store added without both of these reintroduces the
    bug, and it only shows up on someone's second question or under concurrency."""
    source = (Path(__file__).resolve().parent.parent / module).read_text(encoding="utf-8")
    assert "check_same_thread=False" in source, f"{module} would raise when reused across threads"
    assert "self._lock" in source, f"{module} drives a shared connection with no lock"

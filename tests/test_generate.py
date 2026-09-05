"""Phase 7: prompt construction and the QA corpus join.

The model itself is not loaded here. Loading FLAN-T5 takes seconds and needs a
download, and none of the logic worth testing is inside it -- the failure modes
are in the join between retrieval and generation, which is pure Python.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from smartann.generate.corpus import QaCorpus
from smartann.generate.prompt import TEMPLATE, build_prompt, join_chunks


class TestPrompt:
    def test_chunks_are_numbered_in_rank_order(self) -> None:
        assert join_chunks(["alpha", "beta"]) == "[1] alpha\n\n[2] beta"

    def test_blank_chunks_are_dropped_without_shifting_numbering(self) -> None:
        """A dropped chunk must not leave a gap in the numbering."""
        assert join_chunks(["alpha", "   ", "beta"]) == "[1] alpha\n\n[2] beta"

    def test_no_chunks_yields_empty_context(self) -> None:
        assert join_chunks([]) == ""

    def test_prompt_contains_question_and_context(self) -> None:
        p = build_prompt("Who?", ["ctx one"])
        assert "Who?" in p
        assert "[1] ctx one" in p

    def test_template_offers_an_unanswerable_escape(self) -> None:
        """C2 must be able to see reduction destroy an answer.

        Without an explicit refusal option the model invents a span from
        whatever context survives, and a destroyed answer scores as a wrong
        answer rather than as a detectable abstention.
        """
        assert "unanswerable" in TEMPLATE

    def test_one_template_for_every_arm(self) -> None:
        """Reduced and unreduced prompts differ only in the context.

        If the instruction differed between arms, the measured quality gap
        would confound reduction with prompt wording.
        """
        full = build_prompt("Q?", ["a", "b"])
        reduced = build_prompt("Q?", ["a"])
        assert full.split("Context:")[0] == reduced.split("Context:")[0]
        assert full.split("\n\nQuestion:")[1] == reduced.split("\n\nQuestion:")[1]


def _make_db(path: Path, chunk_ids: list[int], answers: str = '["Paris"]') -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE metadata (chunk_id INTEGER PRIMARY KEY, doc_id INTEGER,
                               char_offset INTEGER, char_len INTEGER, text TEXT);
        CREATE TABLE queries (query_id INTEGER PRIMARY KEY, squad_id TEXT,
                              question TEXT, answers TEXT, gold_doc_id INTEGER,
                              split TEXT);
        """
    )
    conn.executemany(
        "INSERT INTO metadata VALUES (?,?,?,?,?)",
        [(c, 0, 0, 4, f"text-{c}") for c in chunk_ids],
    )
    conn.execute(
        "INSERT INTO queries VALUES (?,?,?,?,?,?)",
        (0, "id0", "Where?", answers, 0, "dev"),
    )
    conn.commit()
    conn.close()


class TestQaCorpus:
    def test_chunk_texts_preserve_retrieval_rank(self, tmp_path: Path) -> None:
        """SQL returns rows in no guaranteed order; rank must survive."""
        db = tmp_path / "c.sqlite"
        _make_db(db, [0, 1, 2])
        corpus = QaCorpus(db)
        assert corpus.chunk_texts([2, 0, 1]) == ["text-2", "text-0", "text-1"]
        corpus.close()

    def test_negative_ids_are_dropped(self, tmp_path: Path) -> None:
        """Retrieval pads short result sets with -1."""
        db = tmp_path / "c.sqlite"
        _make_db(db, [0, 1, 2])
        corpus = QaCorpus(db)
        assert corpus.chunk_texts([0, -1, 1]) == ["text-0", "text-1"]
        corpus.close()

    def test_non_contiguous_chunk_ids_raise(self, tmp_path: Path) -> None:
        """A retrieved row index must BE a chunk id.

        If ids were 1-based, every context would be off by one and the
        resulting EM/F1 would read as a weak model rather than a broken join.
        """
        db = tmp_path / "c.sqlite"
        _make_db(db, [1, 2, 3])
        with pytest.raises(ValueError, match="not contiguous"):
            QaCorpus(db)

    def test_questions_carry_gold_answers(self, tmp_path: Path) -> None:
        db = tmp_path / "c.sqlite"
        _make_db(db, [0], answers=json.dumps(["Paris", "paris, France"]))
        corpus = QaCorpus(db)
        q = corpus.questions()[0]
        assert q.answers == ("Paris", "paris, France")
        corpus.close()

    def test_question_with_empty_gold_list_raises(self, tmp_path: Path) -> None:
        """A scoreless question must not be silently counted as a miss."""
        db = tmp_path / "c.sqlite"
        _make_db(db, [0], answers="[]")
        corpus = QaCorpus(db)
        with pytest.raises(ValueError, match="no gold answers"):
            corpus.questions()
        corpus.close()

    def test_missing_database_names_the_fix(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="data prepare squad"):
            QaCorpus(tmp_path / "absent.sqlite")

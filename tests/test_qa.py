"""SQuAD exact match and token F1.

Every case here is a place a reimplementation is known to drift from the
official script. None of them is exercised by a happy-path example, which is
why they are written out individually.
"""

from __future__ import annotations

import pytest

from smartann.bench.qa import exact_match, f1, normalize_answer, score_answer


class TestNormalisation:
    def test_articles_are_stripped(self) -> None:
        assert normalize_answer("the Eiffel Tower") == "eiffel tower"

    def test_punctuation_is_removed(self) -> None:
        assert normalize_answer("Paris, France.") == "paris france"

    def test_case_is_folded_and_whitespace_collapsed(self) -> None:
        assert normalize_answer("  PARIS   France ") == "paris france"

    def test_article_inside_a_word_survives(self) -> None:
        """``\b`` matters: 'theatre' must not become 'atre'."""
        assert normalize_answer("theatre") == "theatre"

    def test_punctuation_removed_before_whitespace_fix(self) -> None:
        """Order matters: hyphen removal must not weld two tokens together
        after the whitespace pass has already run."""
        assert normalize_answer("well-known") == "wellknown"


class TestExactMatch:
    def test_matches_after_normalisation(self) -> None:
        assert exact_match("The Answer.", "answer") == 1.0

    def test_rejects_a_superset(self) -> None:
        assert exact_match("the answer is Paris", "Paris") == 0.0


class TestF1:
    def test_identical_scores_one(self) -> None:
        assert f1("Paris France", "paris france") == 1.0

    def test_disjoint_scores_zero(self) -> None:
        assert f1("Berlin", "Paris") == 0.0

    def test_partial_overlap(self) -> None:
        # pred 2 tokens, gold 3, 2 common -> p=1.0, r=2/3, F1=0.8
        assert f1("New York", "New York City") == pytest.approx(0.8)

    def test_repeated_tokens_use_multiset_intersection(self) -> None:
        """The classic bug: sets would score this 1.0.

        Prediction has 'new' twice, gold once. Multiset intersection counts one
        'new', so precision is 2/3 rather than 1.0. The first draft of this
        test used "a a b" and scored 1.0 -- because 'a' is an article and
        normalisation had already deleted it, leaving nothing to repeat.
        """
        got = f1("New New York", "New York")
        assert got == pytest.approx(2 * (2 / 3) * 1.0 / ((2 / 3) + 1.0))
        assert got < 1.0

    def test_repeated_article_is_deleted_not_counted(self) -> None:
        """Normalisation runs first, so repeated articles cannot affect F1."""
        assert f1("a a b", "a b") == 1.0

    def test_both_empty_after_normalisation_scores_one(self) -> None:
        assert f1(".", ",") == 1.0

    def test_one_empty_scores_zero(self) -> None:
        assert f1("", "paris") == 0.0


class TestScoreAnswer:
    def test_maximises_over_gold_answers(self) -> None:
        em, token_f1 = score_answer("Paris", ["London", "Paris", "Lyon"])
        assert (em, token_f1) == (1.0, 1.0)

    def test_partial_credit_from_the_best_gold(self) -> None:
        em, token_f1 = score_answer("New York", ["Boston", "New York City"])
        assert em == 0.0
        assert token_f1 == pytest.approx(0.8)

    def test_empty_gold_list_raises(self) -> None:
        """A question with no reference is a data gap, not a model failure.

        Returning 0.0 would attribute the gap to the model and silently drag
        the reported mean down (Section 3, rule 2).
        """
        with pytest.raises(ValueError, match="no gold answers"):
            score_answer("Paris", [])

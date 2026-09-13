"""
Hand-computed tests for the metrics.

The expected values here are worked out from the DEFINITIONS, written as
explicit arithmetic, not by calling the functions under test. That is the
whole point - if the implementation and the test both had the same bug,
the test would be worthless.

    pip install pytest
    python -m pytest tests/ -v
"""
import math

import pytest

from contextfuse.metrics import (
    dcg_at_k, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank,
)

# The running example, used throughout:
#   relevant: A(id=1) with grade 2, B(id=2) with grade 1
#   ranking : C(3), A(1), D(4), B(2), E(5)
REL = {1: 2.0, 2: 1.0}
RANKING = [3, 1, 4, 2, 5]


def test_recall():
    assert recall_at_k(RANKING, REL, 1) == 0.0        # C is not relevant
    assert recall_at_k(RANKING, REL, 2) == 0.5        # A found, B not yet
    assert recall_at_k(RANKING, REL, 5) == 1.0        # both found


def test_precision():
    assert precision_at_k(RANKING, REL, 1) == 0.0
    assert precision_at_k(RANKING, REL, 2) == 0.5     # 1 of top 2
    assert precision_at_k(RANKING, REL, 4) == 0.5     # 2 of top 4


def test_reciprocal_rank():
    assert reciprocal_rank(RANKING, REL) == 0.5       # first hit at rank 2
    assert reciprocal_rank([9, 8, 7], REL) == 0.0     # nothing relevant
    assert reciprocal_rank([1, 9, 9], REL) == 1.0     # hit at rank 1


def test_dcg_matches_definition():
    # DCG@5 = 2/log2(3)  [A at rank 2]  +  1/log2(5)  [B at rank 4]
    expected = 2.0 / math.log2(3) + 1.0 / math.log2(5)
    assert dcg_at_k(RANKING, REL, 5) == pytest.approx(expected)


def test_ndcg_matches_definition():
    dcg = 2.0 / math.log2(3) + 1.0 / math.log2(5)
    # ideal ranking puts grade 2 first, then grade 1
    idcg = 2.0 / math.log2(2) + 1.0 / math.log2(3)
    assert ndcg_at_k(RANKING, REL, 5) == pytest.approx(dcg / idcg)


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([1, 2, 3, 4, 5], REL, 5) == pytest.approx(1.0)


def test_ndcg_respects_grades():
    """Putting the grade-2 item first must beat putting the grade-1 item first."""
    assert ndcg_at_k([1, 2], REL, 2) > ndcg_at_k([2, 1], REL, 2)


def test_empty_qrels_is_nan_not_zero():
    """A query with no judgements is UNDEFINED. Scoring it 0 would drag the
    mean down and silently understate your system."""
    assert math.isnan(recall_at_k(RANKING, {}, 5))
    assert math.isnan(ndcg_at_k(RANKING, {}, 5))


def test_binary_relevance_ndcg_degenerates():
    """With one relevant doc, nDCG carries no more information than the rank
    of that doc - which is exactly what MRR already tells you. This is why we
    report Recall and MRR on the BEIR-format subsets, not nDCG."""
    rel = {7: 1.0}
    a = ndcg_at_k([7, 1, 2], rel, 10)
    b = ndcg_at_k([1, 7, 2], rel, 10)
    assert a > b
    assert a == pytest.approx(1.0)
    assert b == pytest.approx(1.0 / math.log2(3))

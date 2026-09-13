"""
The behaviour this module exists to produce, pinned as tests.

Real failure that motivated it: searching "logical reasoning question"
ranked Zoom Q&A screenshots above an actual logic puzzle, because they
contained the common word "question" and the puzzle contained none of the
query words at all.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from contextfuse.adaptive import ALPHA_FLOOR, decide_alpha    # noqa: E402
from contextfuse.lexical import BM25                          # noqa: E402

# 40 documents. "question" is everywhere; "vellore" is in exactly one.
DOCS = ([f"this is a question about topic {i} with some options" for i in range(39)]
        + ["vit vellore campus h block ladies hostel"])
IDS = list(range(40))
BM = BM25(DOCS, IDS)


def test_rare_term_pushes_alpha_high():
    d = decide_alpha("vellore", BM)
    assert d.alpha > 0.7, d.explanation
    assert d.rarest_term == "vellore"
    assert d.rarest_df == 1


def test_common_term_pulls_alpha_low():
    """'question' is in 39 of 40 docs - it cannot discriminate."""
    d = decide_alpha("question", BM)
    assert d.alpha < 0.35, d.explanation


def test_query_with_no_known_terms_goes_fully_visual():
    """
    'a' IS in the index, so an earlier version counted it as lexical
    evidence and returned 0.156 instead of the floor. Filler words carry no
    information and must not be treated as evidence at all.
    """
    d = decide_alpha("a photograph of a golden retriever", BM)
    assert d.alpha == ALPHA_FLOOR
    assert d.known_terms == []
    assert "nothing useful to" in d.explanation


def test_filler_words_are_classified_as_uninformative_not_known():
    d = decide_alpha("a question of the options", BM)
    assert d.known_terms == []
    assert "a" in d.uninformative_terms
    assert d.alpha == ALPHA_FLOOR


def test_rare_term_survives_alongside_filler():
    """
    Filler must be ignored, not allowed to dilute a genuine rare match.

    NOTE the filler chosen here - "a", "is", "with" all appear in the toy
    corpus above. An earlier version of this test used "of" and "the", which
    appear in NO document, so they were correctly classified as UNKNOWN
    rather than filler. Unknown terms SHOULD reduce confidence: they are
    words that could have matched and did not. The test was wrong, not the
    code. Filler means "present but too common to discriminate".
    """
    bare = decide_alpha("vellore", BM).alpha
    padded = decide_alpha("a vellore is with", BM).alpha
    assert padded == bare, "corpus-common filler must not change the decision"


def test_unknown_terms_DO_reduce_confidence():
    """The other side of the same coin, so the distinction stays pinned."""
    bare = decide_alpha("vellore", BM).alpha
    missing = decide_alpha("vellore zzzz qqqq", BM).alpha
    assert missing < bare, "words that could have matched but did not are evidence"


def test_the_actual_failing_query_leans_visual():
    """The case from the real search. 'question' is common, the other two
    words appear nowhere - so this must NOT be treated as a lexical query."""
    d = decide_alpha("logical reasoning question", BM)
    assert d.alpha < 0.35, d.explanation
    assert set(d.unknown_terms) == {"logical", "reasoning"}
    assert "question" in d.uninformative_terms


def test_alpha_always_within_bounds():
    for q in ["", "vellore", "question", "zzzz", "vellore question campus"]:
        d = decide_alpha(q, BM)
        assert 0.0 <= d.alpha <= 1.0


def test_coverage_matters():
    """Matching the rare term plus context beats matching it alone."""
    alone = decide_alpha("vellore zzzz yyyy xxxx", BM).alpha
    covered = decide_alpha("vellore campus hostel", BM).alpha
    assert covered > alone

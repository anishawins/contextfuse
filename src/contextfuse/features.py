"""
Features for learning to rank.

A learned reranker does not look at images or text. It looks at NUMBERS
describing a (query, candidate) pair, produced by retrievers that already
ran. Its job is to learn how to weigh evidence that disagrees.

WHY RANK FEATURES MATTER AND RAW SCORES ARE NOT ENOUGH:
BM25 scores are unbounded and their scale depends on query length and term
rarity - a score of 12 means something different for a one-word query than a
six-word one. Cosine scores are bounded but compressed into a narrow band.
Neither is comparable ACROSS queries, which is exactly what a model trained
over many queries needs. Ranks and within-query normalised scores are
comparable by construction, which is why learning-to-rank systems have used
them since LETOR.

Every feature below is computed WITHIN a query, so a value means the same
thing for every query in the training set.
"""
from __future__ import annotations

import math

import numpy as np

from contextfuse.lexical import BM25, tokenize

FEATURE_NAMES = [
    "dense_score",          # raw cosine
    "dense_norm",           # min-max within this query's candidates
    "dense_z",              # z-score within this query
    "dense_rr",             # 1 / rank in the dense list
    "dense_margin_top",     # score minus the best dense score
    "lex_score",            # raw BM25
    "lex_norm",
    "lex_z",
    "lex_rr",
    "lex_margin_top",
    "lex_found",            # 1 if BM25 retrieved it at all
    "term_coverage",        # fraction of query terms present in the doc
    "max_idf_matched",      # rarest query term this doc actually contains
    "sum_idf_matched",
    "doc_len_norm",         # doc length / average doc length
    "query_len",            # number of query tokens
    "rank_agreement",       # do both retrievers rank it similarly?
]


def _norm(v: np.ndarray) -> np.ndarray:
    lo, hi = float(v.min()), float(v.max())
    return (v - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(v)


def _z(v: np.ndarray) -> np.ndarray:
    s = float(v.std())
    return (v - float(v.mean())) / s if s > 1e-12 else np.zeros_like(v)


def _rr(v: np.ndarray) -> np.ndarray:
    """Reciprocal of the rank implied by these scores (1 = best)."""
    order = np.argsort(-v)
    rank = np.empty(len(v), dtype=float)
    rank[order] = np.arange(1, len(v) + 1)
    return 1.0 / rank


def doc_term_sets(bm25: BM25) -> dict[int, set[str]]:
    """
    corpus_id -> set of terms it contains, built once.

    Without this, computing term coverage means scanning the inverted index
    for every (candidate, term) pair - about 40 million operations over the
    full run. Inverting it once costs one pass and makes each lookup O(1).
    """
    out: dict[int, set[str]] = {cid: set() for cid in bm25.ids}
    for term, postings in bm25.index.items():
        for doc_i, _ in postings:
            out[bm25.ids[doc_i]].add(term)
    return out


def build_features(query: str, candidates: list[int],
                   dense: dict[int, float], lex: dict[int, float],
                   bm25: BM25, doc_len: dict[int, int],
                   doc_terms: dict[int, set[str]] | None = None) -> np.ndarray:
    """Returns [n_candidates, len(FEATURE_NAMES)] float32."""
    if doc_terms is None:
        doc_terms = doc_term_sets(bm25)
    toks = tokenize(query)
    known = [t for t in toks if t in bm25.index]

    d = np.array([dense.get(c, 0.0) for c in candidates], dtype=np.float64)
    l = np.array([lex.get(c, 0.0) for c in candidates], dtype=np.float64)
    found = np.array([1.0 if c in lex else 0.0 for c in candidates])

    d_n, d_z, d_rr = _norm(d), _z(d), _rr(d)
    l_n, l_z, l_rr = _norm(l), _z(l), _rr(l)
    d_m, l_m = d - d.max(), l - l.max()

    cov, mx, sm = [], [], []
    for c in candidates:
        terms = doc_terms.get(c, set())
        present = [t for t in known if t in terms]
        cov.append(len(present) / max(1, len(toks)))
        idfs = [bm25.idf[t] for t in present]
        mx.append(max(idfs) if idfs else 0.0)
        sm.append(sum(idfs))

    dl = np.array([doc_len.get(c, 0) / max(1e-9, bm25.avgdl) for c in candidates])
    ql = np.full(len(candidates), float(len(toks)))
    agree = 1.0 - np.abs(d_rr - l_rr)

    return np.stack([d, d_n, d_z, d_rr, d_m,
                     l, l_n, l_z, l_rr, l_m, found,
                     np.array(cov), np.array(mx), np.array(sm),
                     dl, ql, agree], axis=1).astype(np.float32)

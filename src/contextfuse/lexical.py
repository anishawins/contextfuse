"""
BM25, implemented from the formula. No library.

BM25 scores a document for a query by summing, over the query's terms:

    score(D,Q) = SUM_q  IDF(q) * ( f(q,D) * (k1 + 1) )
                               / ( f(q,D) + k1 * (1 - b + b * |D| / avgdl) )

    IDF(q) = ln( 1 + (N - n(q) + 0.5) / (n(q) + 0.5) )

Three ideas, and you should be able to explain each one on its own:

1. TF - a document containing "postgres" five times beats one containing it
   once. But with DIMINISHING RETURNS: the f/(f + k1...) shape saturates, so
   the 20th occurrence adds almost nothing. k1 controls how fast it saturates.
   Raw term frequency without saturation is why naive TF ranking is bad.

2. IDF - a term appearing in nearly every document tells you nothing. "the"
   has IDF near zero; "psycopg2" has high IDF. Rare terms carry the signal.

3. LENGTH NORMALISATION - a long document contains more words, so it hits
   more query terms by accident. Dividing by |D|/avgdl penalises that.
   b = 0 turns it off entirely, b = 1 applies it fully; 0.75 is the usual
   compromise and is what Robertson's original work settled on.

WHY THIS MATTERS FOR CONTEXTFUSE: dense embeddings capture gist and lose
specifics. Ask for `FATAL: password authentication failed` and an embedding
sees "a database error" - indistinguishable from twenty other database
errors. BM25 matches the literal string. That is the gap the hybrid fills.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Sequence

TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """
    Lowercase, split on anything that is not a letter/digit/underscore.

    Deliberately keeps underscores and digits so identifiers survive:
    `max_length`, `psycopg2`, `utf8` stay whole. A tokenizer that split on
    underscores would destroy exactly the terms lexical search exists to
    catch. No stemming - "connection" and "connections" stay distinct,
    because for error strings and identifiers exactness beats recall.
    """
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, docs: Sequence[str], ids: Sequence[int],
                 k1: float = 1.5, b: float = 0.75):
        if len(docs) != len(ids):
            raise ValueError("docs and ids must be the same length")
        self.ids = list(ids)
        self.k1, self.b = k1, b

        toks = [tokenize(d or "") for d in docs]
        self.doc_len = [len(t) for t in toks]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.N = len(toks)

        # Inverted index: term -> [(doc_index, term_frequency), ...]
        # This is the data structure that makes lexical search fast. Instead
        # of scanning 1360 documents per query, you look up only the documents
        # that contain a query term - usually a tiny fraction of the corpus.
        self.index: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, t in enumerate(toks):
            for term, f in Counter(t).items():
                self.index[term].append((i, f))

        # df = in how many documents does this term appear
        self.idf = {
            term: math.log(1 + (self.N - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self.index.items()
        }

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Returns [(corpus_id, score), ...] best first."""
        scores: dict[int, float] = defaultdict(float)
        for term in tokenize(query):
            postings = self.index.get(term)
            if not postings:
                continue                       # term in no document: contributes 0
            idf = self.idf[term]
            for doc_i, f in postings:
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[doc_i] / self.avgdl)
                scores[doc_i] += idf * (f * (self.k1 + 1)) / denom
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [(self.ids[i], s) for i, s in top]

    def stats(self) -> dict:
        return {"documents": self.N, "vocabulary": len(self.index),
                "avg_doc_len": round(self.avgdl, 1),
                "k1": self.k1, "b": self.b}

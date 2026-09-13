"""Loading ViDoRe V3 Computer Science into plain Python structures."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from datasets import load_dataset

REPO = "vidore/vidore_v3_computer_science"


@dataclass
class Benchmark:
    corpus_ids: list[int]
    images: Any                                   # HF Dataset, lazily decoded
    queries: dict[int, str]                       # query_id -> text
    qrels: dict[int, dict[int, float]]            # query_id -> {corpus_id: grade}
    languages: dict[int, str] = field(default_factory=dict)
    need_of: dict[int, int] = field(default_factory=dict)   # query_id -> need_id

    def __repr__(self) -> str:
        return (f"Benchmark(pages={len(self.corpus_ids)}, "
                f"queries={len(self.queries)}, "
                f"needs={len(set(self.need_of.values())) or 'n/a'})")


def load_vidore_cs(language: str | None = "english") -> Benchmark:
    """
    Load the benchmark, optionally restricted to one language.

    WHY THE LANGUAGE FILTER DEFAULTS TO ENGLISH:
    the 1290 queries are ~215 information needs translated into 6 languages.
    CLIP's text encoder is English-only; SigLIP 2 is multilingual. Comparing
    them over all 1290 would mostly measure "can the model read French",
    not "which retrieves better". Pass language=None deliberately, as a
    separate multilingual experiment - never as the default.
    """
    corpus = load_dataset(REPO, "corpus")["test"]
    q = load_dataset(REPO, "queries")["test"]
    r = load_dataset(REPO, "qrels")["test"]

    qrels: dict[int, dict[int, float]] = defaultdict(dict)
    for row in r:
        qrels[row["query_id"]][row["corpus_id"]] = float(row["score"])

    # Group translations: queries sharing an identical relevant-page set are
    # the same information need. Verified empirically - see
    # scripts/check_query_independence.py (211 groups of 6, 2 of 12).
    need_of: dict[int, int] = {}
    seen: dict[frozenset, int] = {}
    for qid in sorted(qrels):
        key = frozenset(qrels[qid])
        need_of[qid] = seen.setdefault(key, len(seen))

    keep = {row["query_id"] for row in q
            if language is None or row["language"] == language}
    queries = {row["query_id"]: row["query"] for row in q
               if row["query_id"] in keep}
    languages = {row["query_id"]: row["language"] for row in q
                 if row["query_id"] in keep}

    return Benchmark(
        corpus_ids=list(corpus["corpus_id"]),
        images=corpus,
        queries=queries,
        qrels={k: v for k, v in qrels.items() if k in keep},
        languages=languages,
        need_of={k: v for k, v in need_of.items() if k in keep},
    )

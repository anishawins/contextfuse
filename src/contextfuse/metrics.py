"""
Retrieval metrics, written from the definitions.

Every function takes:
    ranking : list of corpus_ids, best first, as your system returned them
    rel     : dict {corpus_id: relevance_score}, ONLY judged-relevant items.
              Anything absent is treated as score 0 (irrelevant).

That "anything absent is irrelevant" line is an assumption, not a fact.
ViDoRe judges ~4.9 pages per query out of 1360. The other ~1355 were never
looked at. Some of them might be relevant. This is called INCOMPLETE
JUDGEMENTS and every benchmark in this field has it. Say so in your report.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def recall_at_k(ranking: Sequence[int], rel: dict[int, float], k: int) -> float:
    """
    Fraction of all known-relevant items that appear in the top k.

        Recall@k = |relevant ∩ top-k| / |relevant|

    Worked example: rel = {A:2, B:1}, ranking = [C, A, D, B, E]
        Recall@1 = 0/2 = 0.0   (only C in top 1, not relevant)
        Recall@5 = 2/2 = 1.0   (both A and B appear)

    Why you care: Recall@10 answers "did the right thing make it onto the
    first screen at all". If recall is low, no amount of reranking saves you
    - reranking can only reorder what retrieval already found.
    """
    if not rel:
        return float("nan")           # undefined, not zero. Do not average NaNs in.
    hits = sum(1 for cid in ranking[:k] if cid in rel)
    return hits / len(rel)


def precision_at_k(ranking: Sequence[int], rel: dict[int, float], k: int) -> float:
    """Fraction of the top k that is relevant. P@k = |relevant ∩ top-k| / k."""
    if k <= 0:
        raise ValueError("k must be positive")
    return sum(1 for cid in ranking[:k] if cid in rel) / k


def reciprocal_rank(ranking: Sequence[int], rel: dict[int, float],
                    k: int | None = None) -> float:
    """
    1 / (rank of the first relevant item). 0 if none found.

    Worked example: rel = {A:2, B:1}, ranking = [C, A, D, B, E]
        first relevant is A at rank 2  ->  RR = 1/2 = 0.5

    Averaged over queries this is MRR. It only looks at the FIRST hit, so it
    measures "how fast does the user get one good result" and is blind to
    everything after that. Use it alongside recall, never instead of it.
    """
    limit = len(ranking) if k is None else k
    for i, cid in enumerate(ranking[:limit], start=1):
        if cid in rel:
            return 1.0 / i
    return 0.0


def dcg_at_k(ranking: Sequence[int], rel: dict[int, float], k: int) -> float:
    """
    Discounted Cumulative Gain.

        DCG@k = sum_{i=1..k}  rel_i / log2(i + 1)

    The log2(i+1) denominator is the "discount": a relevant item at rank 1
    is worth its full score, at rank 2 it is worth 1/log2(3) = 0.63 of it,
    at rank 10 only 1/log2(11) = 0.29. That encodes "lower down is less
    useful to a human", which plain recall does not.
    """
    return sum(rel.get(cid, 0.0) / math.log2(i + 1)
               for i, cid in enumerate(ranking[:k], start=1))


def ndcg_at_k(ranking: Sequence[int], rel: dict[int, float], k: int) -> float:
    """
    nDCG@k = DCG@k / IDCG@k, where IDCG is the DCG of the perfect ranking.

    Dividing by the ideal makes queries comparable: a query with one relevant
    page and a query with twenty both top out at 1.0.

    IMPORTANT for ViDoRe: scores are graded {1, 2}, so this is meaningful.
    On a BINARY benchmark (ViDoRe V1 BEIR, one relevant doc per query) nDCG
    collapses to a monotone function of the first hit's rank - it tells you
    nothing MRR did not already say. Report Recall and MRR there instead.
    """
    if not rel:
        return float("nan")
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    if idcg == 0:
        return float("nan")
    return dcg_at_k(ranking, rel, k) / idcg


# ----------------------------------------------------------------------
# Aggregation with honest error bars
# ----------------------------------------------------------------------
def bootstrap_ci(per_query: Iterable[float], n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """
    Mean plus a percentile bootstrap confidence interval.

    Returns (mean, lo, hi).

    WHY THIS EXISTS, and it is the single most important comment in this file:

    You have ~215 English information needs. A bare mean like "MRR = 0.62"
    invites the reader to believe it is precise. It is not. Resampling the
    queries with replacement 2000 times shows how much that 0.62 would move
    if you had drawn a different 215 queries from the same population.

    THE TRAP: ViDoRe's 1290 queries are 215 needs translated into 6 languages.
    Bootstrapping over 1290 rows treats six translations of one question as
    six independent observations. They are not - they share the same relevant
    pages. Your interval would come out roughly sqrt(6) times too narrow and
    you would claim significance you do not have.

    So: resample INFORMATION NEEDS, never query rows.
    """
    vals = np.asarray([v for v in per_query if not math.isnan(v)], dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    means = vals[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(vals.mean()), float(lo), float(hi)


def evaluate(rankings: dict[int, Sequence[int]],
             qrels: dict[int, dict[int, float]],
             ks: Sequence[int] = (1, 5, 10),
             seed: int = 0) -> dict[str, dict[str, float]]:
    """
    Score a full run. `rankings` and `qrels` are both keyed by query_id.

    Returns {metric_name: {"mean":..., "lo":..., "hi":..., "n":...}}.
    """
    qids = [q for q in qrels if q in rankings]
    out: dict[str, dict[str, float]] = {}

    def add(name: str, values: list[float]) -> None:
        mean, lo, hi = bootstrap_ci(values, seed=seed)
        out[name] = {"mean": mean, "lo": lo, "hi": hi, "n": len(values)}

    for k in ks:
        add(f"recall@{k}", [recall_at_k(rankings[q], qrels[q], k) for q in qids])
        add(f"precision@{k}", [precision_at_k(rankings[q], qrels[q], k) for q in qids])
    add("mrr@10", [reciprocal_rank(rankings[q], qrels[q], 10) for q in qids])
    add("ndcg@10", [ndcg_at_k(rankings[q], qrels[q], 10) for q in qids])
    return out


def format_table(results: dict[str, dict[str, float]], title: str = "") -> str:
    """Pretty-print a result dict with its confidence intervals."""
    lines = []
    if title:
        lines += [title, "-" * len(title)]
    for name, r in results.items():
        lines.append(f"  {name:<14} {r['mean']:.4f}   "
                     f"95% CI [{r['lo']:.4f}, {r['hi']:.4f}]   n={r['n']}")
    return "\n".join(lines)

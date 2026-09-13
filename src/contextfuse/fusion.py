"""
Combining scores from retrievers that do not speak the same language.

THE PROBLEM, and this is a guaranteed interview question:

    BM25 scores are unbounded positives. On this corpus they run roughly
    0 to 30, depending on query length, term rarity and document length.
    Cosine scores are bounded in [-1, 1], and in practice here they cluster
    tightly around 0.11 with a spread of about 0.01.

    Write  S = bm25 + cosine  and the cosine term is arithmetically
    invisible. You have not built a hybrid; you have built BM25 with a
    rounding error attached. It will look like fusion "did nothing", and
    you will draw the wrong conclusion from your own experiment.

So scores must be mapped onto a common scale BEFORE they are combined.
Two standard ways, both implemented here:

  MIN-MAX per query  - rescale each retriever's candidate scores to [0,1].
      Simple, keeps relative distances, but one outlier score squashes
      everything else toward zero.

  RECIPROCAL RANK FUSION (RRF) - throw the scores away, keep only ranks:
          RRF(d) = SUM_retrievers  1 / (k + rank_r(d))
      Scale-free by construction, so no normalisation is needed at all, and
      it is robust when one retriever is much worse than the other. The cost
      is that it discards confidence: a document ranked 1 with a huge margin
      and one ranked 1 by a hair count identically.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence


def minmax(scores: dict[int, float]) -> dict[int, float]:
    """Rescale to [0,1] within this query's candidates."""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-12:
        return {k: 0.0 for k in scores}          # flat: contributes nothing
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def weighted_fusion(runs: Sequence[dict[int, float]],
                    weights: Sequence[float],
                    k: int = 10) -> list[int]:
    """
    Min-max normalise each run, then take a weighted sum.

    A document missing from one run scores 0 for that run rather than being
    dropped - otherwise only documents both retrievers found could ever win,
    which defeats the point of combining them.
    """
    if len(runs) != len(weights):
        raise ValueError("one weight per run")
    total: dict[int, float] = defaultdict(float)
    for run, w in zip(runs, weights):
        if w == 0:
            continue
        for doc, s in minmax(run).items():
            total[doc] += w * s
    return [d for d, _ in sorted(total.items(), key=lambda kv: -kv[1])[:k]]


def rrf(runs: Sequence[Sequence[int]], k_const: int = 60,
        k: int = 10) -> list[int]:
    """
    Reciprocal Rank Fusion over ranked ID lists.

    k_const = 60 is the value from Cormack et al. (2009), where RRF was
    introduced. It damps the influence of the very top ranks so a single
    confident-but-wrong retriever cannot dominate. Treat 60 as a convention
    you inherited and can tune, not a law of nature.
    """
    total: dict[int, float] = defaultdict(float)
    for run in runs:
        for rank, doc in enumerate(run, start=1):
            total[doc] += 1.0 / (k_const + rank)
    return [d for d, _ in sorted(total.items(), key=lambda kv: -kv[1])[:k]]

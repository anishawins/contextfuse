"""
A learned reranker, trained with a pairwise ranking loss.

WHY PAIRWISE AND NOT "PREDICT RELEVANCE 0/1":
treating this as binary classification (pointwise) optimises the wrong thing.
Ranking does not care about absolute scores at all - only about ORDER. A
model that scores every relevant doc 0.51 and every irrelevant one 0.49 is a
terrible classifier and a perfect ranker.

RankNet (Burges et al., 2005) formalises that. For every pair (i, j) where i
is more relevant than j, it asks the model to score i above j:

    L = SUM over pairs  -log sigmoid( s_i - s_j )

Only the DIFFERENCE of scores appears, so the model is free to put them on
any scale it likes. And because pairs are formed WITHIN a query, the loss
never compares a document from one query against another - which is exactly
the mistake raw-score fusion makes.

Graded relevance comes free: ViDoRe grades pages 1 or 2, so a grade-2 page
forms a training pair against grade-1 pages as well as against irrelevant
ones, and the model learns the finer distinction.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Reranker(nn.Module):
    """Deliberately small: ~10k training pairs cannot support more."""

    def __init__(self, n_features: int, hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def ranknet_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    scores, labels: [n_candidates] for ONE query.

    Builds every ordered pair where labels differ and asks the model to get
    the direction right. Returns 0 when a query has no usable pair (all
    candidates equally relevant), rather than NaN.
    """
    diff_s = scores[:, None] - scores[None, :]
    diff_l = labels[:, None] - labels[None, :]
    mask = diff_l > 0                       # i strictly more relevant than j
    if not mask.any():
        return scores.sum() * 0.0           # keeps the graph, contributes nothing
    return -F.logsigmoid(diff_s[mask]).mean()

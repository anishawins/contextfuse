"""
The SigLIP sigmoid contrastive loss, implemented from the paper's definition.

WHAT CONTRASTIVE LEARNING DOES:
you have pairs that belong together - an image and its caption. Take a batch
of N such pairs and compute every image-text similarity, giving an N x N
matrix. The N diagonal entries are the true pairs. The N^2 - N off-diagonal
entries are pairs that do NOT belong together. Train so the diagonal scores
high and everything else scores low. No labels needed beyond "these two came
from the same row" - the other examples in the batch supply the negatives.

CLIP'S SOFTMAX LOSS vs SIGLIP'S SIGMOID LOSS - this is the interview question:

  CLIP applies softmax over each row and column, then cross-entropy. Softmax
  is NORMALISED ACROSS THE BATCH: raising one similarity necessarily lowers
  the others. So every pair's gradient depends on every other pair, the loss
  needs a global view of the batch, and quality depends heavily on batch size
  (CLIP used 32,768).

  SigLIP treats each of the N^2 cells as an INDEPENDENT binary question:
  "do these two belong together, yes or no?" Binary cross-entropy on each
  cell, no normalisation across the batch. Cells are independent, so the loss
  works at far smaller batch sizes and needs no all-gather across devices.

  That is the whole reason SigLIP is the sensible choice for a project on one
  laptop: the objective was designed to survive small batches.

THE FORMULA (Zhai et al., "Sigmoid Loss for Language Image Pre-Training"):

    z_ij = t * (x_i . y_j) + b
    L    = -(1/N) * SUM_i SUM_j  log sigmoid( label_ij * z_ij )

    label_ij = +1 when i == j (a true pair), -1 otherwise.

  t is a learnable temperature and b a learnable bias. The bias matters: the
  matrix is overwhelmingly negatives (N positives against N^2 - N negatives),
  so without a bias initialised negative the model starts by predicting
  "everything matches" and wastes early training climbing out of that hole.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidContrastiveLoss(nn.Module):
    """
    Args:
        init_logit_scale: log of the initial temperature. log(10) ~ 2.3,
            matching the SigLIP paper's initialisation.
        init_logit_bias: initial bias. Strongly negative (-10) because the
            similarity matrix is ~1/N positives.
        learnable: if False, temperature and bias stay fixed - useful as an
            ablation to show they actually matter.
    """

    def __init__(self, init_logit_scale: float = 2.3026,
                 init_logit_bias: float = -10.0, learnable: bool = True):
        super().__init__()
        t = torch.tensor(float(init_logit_scale))
        b = torch.tensor(float(init_logit_bias))
        if learnable:
            self.logit_scale = nn.Parameter(t)
            self.logit_bias = nn.Parameter(b)
        else:
            self.register_buffer("logit_scale", t)
            self.register_buffer("logit_bias", b)

    def forward(self, img: torch.Tensor, txt: torch.Tensor,
                positives: torch.Tensor | None = None) -> torch.Tensor:
        """
        img: [M, D], txt: [N, D], both expected L2-normalised.
        positives: optional [N, M] boolean matrix of true pairs. Defaults to
            the identity, which requires M == N.

        THE ASYMMETRIC CASE MATTERS HERE. For caption->image training, M == N
        and the identity is right. For ViDoRe, a batch is N queries scored
        against ALL M pages in the corpus, and a query can have several
        relevant pages. Because the sigmoid loss treats every cell as an
        independent yes/no question, multiple positives per row need no
        special handling at all - you just mark them. A softmax loss would
        need the probability mass split across them, which is one more
        practical reason the sigmoid objective suits this project.

        Passing the matrix explicitly also fixes false negatives: two
        captions of the SAME image in one batch are both correct, and
        scoring the off-diagonal one as negative teaches the model that a
        right answer is wrong.
        """
        if positives is None and img.shape != txt.shape:
            raise ValueError(
                f"shape mismatch {img.shape} vs {txt.shape} and no positives "
                f"matrix given - cannot assume the identity")
        if img.shape[-1] != txt.shape[-1]:
            raise ValueError(f"dim mismatch: {img.shape[-1]} vs {txt.shape[-1]}")

        logits = self.logit_scale.exp() * (txt @ img.t()) + self.logit_bias

        if positives is None:
            positives = torch.eye(img.shape[0], dtype=torch.bool, device=img.device)
        elif positives.shape != logits.shape:
            raise ValueError(
                f"positives {tuple(positives.shape)} must match the logit "
                f"matrix {tuple(logits.shape)} = (n_text, n_image)")
        # +1 for true pairs, -1 for everything else
        labels = torch.where(positives, 1.0, -1.0)

        # logsigmoid is numerically stable; log(sigmoid(x)) is not
        # Normalise by the number of QUERIES (rows), not cells: otherwise a
        # batch scored against 1360 pages looks 1360x worse than the same
        # batch scored against 32, and the loss is not comparable between
        # configurations.
        return -F.logsigmoid(labels * logits).sum() / logits.shape[0]


class InfoNCELoss(nn.Module):
    """
    CLIP's softmax objective, for the ablation that shows why SigLIP's
    differs. Symmetric cross-entropy over rows and columns.
    """

    def __init__(self, init_logit_scale: float = 2.6593):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(float(init_logit_scale)))

    def forward(self, img: torch.Tensor, txt: torch.Tensor,
                positives: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.logit_scale.exp().clamp(max=100.0) * (txt @ img.t())
        target = torch.arange(img.shape[0], device=img.device)
        return 0.5 * (F.cross_entropy(logits, target)
                      + F.cross_entropy(logits.t(), target))

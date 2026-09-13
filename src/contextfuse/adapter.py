"""
A small trainable head on top of frozen SigLIP embeddings.

WHY AN ADAPTER RATHER THAN FINE-TUNING THE ENCODER:
the encoder is frozen, so its output for a given image never changes and can
be computed once and cached. Training then touches only this head, which
costs seconds per epoch instead of minutes. You can run twenty experiments
in an afternoon.

The limit is equally clear and must go in your report: a head can only
re-shape the space the encoder already produced. If the encoder discarded
information - and at 224x224 it discards the text on a document page - no
head can recover it. That is what Stage 2 (LoRA inside the ViT) is for.

ARCHITECTURE: a residual bottleneck MLP.

    out = normalise( x + W2 · gelu( W1 · LayerNorm(x) ) )

The RESIDUAL matters. Initialised so the MLP branch starts near zero, the
adapter begins as an identity function - the model starts exactly at
zero-shot SigLIP performance and can only improve from there. Without it,
a randomly initialised head would destroy the pretrained geometry and spend
its first epochs climbing back to where it started.

Separate heads for image and text: the two modalities have different
statistics, and a shared head has to compromise between them. Worth an
ablation - `--shared-head` runs that comparison.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int = 256, dropout: float = 0.1,
                 init_scale: float = 1e-4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.drop = nn.Dropout(dropout)
        # Near-zero init on the output layer => starts as identity.
        # The residual branch sums `bottleneck` terms, so its magnitude grows
        # with sqrt(bottleneck): at std 1e-3 and bottleneck 256 the output
        # already drifts ~0.017 per component. 1e-4 keeps the starting point
        # indistinguishable from zero-shot at any sensible bottleneck size.
        nn.init.normal_(self.up.weight, std=init_scale)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up(self.drop(F.gelu(self.down(self.norm(x)))))
        return F.normalize(x + h, p=2, dim=-1)


class DualAdapter(nn.Module):
    """One adapter per modality (or one shared, for the ablation)."""

    def __init__(self, dim: int, bottleneck: int = 256, dropout: float = 0.1,
                 shared: bool = False):
        super().__init__()
        self.image = ResidualAdapter(dim, bottleneck, dropout)
        self.text = self.image if shared else ResidualAdapter(dim, bottleneck, dropout)
        self.shared = shared

    def forward(self, img: torch.Tensor, txt: torch.Tensor):
        return self.image(img), self.text(txt)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

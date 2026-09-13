"""
LoRA: Low-Rank Adaptation, implemented by hand.
Hu et al. (2021), "LoRA: Low-Rank Adaptation of Large Language Models".

THE PROBLEM:
SigLIP 2 base has ~400M parameters. Fine-tuning all of them on a laptop means
holding the weights, their gradients, and two Adam moment buffers per weight -
roughly 4x the model in memory - and it overfits immediately on a small
dataset. But the alternative we already tried, a head on FROZEN embeddings,
cannot help when the encoder has discarded the information (measured: ~40
configurations, no improvement).

THE IDEA:
a fine-tuning update to a weight matrix W is empirically LOW RANK - it does
not need all d x k degrees of freedom to express the change. So instead of
learning a full update dW, learn two thin matrices:

        W' = W + (alpha / r) * B @ A

    W : [d, k]  frozen, never updated
    A : [r, k]  trainable, initialised from a small normal distribution
    B : [d, r]  trainable, initialised to ZERO
    r : the rank, typically 4 to 16

Parameters trained: r*(d + k) instead of d*k. For a 768x768 attention
projection at r=8 that is 12,288 instead of 589,824 - a 48x reduction.

WHY B STARTS AT ZERO, and this is the question after "what is LoRA":
B @ A = 0 at initialisation, so W' == W exactly. The adapted model begins
IDENTICAL to the pretrained one, and any change from that point is
attributable to training. Initialising both randomly would corrupt the
pretrained weights before the first step - which is precisely the failure we
measured with the residual adapter, where training destroyed a good
representation in its first epoch.

WHY alpha/r: it decouples the learning rate from the rank. Doubling r doubles
the number of summed terms in B@A, so without the scaling the effective update
magnitude would change every time you changed the rank, and every
hyperparameter would need retuning.

AT INFERENCE B@A can be folded into W, so a deployed LoRA model costs nothing
extra - unlike an adapter layer, which adds depth to every forward pass.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank update."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16,
                 dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)          # the pretrained weights never move

        self.r, self.alpha = r, alpha
        self.scaling = alpha / r
        d_out, d_in = base.out_features, base.in_features

        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        # Kaiming init on A, zeros on B => B@A == 0 => identical to base at step 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        # x @ A.T @ B.T, kept in that order so the r-dimensional bottleneck is
        # materialised rather than the full d_out x d_in update matrix
        return out + self.drop(x) @ self.lora_A.t() @ self.lora_B.t() * self.scaling

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """Fold the update into the base weights. Inference cost returns to zero."""
        merged = nn.Linear(self.base.in_features, self.base.out_features,
                           bias=self.base.bias is not None)
        merged.weight.copy_(self.base.weight + self.scaling * (self.lora_B @ self.lora_A))
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
        return merged

    def extra_repr(self) -> str:
        return (f"r={self.r}, alpha={self.alpha}, "
                f"trainable={self.lora_A.numel() + self.lora_B.numel():,}")


def apply_lora(model: nn.Module, target_names: tuple[str, ...] = ("q_proj", "v_proj"),
               r: int = 8, alpha: int = 16, dropout: float = 0.0) -> dict:
    """
    Freeze `model` and replace the named nn.Linear submodules with LoRALinear.

    WHICH LAYERS: the LoRA paper adapts the attention QUERY and VALUE
    projections and finds that sufficient. The intuition is that q_proj changes
    what each position asks for and v_proj changes what it contributes, which
    together cover most of what an attention layer needs to re-learn; k_proj is
    largely redundant with q_proj, and the MLP blocks are far larger for less
    benefit per parameter.

    Returns a summary dict - print it, and put the numbers in the report.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    replaced, lora_params = [], 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in target_names:
                wrapped = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
                setattr(module, child_name, wrapped)
                replaced.append(f"{name}.{child_name}" if name else child_name)
                lora_params += wrapped.lora_A.numel() + wrapped.lora_B.numel()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "layers_adapted": len(replaced),
        "targets": list(target_names),
        "rank": r, "alpha": alpha,
        "lora_parameters": lora_params,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total if total else 0.0,
        "example_layers": replaced[:4],
    }


def merge_all(model: nn.Module) -> int:
    """Fold every LoRALinear back into a plain nn.Linear. Returns the count."""
    n = 0
    for _, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, child_name, child.merge())
                n += 1
    return n

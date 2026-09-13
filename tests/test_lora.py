"""
Properties LoRA must have. These are the interview answers as assertions.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from contextfuse.lora import LoRALinear, apply_lora, merge_all   # noqa: E402


def test_identity_at_initialisation():
    """
    THE POINT OF ZERO-INITIALISING B. The adapted layer must reproduce the
    pretrained layer exactly before training, so any later change is
    attributable to learning rather than to random weights corrupting a good
    representation.
    """
    base = nn.Linear(64, 32)
    lora = LoRALinear(base, r=8)
    lora.eval()
    x = torch.randn(5, 64)
    assert torch.allclose(lora(x), base(x), atol=1e-6)


def test_base_weights_are_frozen():
    base = nn.Linear(64, 32)
    lora = LoRALinear(base, r=4)
    assert not lora.base.weight.requires_grad
    assert lora.lora_A.requires_grad and lora.lora_B.requires_grad


def test_gradients_reach_only_the_low_rank_matrices():
    base = nn.Linear(64, 32)
    lora = LoRALinear(base, r=4)
    lora(torch.randn(5, 64)).sum().backward()
    assert lora.lora_A.grad is not None
    assert lora.lora_B.grad is not None
    assert lora.base.weight.grad is None


def test_parameter_count_is_r_times_d_plus_k():
    d_in, d_out, r = 768, 768, 8
    lora = LoRALinear(nn.Linear(d_in, d_out), r=r)
    trainable = lora.lora_A.numel() + lora.lora_B.numel()
    assert trainable == r * (d_in + d_out) == 12288
    assert trainable < d_in * d_out / 40        # 48x fewer than a full update


def test_merging_preserves_the_function():
    """B@A folds into W, so a deployed model pays no inference cost."""
    base = nn.Linear(48, 24)
    lora = LoRALinear(base, r=4)
    with torch.no_grad():                       # give it a non-zero update
        lora.lora_B.normal_(std=0.05)
    lora.eval()
    x = torch.randn(7, 48)
    before = lora(x)
    after = lora.merge()(x)
    assert torch.allclose(before, after, atol=1e-5)


def test_alpha_over_r_decouples_scale_from_rank():
    """
    Without the alpha/r scaling, changing the rank would change the update
    magnitude and every other hyperparameter would need retuning.
    """
    assert LoRALinear(nn.Linear(8, 8), r=4, alpha=16).scaling == 4.0
    assert LoRALinear(nn.Linear(8, 8), r=8, alpha=16).scaling == 2.0
    assert LoRALinear(nn.Linear(8, 8), r=16, alpha=16).scaling == 1.0


def test_apply_lora_targets_only_named_layers():
    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(32, 32)
            self.k_proj = nn.Linear(32, 32)
            self.v_proj = nn.Linear(32, 32)
            self.out_proj = nn.Linear(32, 32)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Attn() for _ in range(3)])

    net = Net()
    info = apply_lora(net, ("q_proj", "v_proj"), r=8)
    assert info["layers_adapted"] == 6                    # 2 per block x 3
    assert isinstance(net.layers[0].q_proj, LoRALinear)
    assert isinstance(net.layers[0].k_proj, nn.Linear)     # untouched
    assert merge_all(net) == 6
    assert isinstance(net.layers[0].q_proj, nn.Linear)     # folded back


def test_trainable_fraction_at_realistic_width():
    """
    The parameter saving only means anything at realistic widths. On a 32-dim
    toy layer rank 8 is a quarter of the dimension and LoRA saves nothing -
    an earlier version of this file asserted a fraction threshold against
    exactly that toy model and failed for a reason that had nothing to do
    with the implementation.

    At transformer width (768) the saving is the whole point.
    """
    class Attn(nn.Module):
        def __init__(self, d=768):
            super().__init__()
            self.q_proj = nn.Linear(d, d)
            self.k_proj = nn.Linear(d, d)
            self.v_proj = nn.Linear(d, d)
            self.out_proj = nn.Linear(d, d)

    class Block(nn.Module):
        def __init__(self, d=768):
            super().__init__()
            self.self_attn = Attn(d)
            self.mlp = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(),
                                     nn.Linear(d * 4, d))

    class Tower(nn.Module):
        def __init__(self, n=12):
            super().__init__()
            self.layers = nn.ModuleList([Block() for _ in range(n)])

    info = apply_lora(Tower(), ("q_proj", "v_proj"), r=8)
    assert info["layers_adapted"] == 24
    assert info["trainable_fraction"] < 0.01, info["trainable_fraction"]
    # r*(d_in + d_out) per adapted layer
    assert info["lora_parameters"] == 24 * 8 * (768 + 768)


def test_rank_must_be_positive():
    with pytest.raises(ValueError):
        LoRALinear(nn.Linear(8, 8), r=0)

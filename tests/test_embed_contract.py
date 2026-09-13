"""
Guards the model-output contract that broke on transformers 5.

The bug: get_image_features returned a tensor in transformers 4.x and an
output OBJECT in 5.x. Code that assumed a tensor crashed. These tests pin
the unwrapping behaviour so a future version bump fails HERE, in 0.1s,
instead of 40 minutes into an embedding run.

No model download - the tensor shapes are faked.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from contextfuse.embed import SigLIPEncoder                       # noqa: E402


class FakeOutput:
    """Stands in for BaseModelOutputWithPooling."""
    def __init__(self, pooled):
        self.pooler_output = pooled
        self.last_hidden_state = torch.randn(pooled.shape[0], 7, pooled.shape[1])

    def keys(self):
        return ["last_hidden_state", "pooler_output"]


def test_pool_accepts_bare_tensor():
    t = torch.randn(3, 768)
    assert torch.equal(SigLIPEncoder._pool(t), t)


def test_pool_unwraps_output_object():
    t = torch.randn(3, 768)
    assert torch.equal(SigLIPEncoder._pool(FakeOutput(t)), t)


def test_pool_rejects_anything_else_loudly():
    """Silence here would mean garbage embeddings, not a crash."""
    with pytest.raises(TypeError):
        SigLIPEncoder._pool({"not": "an output"})


def test_l2_produces_unit_vectors():
    x = torch.randn(16, 768) * 50          # deliberately large magnitudes
    n = SigLIPEncoder._l2(x).norm(dim=-1)
    assert torch.allclose(n, torch.ones(16), atol=1e-5)


def test_l2_preserves_direction():
    """Normalising must not change WHICH vector is most similar - only scale."""
    a = torch.randn(1, 768)
    scaled = a * 17.0
    assert torch.allclose(SigLIPEncoder._l2(a), SigLIPEncoder._l2(scaled), atol=1e-5)

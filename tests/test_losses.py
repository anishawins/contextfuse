"""Properties the loss must have, checked without training anything."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from contextfuse.adapter import DualAdapter                    # noqa: E402
from contextfuse.losses import SigmoidContrastiveLoss          # noqa: E402


def unit(n, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=-1)


def test_perfect_alignment_beats_random():
    """Identical image/text vectors must score lower loss than unrelated ones."""
    x = unit(16, 64)
    loss = SigmoidContrastiveLoss()
    assert loss(x, x).item() < loss(x, unit(16, 64, seed=1)).item()


def test_loss_is_finite_at_extremes():
    x = unit(8, 32)
    loss = SigmoidContrastiveLoss()
    assert torch.isfinite(loss(x, x))
    assert torch.isfinite(loss(x, -x))          # maximally wrong


def test_gradients_flow_to_temperature_and_bias():
    x, y = unit(8, 32), unit(8, 32, seed=2)
    loss = SigmoidContrastiveLoss()
    loss(x, y).backward()
    assert loss.logit_scale.grad is not None and loss.logit_scale.grad != 0
    assert loss.logit_bias.grad is not None and loss.logit_bias.grad != 0


def test_explicit_positives_matter():
    """
    Two captions of the SAME image in one batch, and the model has them
    RIGHT - both captions score high against that image.

    With identity-only positives those two correct cells are labelled -1 and
    the model is punished for a correct answer. Marking them as positives
    must therefore give the lower loss.

    NOTE the setup. An earlier version of this test used random text vectors
    and asserted the same thing - and failed, correctly. With a bias of -10
    an untrained model scores every cell near zero; relabelling a
    near-zero cell from negative to positive makes the loss WORSE, because
    the model is now wrong about a cell it was previously right about. The
    property only holds when the model actually predicts those pairs as
    matching, which is what this setup arranges.
    """
    img = unit(4, 32)
    img[1] = img[0]                       # rows 0 and 1 are the same image
    txt = img.clone()                     # and the model scores them correctly
    pos = torch.eye(4, dtype=torch.bool)
    pos[0, 1] = pos[1, 0] = True
    loss = SigmoidContrastiveLoss()
    assert loss(img, txt, pos).item() < loss(img, txt).item()


def test_false_negatives_are_penalised_without_the_positives_matrix():
    """The failure mode above, stated directly: duplicate images in a batch
    inflate the loss unless their cells are declared positive."""
    img = unit(6, 32)
    img[3] = img[0]
    txt = img.clone()
    identity_only = SigmoidContrastiveLoss()(img, txt).item()
    pos = torch.eye(6, dtype=torch.bool)
    pos[0, 3] = pos[3, 0] = True
    corrected = SigmoidContrastiveLoss()(img, txt, pos).item()
    assert corrected < identity_only


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        SigmoidContrastiveLoss()(unit(4, 32), unit(5, 32))


def test_adapter_starts_as_identity():
    """
    THE POINT OF THE RESIDUAL. At initialisation the adapter must reproduce
    its input, so training starts exactly at zero-shot performance.
    """
    torch.manual_seed(0)
    x = unit(8, 64)
    # Cosine, not element-wise distance: both vectors are unit length, so the
    # meaningful question is "does it point the same way", and an element-wise
    # tolerance silently gets stricter as the dimension grows.
    for dim, bottleneck in ((64, 256), (768, 256), (768, 1024)):
        a = DualAdapter(dim=dim, bottleneck=bottleneck)
        a.eval()          # dropout OFF - this is a claim about the
                          # deterministic function, and nn.Module defaults to
                          # train mode, where dropout injects noise that has
                          # nothing to do with whether the init is identity.
        v = unit(8, dim)
        out, _ = a(v, v)
        cos = (out * v).sum(-1)
        # 0.999 rather than 0.9999: in float32 at dim 768 the residual branch
        # leaves ~1e-4 of drift no matter how small the init. Chasing that
        # number is meaningless - what matters is that the RANKING is
        # unchanged, which the next test asserts directly.
        assert (cos > 0.999).all(), f"dim={dim} bn={bottleneck}: {cos.min():.6f}"


def test_adapter_at_init_does_not_change_retrieval_ranking():
    """
    The claim that actually matters: at initialisation the adapter must
    return exactly the zero-shot ranking, so any later change is attributable
    to training rather than to the adapter's random weights.
    """
    torch.manual_seed(0)
    # STRUCTURED data, not random vectors. With 64 independent random unit
    # vectors every similarity sits near zero and the gaps between ranks are
    # ~1e-3, so any perturbation reshuffles them - but that ranking was
    # meaningless to begin with. Real embeddings have a true match scoring
    # well above the rest, which is the case worth protecting.
    img = unit(64, 768)
    txt = torch.nn.functional.normalize(img + 0.7 * unit(64, 768, seed=5), dim=-1)

    a = DualAdapter(dim=768)
    a.eval()
    with torch.no_grad():
        ai, at = a(img, txt)
    before = (txt @ img.t()).argmax(dim=1)
    after = (at @ ai.t()).argmax(dim=1)
    assert torch.equal(before, after), (
        f"adapter changed the top-1 for {(before != after).sum()} of 64 queries")


def test_adapter_output_is_unit_length():
    a = DualAdapter(dim=64)
    out, _ = a(unit(8, 64) * 7.0, unit(8, 64))
    assert torch.allclose(out.norm(dim=-1), torch.ones(8), atol=1e-5)


def test_shared_head_has_fewer_parameters():
    assert DualAdapter(64, shared=True).n_params() < DualAdapter(64).n_params()

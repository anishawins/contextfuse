import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from contextfuse.stats import paired_bootstrap          # noqa: E402


def test_identical_systems_show_no_difference():
    a = [0.1, 0.5, 0.9, 0.3] * 25
    r = paired_bootstrap(a, a)
    assert r["mean_diff"] == 0.0
    assert not r["significant"]
    assert r["ties"] == len(a)


def test_uniform_improvement_is_detected():
    a = [0.1, 0.2, 0.3, 0.4] * 25
    b = [x + 0.1 for x in a]
    r = paired_bootstrap(a, b)
    assert r["significant"]
    assert r["mean_diff"] > 0
    assert r["wins"] == len(a)


def test_pairing_beats_unpaired_on_correlated_data():
    """
    The point of the whole module: a small consistent gain on noisy queries.
    Query difficulty varies hugely (0.0 to 0.9) but B beats A on every one.
    An unpaired comparison would drown in that spread; the paired test sees it.
    """
    import numpy as np
    rng = np.random.default_rng(0)
    difficulty = rng.uniform(0, 0.9, 200)
    a = difficulty
    b = difficulty + 0.02
    r = paired_bootstrap(list(a), list(b))
    assert r["significant"], "paired test must detect a consistent small gain"
    # the unpaired spread is an order of magnitude larger than the effect
    assert a.std() > 10 * 0.02


def test_mismatched_lengths_rejected():
    import pytest
    with pytest.raises(ValueError):
        paired_bootstrap([0.1, 0.2], [0.1])

"""
Comparing two systems properly.

THE MISTAKE EVERYONE MAKES, including my own output an hour ago:
running both systems, printing a confidence interval for each, and
concluding "the intervals overlap, so there is no difference."

That is the WRONG TEST. Those two intervals describe how much each system's
score would wobble across different samples of queries. But both systems were
evaluated on the SAME queries. A query that is hard for BM25 is usually hard
for the hybrid too, so their errors move together. Comparing two independent
intervals throws that pairing away and is badly underpowered - it will call
a real improvement "not significant" all the time.

THE RIGHT TEST is paired: for each query compute the DIFFERENCE between the
two systems, then bootstrap the mean of those differences. The shared
difficulty cancels out. If the interval for the difference excludes zero,
the improvement is established.

Same logic as a paired t-test versus an unpaired one, without assuming the
differences are normally distributed - which, for per-query nDCG bounded in
[0,1] and often exactly 0, they are emphatically not.
"""
from __future__ import annotations

import numpy as np


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 10000,
                     alpha: float = 0.05, seed: int = 0) -> dict:
    """
    Is system B better than system A? Both scored on the SAME queries, in the
    same order.

    Returns the mean difference, its CI, and a two-sided bootstrap p-value.
    """
    a_arr, b_arr = np.asarray(a, float), np.asarray(b, float)
    if a_arr.shape != b_arr.shape:
        raise ValueError("paired comparison needs the same queries in both runs")
    d = b_arr - a_arr
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    # Two-sided p: how often does the resampled mean land on the other side
    # of zero from the observed effect. Doubled for two-sidedness.
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {
        "mean_diff": float(d.mean()),
        "ci_lo": float(lo), "ci_hi": float(hi),
        "p_value": float(min(p, 1.0)),
        "n": int(d.size),
        "wins": int((d > 0).sum()),
        "losses": int((d < 0).sum()),
        "ties": int((d == 0).sum()),
        "significant": bool(lo > 0 or hi < 0),
    }


def describe(name_a: str, name_b: str, r: dict) -> str:
    verdict = ("ESTABLISHED" if r["significant"]
               else "NOT established (interval includes 0)")
    return (f"  {name_b} vs {name_a}\n"
            f"    mean difference : {r['mean_diff']:+.4f}   "
            f"95% CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]\n"
            f"    p (two-sided)   : {r['p_value']:.4f}\n"
            f"    per-query       : {r['wins']} better, {r['losses']} worse, "
            f"{r['ties']} identical  (n={r['n']})\n"
            f"    verdict         : {verdict}")

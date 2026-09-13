#!/usr/bin/env python3
"""
Zero-shot SigLIP 2 vs the trained adapter, measured properly.

    python scripts/compare_runs.py

WHY THE STANDARD TEST SET IS NOT ENOUGH:
the Flickr30k 1k test set scores 5000 captions against 1000 candidate images.
Zero-shot SigLIP 2 already reaches Recall@10 = 0.949 there. With 5% headroom
left, that benchmark cannot resolve a real improvement - not because the
model did not improve, but because the measurement has no room.

So this reports TWO evaluations:

  EASY   5000 test captions vs the 1000 test images.
         The standard, comparable-to-published protocol. Reported for
         comparability, not for sensitivity.

  HARD   the same 5000 test captions vs 31,783 candidates - the 1000 test
         images PLUS all 30,783 training-pool images as distractors.
         Still completely clean: the captions were never trained on, and the
         extra images serve only as distractors. Thirty times more ways to be
         wrong, so the measurement can actually see a difference.

Adding distractors is the standard way to stop a saturated benchmark from
hiding your result. It makes the task harder, not easier, so it cannot
flatter the model.

Both are scored with a PAIRED bootstrap over the 5000 captions, because both
systems ran on the same queries.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _venv_check import require_venv                       # noqa: E402
require_venv()

from contextfuse.adapter import DualAdapter                # noqa: E402
from contextfuse.stats import describe, paired_bootstrap   # noqa: E402

CHUNK = 256          # captions per scoring chunk, to bound memory


def ranks_of(txt: np.ndarray, img: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """Rank of the correct image for each caption. 1 = perfect."""
    out = np.empty(len(txt), dtype=np.int64)
    for s in range(0, len(txt), CHUNK):
        sc = txt[s:s + CHUNK] @ img.T
        g = gold[s:s + CHUNK]
        gold_score = sc[np.arange(len(g)), g]
        # rank = how many candidates score strictly higher, plus one
        out[s:s + CHUNK] = (sc > gold_score[:, None]).sum(1) + 1
    return out


def report(name: str, r_before: np.ndarray, r_after: np.ndarray,
           n_candidates: int) -> dict:
    print(f"\n{'='*66}\n  {name}   ({n_candidates:,} candidates, {len(r_before):,} queries)\n{'='*66}")
    row = {}
    for k in (1, 5, 10):
        b = (r_before <= k).astype(float)
        a = (r_after <= k).astype(float)
        st = paired_bootstrap(list(b), list(a))
        row[f"recall@{k}"] = {"before": float(b.mean()), "after": float(a.mean()),
                             **st}
        mark = "  ESTABLISHED" if st["significant"] else "  not established"
        print(f"  Recall@{k:<3} {b.mean():.4f} -> {a.mean():.4f}   "
              f"{st['mean_diff']:+.4f}  95% CI [{st['ci_lo']:+.4f}, {st['ci_hi']:+.4f}]"
              f"  p={st['p_value']:.4f}{mark}")

    mb, ma = 1.0 / r_before, 1.0 / r_after
    st = paired_bootstrap(list(mb), list(ma))
    row["mrr"] = {"before": float(mb.mean()), "after": float(ma.mean()), **st}
    print(f"\n  MRR       {mb.mean():.4f} -> {ma.mean():.4f}")
    print(describe("zero-shot", "adapter", st))
    print(f"\n  median rank {np.median(r_before):.0f} -> {np.median(r_after):.0f}")
    return row


def main() -> None:
    z = np.load(ROOT / "data" / "flickr30k" / "siglip_frozen.npz")
    test_img, test_txt, gold = z["test_img"], z["test_txt"], z["test_owner"]
    pool_img = z["pool_img"]

    dim = test_img.shape[1]
    model = DualAdapter(dim, bottleneck=256, dropout=0.1)
    model.load_state_dict(torch.load(ROOT / "data" / "adapter_flickr.pt",
                                     map_location="cpu"))
    model.eval()

    with torch.no_grad():
        a_test_img = model.image(torch.tensor(test_img)).numpy()
        a_pool_img = model.image(torch.tensor(pool_img)).numpy()
        a_test_txt = model.text(torch.tensor(test_txt)).numpy()

    results = {}

    # ---- EASY: the standard protocol -----------------------------------
    results["easy_1k"] = report(
        "EASY - standard Flickr30k 1k test protocol",
        ranks_of(test_txt, test_img, gold),
        ranks_of(a_test_txt, a_test_img, gold),
        len(test_img))

    # ---- HARD: same queries, 30x the distractors ------------------------
    # Gold images keep index 0..999, distractors are appended after them, so
    # the gold index is unchanged and no relabelling can go wrong.
    big_before = np.concatenate([test_img, pool_img], axis=0)
    big_after = np.concatenate([a_test_img, a_pool_img], axis=0)
    results["hard_31k"] = report(
        "HARD - same queries, all 30,783 pool images added as distractors",
        ranks_of(test_txt, big_before, gold),
        ranks_of(a_test_txt, big_after, gold),
        len(big_before))

    out = ROOT / "results" / "compare_zeroshot_vs_adapter.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n  saved -> results/{out.name}")

    h = results["hard_31k"]["recall@10"]
    print(f"""
  HOW TO STATE THIS IN THE REPORT

    On the standard 1k protocol, zero-shot SigLIP 2 already reaches
    Recall@10 = {results['easy_1k']['recall@10']['before']:.3f}, leaving little measurable headroom.
    Enlarging the candidate set to 31,783 images - the same queries against
    thirty times more distractors - the adapter improves Recall@10 from
    {h['before']:.3f} to {h['after']:.3f} ({(h['after']-h['before'])/h['before']*100:+.1f}% relative), p = {h['p_value']:.4f}.

  Report BOTH. The easy one is comparable to published work; the hard one
  is what can actually resolve the effect. Hiding either would be a choice
  a reviewer is entitled to ask about.""")


if __name__ == "__main__":
    main()

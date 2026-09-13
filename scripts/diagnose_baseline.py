#!/usr/bin/env python3
"""
Is the low score a bug, or is SigLIP genuinely bad at this?

Never accept a bad number because it matches your story. Four checks, each
distinguishing a different failure. Uses the CACHED embeddings - no GPU work.

    python scripts/diagnose_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _venv_check import require_venv  # noqa: E402
require_venv()

from contextfuse.data import load_vidore_cs                    # noqa: E402
from contextfuse.embed import SigLIPEncoder                    # noqa: E402


def rule(t):
    print(f"\n{'='*70}\n{t}\n{'='*70}")


cache = next((ROOT / "data" / "embeddings").glob("vidore_cs__*.npz"))
z = np.load(cache)
img, ids = z["emb"], list(z["ids"])
print(f"cached embeddings: {img.shape} from {cache.name}")

# ---------------------------------------------------------------- CHECK 1
rule("1  Are the embeddings healthy, or collapsed?")
print(f"  NaN / inf            : {np.isnan(img).any()} / {np.isinf(img).any()}")
norms = np.linalg.norm(img, axis=1)
print(f"  L2 norms             : min={norms.min():.4f} max={norms.max():.4f}  (want 1.0)")
print(f"  per-dim std (mean)   : {img.std(axis=0).mean():.5f}")
print(f"  dims with ~zero std  : {(img.std(axis=0) < 1e-4).sum()} of {img.shape[1]}")

sample = img[np.random.default_rng(0).choice(len(img), 400, replace=False)]
sim = sample @ sample.T
off = sim[~np.eye(len(sample), dtype=bool)]
print(f"\n  mean pairwise cosine BETWEEN PAGES : {off.mean():.4f}")
print(f"  (min {off.min():.3f}, max {off.max():.3f}, std {off.std():.3f})")
print("""
  READ THIS CAREFULLY:
    ~0.0-0.4  pages are well separated. The encoder distinguishes them.
    ~0.6-0.8  pages are crowding together. Weak but usable signal.
    >0.90     EMBEDDING COLLAPSE. At 224x224 every textbook page is just
              'a rectangle of grey text'. The model literally cannot tell
              them apart, so retrieval is near-random by construction.
              That is a FINDING about resolution, not a bug in your code.""")

# ---------------------------------------------------------------- CHECK 2
rule("2  Where do the gold pages actually rank, over all 1360?")
bench = load_vidore_cs("english")
enc = SigLIPEncoder()
qids = sorted(bench.queries)
q_emb = enc.encode_texts([bench.queries[q] for q in qids])

pos = {cid: i for i, cid in enumerate(ids)}
scores = q_emb @ img.T                                  # [Q, 1360]
order = np.argsort(-scores, axis=1)
rank_of = np.empty_like(order)
np.put_along_axis(rank_of, order, np.arange(order.shape[1])[None, :].repeat(len(qids), 0), axis=1)

best, gold_s, other_s = [], [], []
for i, q in enumerate(qids):
    g = [pos[c] for c in bench.qrels[q] if c in pos]
    if not g:
        continue
    best.append(min(rank_of[i, j] for j in g) + 1)
    gold_s.append(scores[i, g].mean())
    mask = np.ones(len(ids), bool); mask[g] = False
    other_s.append(scores[i, mask].mean())

best = np.array(best)
print(f"  rank of FIRST gold page (1 = perfect, 1360 = worst)")
print(f"    median {np.median(best):.0f}   mean {best.mean():.0f}")
print(f"    p25 {np.percentile(best,25):.0f}   p75 {np.percentile(best,75):.0f}")
print(f"  random baseline would be ~{len(ids)/2:.0f}")
print(f"\n  in top 10  : {(best<=10).mean():.1%}")
print(f"  in top 50  : {(best<=50).mean():.1%}")
print(f"  in top 100 : {(best<=100).mean():.1%}")
print("""
  If median rank is near 680, the system is guessing and something is wrong.
  If it is ~50-150, the model ranks gold pages well above chance but not into
  the top 10 - real, weak signal. That is a resolution problem, not a bug.""")

# ---------------------------------------------------------------- CHECK 3
rule("3  Does the model score gold pages above non-gold at all?")
gold_s, other_s = np.array(gold_s), np.array(other_s)
print(f"  mean cosine, query vs GOLD pages     : {gold_s.mean():.4f}")
print(f"  mean cosine, query vs everything else: {other_s.mean():.4f}")
d = gold_s - other_s
print(f"  difference (gold - other)            : {d.mean():+.4f}")
print(f"  queries where gold scores higher     : {(d>0).mean():.1%}")
print("""
  ~50% would mean no signal whatsoever - suspect an ID-alignment bug.
  Comfortably above 50% means the ranking works, just not sharply.""")

# ---------------------------------------------------------------- CHECK 4
rule("4  ID alignment - the silent killer")
print(f"  embedding rows        : {len(ids)}")
print(f"  corpus_ids from data  : {len(bench.corpus_ids)}")
print(f"  identical, in order   : {ids == bench.corpus_ids}")
print(f"  first 5 cached / data : {ids[:5]} / {bench.corpus_ids[:5]}")
print(f"  qrels reference ids   : all present = "
      f"{all(c in pos for q in bench.qrels for c in bench.qrels[q])}")
print("""
  If these disagree, every score you have is meaningless - you would be
  comparing query 7 against page 7's neighbour. This check costs nothing
  and catches the error that is hardest to notice.""")

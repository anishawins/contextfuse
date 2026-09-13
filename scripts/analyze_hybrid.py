#!/usr/bin/env python3
"""
Two questions the last run raised but could not answer.

    python scripts/analyze_hybrid.py

Q1  Is D_weighted actually better than BM25 alone?
    The previous output compared two independent confidence intervals, saw
    them overlap, and said "not established". That was the wrong test - both
    systems ran on the SAME queries, so the comparison must be PAIRED.

Q2  WHERE does the visual signal help, if anywhere?
    92% of ViDoRe's judgements are content_type 'Text'. A method that reads
    text should dominate there. The real question is whether SigLIP earns its
    place on pages that are charts, tables, infographics and images - the
    pages where there is less text to read. If it does, your hybrid has a
    principled reason to exist rather than a hopeful one.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _venv_check import require_venv                        # noqa: E402
require_venv()

from contextfuse.data import load_vidore_cs                 # noqa: E402
from contextfuse.embed import SigLIPEncoder                 # noqa: E402
from contextfuse.fusion import weighted_fusion              # noqa: E402
from contextfuse.lexical import BM25                        # noqa: E402
from contextfuse.metrics import ndcg_at_k, recall_at_k, reciprocal_rank, bootstrap_ci  # noqa: E402
from contextfuse.splits import split_queries                # noqa: E402
from contextfuse.stats import describe, paired_bootstrap    # noqa: E402

ALPHA = json.loads((ROOT / "results" / "hybrid__english.json").read_text()
                   )["protocol"]["best_alpha_bm25"]
CAND = 100
VISUAL = {"Table", "Chart", "Infographic", "Image"}

bench = load_vidore_cs("english")
qids = sorted(bench.queries)
dev, test = split_queries({q: bench.need_of[q] for q in qids})
print(f"alpha (tuned on dev, reused unchanged): {ALPHA}   test n={len(test)}")

z = np.load(next((ROOT / "data" / "embeddings").glob("vidore_cs__*.npz")))
img, ids = z["emb"], list(z["ids"])
q_emb = SigLIPEncoder().encode_texts([bench.queries[q] for q in qids])
dense = q_emb @ img.T
bm25 = BM25(bench.images["markdown"], ids)

runs = {"A_bm25": {}, "B_siglip": {}, "D_weighted": {}}
for i, q in enumerate(qids):
    top = np.argsort(-dense[i])[:CAND]
    d_run = {int(ids[j]): float(dense[i, j]) for j in top}
    l_run = dict(bm25.search(bench.queries[q], k=CAND))
    runs["A_bm25"][q] = [d for d, _ in sorted(l_run.items(), key=lambda kv: -kv[1])[:10]]
    runs["B_siglip"][q] = [d for d, _ in sorted(d_run.items(), key=lambda kv: -kv[1])[:10]]
    runs["D_weighted"][q] = weighted_fusion([l_run, d_run], [ALPHA, 1 - ALPHA])


def per_query(name, subset, fn=lambda r, g: ndcg_at_k(r, g, 10)):
    return [fn(runs[name][q], bench.qrels[q]) for q in subset]


# ---------------------------------------------------------------- Q1
print(f"\n{'='*70}\nQ1  Paired comparison on TEST\n{'='*70}")
for metric, fn in (("nDCG@10", lambda r, g: ndcg_at_k(r, g, 10)),
                   ("Recall@10", lambda r, g: recall_at_k(r, g, 10)),
                   ("MRR@10", lambda r, g: reciprocal_rank(r, g, 10))):
    print(f"\n  --- {metric} ---")
    print(describe("A_bm25", "D_weighted",
                   paired_bootstrap(per_query("A_bm25", test, fn),
                                    per_query("D_weighted", test, fn))))

# ---------------------------------------------------------------- Q2
print(f"\n{'='*70}\nQ2  Does the visual signal help on non-text pages?\n{'='*70}")
qmeta = {r["query_id"]: (r["content_type"] or [])
         for r in load_dataset("vidore/vidore_v3_computer_science", "queries")["test"]}

text_q = [q for q in test if not (set(qmeta.get(q, [])) & VISUAL)]
vis_q = [q for q in test if set(qmeta.get(q, [])) & VISUAL]
print(f"  text-only queries        : {len(text_q)}")
print(f"  queries touching visuals : {len(vis_q)}")
print(f"  visual types present     : "
      f"{dict(Counter(t for q in vis_q for t in qmeta[q] if t in VISUAL))}")

for label, subset in (("TEXT-ONLY", text_q), ("VISUAL", vis_q)):
    if len(subset) < 10:
        print(f"\n  [{label}] only {len(subset)} queries - too few to conclude anything.")
        continue
    print(f"\n  ===== {label}  (n={len(subset)}) =====")
    for name in ("A_bm25", "B_siglip", "D_weighted"):
        m, lo, hi = bootstrap_ci(per_query(name, subset))
        print(f"    {name:<12} nDCG@10 {m:.4f}  [{lo:.4f}, {hi:.4f}]")
    print()
    print(describe("A_bm25", "D_weighted",
                   paired_bootstrap(per_query("A_bm25", subset),
                                    per_query("D_weighted", subset))))

print("""
======================================================================
HOW TO READ THIS

If the paired test says D_weighted beats A_bm25 on VISUAL queries but not
on TEXT-ONLY queries, you have a precise, defensible claim:

  "Dense visual retrieval contributes nothing on text-dominated pages,
   where lexical search is already near-sufficient, but contributes
   measurably where the answer is carried by a chart, table or figure."

That is a far better result than 'hybrid wins'. It says WHEN and WHY, it
predicts what happens on a screenshot corpus, and it is the argument for
keeping the vision encoder in the system at all.

If it helps nowhere, say so. A negative result you measured carefully and
explained is worth more in a viva than a positive one you cannot defend.
======================================================================""")

#!/usr/bin/env python3
"""
Configuration D: BM25 + SigLIP fused. Plus the query-provenance check.

    python scripts/run_hybrid.py

DISCIPLINE: the fusion weight is tuned on DEV queries only, then applied
once to TEST. The TEST number is the one you report. Tuning and reporting on
the same queries is the most common way student projects overstate results.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _venv_check import require_venv                            # noqa: E402
require_venv()

from contextfuse.data import load_vidore_cs                     # noqa: E402
from contextfuse.fusion import rrf, weighted_fusion             # noqa: E402
from contextfuse.lexical import BM25                            # noqa: E402
from contextfuse.metrics import evaluate, format_table, ndcg_at_k  # noqa: E402
from contextfuse.splits import split_queries                    # noqa: E402

CAND = 100          # candidates pulled from each retriever before fusion

bench = load_vidore_cs("english")
qids = sorted(bench.queries)
dev, test = split_queries({q: bench.need_of[q] for q in qids})
print(f"{bench}\n  dev={len(dev)}  test={len(test)}  (split by information need)")

# ---- the two runs -------------------------------------------------------
z = np.load(next((ROOT / "data" / "embeddings").glob("vidore_cs__*.npz")))
img, ids = z["emb"], list(z["ids"])

from contextfuse.embed import SigLIPEncoder                      # noqa: E402
q_emb = SigLIPEncoder().encode_texts([bench.queries[q] for q in qids])
dense_scores = q_emb @ img.T

bm25 = BM25(bench.images["markdown"], ids)

dense_run, lex_run = {}, {}
for i, q in enumerate(qids):
    top = np.argsort(-dense_scores[i])[:CAND]
    dense_run[q] = {int(ids[j]): float(dense_scores[i, j]) for j in top}
    lex_run[q] = dict(bm25.search(bench.queries[q], k=CAND))

# ---- tune alpha on DEV ONLY --------------------------------------------
print("\n  tuning fusion weight on DEV ...")
best_a, best_s = None, -1.0
history = []
for a in np.round(np.arange(0.0, 1.01, 0.05), 2):
    s = np.mean([
        ndcg_at_k(weighted_fusion([lex_run[q], dense_run[q]], [a, 1 - a]),
                  bench.qrels[q], 10)
        for q in dev])
    history.append({"alpha_bm25": float(a), "dev_ndcg@10": float(s)})
    if s > best_s:
        best_a, best_s = float(a), float(s)
print(f"  best alpha (BM25 weight) = {best_a}   dev nDCG@10 = {best_s:.4f}")
print("  alpha=1.0 means BM25 only; alpha=0.0 means SigLIP only.")

# ---- evaluate everything on TEST ---------------------------------------
runs = {
    "A_bm25_only":   {q: [d for d, _ in sorted(lex_run[q].items(), key=lambda kv: -kv[1])[:10]] for q in test},
    "B_siglip_only": {q: [d for d, _ in sorted(dense_run[q].items(), key=lambda kv: -kv[1])[:10]] for q in test},
    "D_weighted":    {q: weighted_fusion([lex_run[q], dense_run[q]], [best_a, 1 - best_a]) for q in test},
    "D_rrf":         {q: rrf([[d for d, _ in sorted(lex_run[q].items(), key=lambda kv: -kv[1])],
                              [d for d, _ in sorted(dense_run[q].items(), key=lambda kv: -kv[1])]]) for q in test},
}

gold = {q: bench.qrels[q] for q in test}
out = {}
for name, r in runs.items():
    res = evaluate(r, gold)
    out[name] = res
    print()
    print(format_table(res, f"{name}   (TEST, n={len(test)})"))

print("\n" + "=" * 62)
print(f"  {'config':<16}{'nDCG@10':>22}{'Recall@10':>22}")
for name in runs:
    n, r = out[name]["ndcg@10"], out[name]["recall@10"]
    print(f"  {name:<16}{n['mean']:>9.4f} [{n['lo']:.3f},{n['hi']:.3f}]"
          f"{r['mean']:>9.4f} [{r['lo']:.3f},{r['hi']:.3f}]")
print("\n  Read the intervals, not the means. If D_weighted's interval")
print("  overlaps A_bm25_only's, adding SigLIP did NOT demonstrably help.")
print("  That is a legitimate result. Report it as one.")

# ---- who wrote the query? ----------------------------------------------
print("\n" + "=" * 62)
print("  QUERY PROVENANCE - is BM25 inflated by leakage?")
print("=" * 62)
qmeta = {r["query_id"]: r["query_generator"]
         for r in __import__("datasets").load_dataset(
             "vidore/vidore_v3_computer_science", "queries")["test"]}
for gen in ("human", "sdg"):
    sub = [q for q in test if qmeta.get(q) == gen]
    if not sub:
        continue
    for name in ("A_bm25_only", "B_siglip_only"):
        res = evaluate({q: runs[name][q] for q in sub}, {q: gold[q] for q in sub})
        n = res["ndcg@10"]
        print(f"  {gen:<6} {name:<16} n={len(sub):<4} "
              f"nDCG@10 {n['mean']:.4f} [{n['lo']:.3f},{n['hi']:.3f}]")
print("""
  'sdg' queries were machine-generated. If they were written by an LLM
  reading each page's markdown, then BM25 searching that same markdown is
  searching the text the question came from. A large human-vs-sdg gap for
  BM25 is evidence of that. It does not invalidate the benchmark - it is a
  limitation you disclose, and it makes the human-only number the honest
  headline.""")

(ROOT / "results" / "hybrid__english.json").write_text(json.dumps({
    "tag": "hybrid",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "protocol": {"candidates_per_retriever": CAND,
                 "alpha_tuned_on": "dev", "reported_on": "test",
                 "dev_n": len(dev), "test_n": len(test),
                 "best_alpha_bm25": best_a, "dev_sweep": history},
    "metrics": out,
}, indent=2))
print(f"\n  saved -> results/hybrid__english.json")

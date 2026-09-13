#!/usr/bin/env python3
"""
Configuration A: BM25 over the markdown text ViDoRe ships with each page.

    python scripts/run_bm25.py

TWO JOBS AT ONCE.

1. It is configuration A of your experimental matrix.

2. It is a CONTROL for the low SigLIP score. Same qrels, same metrics, same
   harness - completely different retrieval mechanism. If BM25 scores well
   here, the evaluation code is correct and SigLIP is genuinely weak on this
   data. If BM25 is ALSO near zero, the bug is in the harness, not the model.
   Changing one variable at a time is the whole of experimental method.

NOTE ON WHAT THIS IS NOT: ViDoRe's `markdown` is publisher-quality extracted
text, not OCR output. So this is a PERFECT-OCR UPPER BOUND - the score your
real OCR pipeline could reach if it made zero mistakes. The gap between this
and BM25-over-your-own-OCR is exactly what OCR error costs you. Do not report
this as "my OCR pipeline's score". It is not.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _venv_check import require_venv  # noqa: E402
require_venv()

from contextfuse.data import load_vidore_cs                  # noqa: E402
from contextfuse.lexical import BM25                         # noqa: E402
from contextfuse.metrics import evaluate, format_table       # noqa: E402

bench = load_vidore_cs("english")
print(bench)

corpus = bench.images
texts = corpus["markdown"]
ids = bench.corpus_ids
empty = sum(1 for t in texts if not t or not t.strip())
print(f"  pages with empty markdown: {empty} of {len(texts)}")

t0 = time.perf_counter()
bm25 = BM25(texts, ids)
t_index = time.perf_counter() - t0
print(f"  index built in {t_index:.2f}s -> {bm25.stats()}")

qids = sorted(bench.queries)
rankings, lat = {}, []
for q in qids:
    t0 = time.perf_counter()
    hits = bm25.search(bench.queries[q], k=10)
    lat.append((time.perf_counter() - t0) * 1000)
    rankings[q] = [cid for cid, _ in hits]

res = evaluate(rankings, {q: bench.qrels[q] for q in qids})
print()
print(format_table(res, f"BM25 over provided markdown (perfect-OCR bound) | n={len(qids)}"))
p50, p95 = float(np.percentile(lat, 50)), float(np.percentile(lat, 95))
print(f"\n  search latency  p50={p50:.2f} ms  p95={p95:.2f} ms")

out = ROOT / "results" / "bm25_markdown__english.json"
out.write_text(json.dumps({
    "tag": "bm25_markdown",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "config": {"k1": bm25.k1, "b": bm25.b, "text_source": "provided markdown",
               "caveat": "perfect-OCR upper bound, NOT an OCR pipeline result"},
    "dataset": {"pages": len(ids), "queries": len(qids)},
    "metrics": res,
    "latency_ms": {"search_p50": p50, "search_p95": p95, "index_build_s": t_index},
}, indent=2))
print(f"  saved -> results/{out.name}")

sig = ROOT / "results" / "siglip2_only__english.json"
if sig.exists():
    s = json.loads(sig.read_text())["metrics"]
    print("\n  --- side by side (english, n=215) ---")
    print(f"  {'metric':<12} {'BM25':>18}   {'SigLIP-2':>18}")
    for m in ("recall@10", "mrr@10", "ndcg@10"):
        print(f"  {m:<12} {res[m]['mean']:>10.4f} [{res[m]['lo']:.3f},{res[m]['hi']:.3f}]"
              f"   {s[m]['mean']:>10.4f} [{s[m]['lo']:.3f},{s[m]['hi']:.3f}]")
    print("\n  Overlapping confidence intervals mean the difference is NOT")
    print("  established. Non-overlapping is suggestive, not proof.")

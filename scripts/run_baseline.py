#!/usr/bin/env python3
"""
ContextFuse baseline: SigLIP 2 dense retrieval on ViDoRe V3 Computer Science.

    source .venv/bin/activate
    python scripts/run_baseline.py                # english, 215 queries
    python scripts/run_baseline.py --limit 40     # fast smoke test first
    python scripts/run_baseline.py --language all # multilingual, secondary

Image embeddings are cached to data/embeddings/, so the second run is fast.

This is configuration B in the experimental matrix (SigLIP only). It is the
number every later configuration has to beat. Nothing is fabricated here -
whatever it prints is what your machine measured.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _venv_check import require_venv  # noqa: E402
require_venv()

from contextfuse.data import load_vidore_cs                      # noqa: E402
from contextfuse.embed import DEFAULT_CKPT, SigLIPEncoder        # noqa: E402
from contextfuse.metrics import evaluate, format_table           # noqa: E402
from contextfuse.search import ExactSearch                       # noqa: E402


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                             # noqa: BLE001
        return "not-a-git-repo"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", default="english",
                    help="'english' (default), a language name, or 'all'")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N queries - for smoke tests")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="siglip2_only")
    args = ap.parse_args()

    # Reproducibility. Inference should be deterministic anyway; we fix the
    # seeds so the BOOTSTRAP resampling is identical between runs, otherwise
    # the confidence intervals wobble and you cannot compare two runs.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    lang = None if args.language == "all" else args.language
    print(f"loading benchmark (language={args.language}) ...")
    bench = load_vidore_cs(language=lang)
    print(f"  {bench}")

    qids = sorted(bench.queries)
    if args.limit:
        qids = qids[:args.limit]
    texts = [bench.queries[q] for q in qids]
    n_needs = len({bench.need_of[q] for q in qids})
    print(f"  evaluating {len(qids)} queries spanning {n_needs} information needs")

    enc = SigLIPEncoder(args.ckpt)
    print(f"  model={args.ckpt}  device={enc.device}  dim={enc.dim}")

    # ---- image embeddings, cached -------------------------------------
    cache = ROOT / "data" / "embeddings" / f"vidore_cs__{args.ckpt.replace('/', '_')}.npz"
    if cache.exists():
        z = np.load(cache)
        img_emb, corpus_ids = z["emb"], list(z["ids"])
        print(f"  loaded cached image embeddings {img_emb.shape} from {cache.name}")
        t_index = 0.0
    else:
        print(f"  encoding {len(bench.corpus_ids)} page images ...")
        t0 = time.perf_counter()
        img_emb = enc.encode_images(bench.images, batch_size=args.batch_size)
        t_index = time.perf_counter() - t0
        corpus_ids = bench.corpus_ids
        np.savez_compressed(cache, emb=img_emb, ids=np.asarray(corpus_ids))
        print(f"  encoded in {t_index:.1f}s "
              f"({len(corpus_ids)/t_index:.1f} pages/s) -> {cache.name}")

    # ---- query embeddings ---------------------------------------------
    t0 = time.perf_counter()
    q_emb = enc.encode_texts(texts)
    t_qenc = time.perf_counter() - t0

    # ---- search ---------------------------------------------------------
    index = ExactSearch(img_emb, corpus_ids)
    lat = []
    rankings = {}
    for qid, vec in zip(qids, q_emb):
        t0 = time.perf_counter()
        ids, _ = index.search(vec[None, :], k=10)
        lat.append((time.perf_counter() - t0) * 1000)
        rankings[qid] = ids[0].tolist()

    # ---- score ------------------------------------------------------------
    results = evaluate(rankings, {q: bench.qrels[q] for q in qids}, seed=args.seed)

    p50, p95 = float(np.percentile(lat, 50)), float(np.percentile(lat, 95))
    print()
    print(format_table(results, f"SigLIP-2 dense only  |  {args.language}  |  n={len(qids)}"))
    print(f"\n  search latency   p50={p50:.3f} ms   p95={p95:.3f} ms  (exact, N={len(corpus_ids)})")
    print(f"  query encode     {t_qenc/len(qids)*1000:.1f} ms/query")

    record = {
        "tag": args.tag,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "config": {
            "model": args.ckpt, "device": enc.device, "embedding_dim": enc.dim,
            "language": args.language, "seed": args.seed,
            "batch_size": args.batch_size, "retrieval": "exact cosine",
        },
        "dataset": {
            "repo": "vidore/vidore_v3_computer_science",
            "pages": len(corpus_ids), "queries": len(qids),
            "information_needs": n_needs,
        },
        "hardware": {
            "platform": platform.platform(), "machine": platform.machine(),
            "torch": torch.__version__,
        },
        "metrics": results,
        "latency_ms": {"search_p50": p50, "search_p95": p95,
                       "query_encode_mean": t_qenc / len(qids) * 1000,
                       "index_build_s": t_index},
    }
    out = ROOT / "results" / f"{args.tag}__{args.language}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"\n  saved -> results/{out.name}")

    print("\n  --- three example queries, top 3 each ---")
    for qid in qids[:3]:
        gold = set(bench.qrels[qid])
        print(f"\n   Q{qid}: {bench.queries[qid][:90]}")
        for rank, cid in enumerate(rankings[qid][:3], 1):
            print(f"      {rank}. page {cid}  {'<-- RELEVANT' if cid in gold else ''}")
        print(f"      (gold pages: {sorted(gold)})")


if __name__ == "__main__":
    main()

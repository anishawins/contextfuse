#!/usr/bin/env python3
"""
Search your own images.

    python scripts/search.py "postgres connection refused"
    python scripts/search.py "the slide about CNNs" -k 5
    python scripts/search.py "cuda error" --open

Every result comes with WHY it was returned. A search engine that cannot
explain itself is one you cannot debug and cannot defend.
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _venv_check import require_venv                        # noqa: E402
require_venv()

from contextfuse.adaptive import decide_alpha               # noqa: E402
from contextfuse.embed import SigLIPEncoder                 # noqa: E402
from contextfuse.fusion import minmax                       # noqa: E402
from contextfuse.ingest import hamming                      # noqa: E402
from contextfuse.lexical import tokenize                    # noqa: E402

DATA = ROOT / "data" / "personal"

# STARTING POINT, NOT A TUNED VALUE. On ViDoRe the best BM25 weight was 0.9,
# but that corpus was 1360 near-identical textbook pages where the visual
# signal was nearly useless. Screenshots are visually distinct, so the dense
# side should carry more weight here. 0.6 is a considered guess. Once you
# have labelled queries of your own, tune it properly on a dev split.
ALPHA_LEXICAL = 0.6


def snippet(text: str, terms: set[str], width: int = 100) -> str:
    """Return the line of OCR text that best matches the query."""
    best, score = "", -1
    for line in (text or "").splitlines():
        hits = sum(1 for t in tokenize(line) if t in terms)
        if hits > score:
            best, score = line, hits
    return (best[:width] + "...") if len(best) > width else best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=-1.0,
                    help="lexical weight 0..1; -1 (default) picks it per query")
    ap.add_argument("--open", action="store_true", help="open the top hit")
    ap.add_argument("--no-dedup", action="store_true")
    a = ap.parse_args()
    query = " ".join(a.query)

    if not (DATA / "records.json").exists():
        sys.exit("!!! no index. Run: python scripts/build_personal_index.py <folder>")

    records = json.loads((DATA / "records.json").read_text())
    emb = np.load(DATA / "embeddings.npy")
    with (DATA / "bm25.pkl").open("rb") as fh:
        bm25 = pickle.load(fh)
    by_id = {r["id"]: r for r in records}

    t0 = time.perf_counter()
    enc = SigLIPEncoder()
    q_vec = enc.encode_texts([query])[0]
    dense = {int(r["id"]): float(s) for r, s in zip(records, emb @ q_vec)}
    lex = dict(bm25.search(query, k=len(records)))
    t_search = (time.perf_counter() - t0) * 1000

    dn, ln = minmax(dense), minmax(lex)
    if a.alpha < 0:
        d = decide_alpha(query, bm25)
        alpha, why = d.alpha, d.explanation
    else:
        alpha, why = a.alpha, "fixed weight set by hand"
    fused = {i: alpha * ln.get(i, 0.0) + (1 - alpha) * dn.get(i, 0.0)
             for i in dense}

    order = sorted(fused, key=lambda i: -fused[i])
    terms = set(tokenize(query))

    shown, seen_ph = [], []
    for i in order:
        r = by_id[i]
        if not a.no_dedup and any(hamming(r["phash"], p) <= 6 for p in seen_ph):
            continue                                  # near-duplicate already shown
        seen_ph.append(r["phash"])
        shown.append(i)
        if len(shown) >= a.k:
            break

    print(f'\n  "{query}"     {len(records)} images     {t_search:.0f} ms')
    print(f"  OCR weight {alpha:.2f}  ->  {why}\n")
    for rank, i in enumerate(shown, 1):
        r = by_id[i]
        matched = sorted(terms & set(tokenize(r["ocr_text"])))
        print(f"  {rank}. {r['filename']}")
        print(f"     score {fused[i]:.3f}   "
              f"visual {dn.get(i,0):.2f}   text {ln.get(i,0):.2f}")
        why = []
        if dn.get(i, 0) > 0.6:
            why.append("strong visual similarity")
        if matched:
            why.append(f"OCR match: {', '.join(repr(m) for m in matched[:4])}")
        if r["source"] != "unknown":
            why.append(f"source: {r['source']}")
        why.append(r["mtime_iso"][:10])
        if r["ocr_confidence"] and r["ocr_confidence"] < 0.5:
            why.append(f"LOW OCR confidence {r['ocr_confidence']:.2f}")
        print(f"     why: " + "  |  ".join(why))
        s = snippet(r["ocr_text"], terms)
        if s:
            print(f"     text: {s}")
        print()

    if a.open and shown:
        subprocess.run(["open", by_id[shown[0]]["path"]], check=False)


if __name__ == "__main__":
    main()

"""
FastAPI server for ContextFuse.

Loads the model and index ONCE at startup and keeps them in memory. That is
the entire reason the CLI felt slow: `search.py` paid ~8 seconds of model
loading per query, while the search itself takes under a millisecond.
Knowing which part of a latency number is startup and which is real work is
the difference between optimising the right thing and the wrong thing.

    uvicorn contextfuse.api:app --reload --port 8000
"""
from __future__ import annotations

import io
import json
import pickle
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from PIL import Image
from pydantic import BaseModel, Field

from contextfuse.adaptive import decide_alpha
from contextfuse.embed import SigLIPEncoder
from contextfuse.fusion import minmax
from contextfuse.ingest import hamming
from contextfuse.lexical import tokenize

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "personal"


class Hit(BaseModel):
    """Response schema. Pydantic validates and documents it for free -
    /docs gives you an interactive API reference generated from this."""
    rank: int
    id: int
    filename: str
    score: float = Field(description="fused score, 0..1")
    visual: float = Field(description="normalised SigLIP cosine")
    text: float = Field(description="normalised BM25")
    matched_terms: list[str]
    reasons: list[str]
    snippet: str
    source: str
    date: str
    ocr_confidence: float
    width: int
    height: int


class SearchResponse(BaseModel):
    query: str
    total_indexed: int
    returned: int
    took_ms: float
    alpha: float
    alpha_auto: bool
    alpha_reason: str
    hits: list[Hit]


class Index:
    """Everything loaded once, held in memory."""

    def __init__(self) -> None:
        if not (DATA / "records.json").exists():
            raise RuntimeError(
                "No index found. Run: python scripts/build_personal_index.py <folder>")
        self.records = json.loads((DATA / "records.json").read_text())
        self.emb = np.load(DATA / "embeddings.npy")
        with (DATA / "bm25.pkl").open("rb") as fh:
            self.bm25 = pickle.load(fh)
        self.by_id = {r["id"]: r for r in self.records}
        self.encoder = SigLIPEncoder()

        # PATH TRAVERSAL DEFENCE. The /image endpoint takes an id, not a path,
        # but we still resolve every indexed file once and keep the allowed
        # set. A request can only ever reach a file that was actually indexed.
        # Never build a filesystem path from user input.
        self.allowed: dict[int, Path] = {
            r["id"]: Path(r["path"]).resolve() for r in self.records
        }
        man = json.loads((DATA / "manifest.json").read_text())
        self.root = Path(man["folder"]).resolve()

    def search(self, q: str, k: int, alpha: float, dedup: bool
               ) -> tuple[list[dict], float, float, str]:
        t0 = time.perf_counter()
        auto = alpha < 0
        reason = "fixed weight set by hand"
        if auto:
            d = decide_alpha(q, self.bm25)
            alpha, reason = d.alpha, d.explanation
        vec = self.encoder.encode_texts([q])[0]
        dense = {int(r["id"]): float(s) for r, s in zip(self.records, self.emb @ vec)}
        lex = dict(self.bm25.search(q, k=len(self.records)))
        dn, ln = minmax(dense), minmax(lex)
        fused = {i: alpha * ln.get(i, 0.0) + (1 - alpha) * dn.get(i, 0.0) for i in dense}

        terms = set(tokenize(q))
        out, seen = [], []
        for i in sorted(fused, key=lambda x: -fused[x]):
            r = self.by_id[i]
            if dedup and any(hamming(r["phash"], p) <= 6 for p in seen):
                continue
            seen.append(r["phash"])
            matched = sorted(terms & set(tokenize(r["ocr_text"] or "")))
            reasons = []
            if ln.get(i, 0) > 0.3 and matched:
                reasons.append(f"exact text match: {', '.join(matched[:3])}")
            if dn.get(i, 0) > 0.75:
                reasons.append("strong visual similarity")
            elif dn.get(i, 0) > 0.5:
                reasons.append("moderate visual similarity")
            if r["source"] != "unknown":
                reasons.append(f"source: {r['source']}")
            if r["ocr_confidence"] and r["ocr_confidence"] < 0.5:
                reasons.append(f"low OCR confidence ({r['ocr_confidence']:.2f})")
            out.append({
                "rank": len(out) + 1, "id": i, "filename": r["filename"],
                "score": round(fused[i], 4),
                "visual": round(dn.get(i, 0.0), 4), "text": round(ln.get(i, 0.0), 4),
                "matched_terms": matched[:8], "reasons": reasons,
                "snippet": self._snippet(r["ocr_text"], terms),
                "source": r["source"], "date": r["mtime_iso"][:10],
                "ocr_confidence": round(r["ocr_confidence"], 3),
                "width": r["width"], "height": r["height"],
            })
            if len(out) >= k:
                break
        return out, (time.perf_counter() - t0) * 1000, alpha, reason

    @staticmethod
    def _snippet(text: str, terms: set[str], width: int = 160) -> str:
        best, score = "", -1
        for line in (text or "").splitlines():
            hits = sum(1 for t in tokenize(line) if t in terms)
            if hits > score:
                best, score = line.strip(), hits
        return best[:width] + ("..." if len(best) > width else "")


app = FastAPI(title="ContextFuse", version="0.1.0",
              description="Local-first multimodal search over your own images.")


@lru_cache(maxsize=1)
def get_index() -> Index:
    return Index()


@app.get("/api/search", response_model=SearchResponse)
def api_search(
    q: str = Query(min_length=1, max_length=300),
    k: int = Query(12, ge=1, le=50),
    alpha: float = Query(-1.0, ge=-1.0, le=1.0,
                         description="weight on lexical/OCR score; "
                                     "-1 (default) chooses it per query"),
    dedup: bool = Query(True),
) -> SearchResponse:
    idx = get_index()
    requested_auto = alpha < 0
    hits, ms, used_alpha, reason = idx.search(q, k, alpha, dedup)
    return SearchResponse(query=q, total_indexed=len(idx.records),
                          returned=len(hits), took_ms=round(ms, 2),
                          alpha=round(used_alpha, 3), alpha_auto=requested_auto,
                          alpha_reason=reason, hits=[Hit(**h) for h in hits])


@app.get("/api/stats")
def api_stats() -> dict:
    idx = get_index()
    texts = [r for r in idx.records if (r["ocr_text"] or "").strip()]
    return {
        "images": len(idx.records),
        "with_text": len(texts),
        "without_text": len(idx.records) - len(texts),
        "near_duplicates": sum(1 for r in idx.records if r["duplicate_of"] is not None),
        "vocabulary": idx.bm25.stats()["vocabulary"],
        "embedding_dim": int(idx.emb.shape[1]),
        "folder": str(idx.root),
    }


@app.get("/image/{image_id}")
def image(image_id: int, w: int = Query(520, ge=64, le=2000)) -> Response:
    """
    Serve an indexed image, resized.

    The endpoint takes an ID, never a path. `allowed` maps id -> resolved
    path for files that were actually indexed, so there is no way to
    construct a request that reads an arbitrary file. This is the whole
    defence against path traversal: do not accept paths from clients.
    """
    idx = get_index()
    path = idx.allowed.get(image_id)
    if path is None or not path.is_file():
        raise HTTPException(404, "no such image")
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.width > w:
            im = im.resize((w, int(im.height * w / im.width)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=82)
    return Response(buf.getvalue(), media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return (Path(__file__).parent / "web" / "index.html").read_text()

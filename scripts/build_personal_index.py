#!/usr/bin/env python3
"""
Build a searchable index over YOUR images.

    python scripts/build_personal_index.py ~/Desktop/my_screenshots

Everything stays on this machine. Nothing is uploaded. The index lands in
data/personal/ which .gitignore already blocks.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _venv_check import require_venv                     # noqa: E402
require_venv()

from contextfuse.embed import SigLIPEncoder              # noqa: E402
from contextfuse.ingest import scan                      # noqa: E402
from contextfuse.lexical import BM25                     # noqa: E402
from contextfuse.ocr import get_engine                    # noqa: E402

OUT = ROOT / "data" / "personal"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--include-private", action="store_true",
                    help="index files whose names look sensitive (default: skip)")
    ap.add_argument("--ocr", default="auto", choices=["auto", "apple", "tesseract"])
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()

    folder = a.folder.expanduser().resolve()
    if not folder.is_dir():
        sys.exit(f"!!! not a folder: {folder}")
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- 1. scan --------------------------------------------------------
    print(f"scanning {folder} ...")
    records, skipped = scan(folder, skip_private=not a.include_private)
    if not records:
        n_files = sum(1 for p in folder.rglob("*") if p.is_file())
        msg = [f"!!! indexed 0 images from {folder}",
               f"    the folder contains {n_files} files in total"]
        if skipped:
            msg.append(f"    {len(skipped)} were SKIPPED:")
            msg += [f"      - {s}" for s in skipped[:15]]
            if len(skipped) > 15:
                msg.append(f"      ... and {len(skipped)-15} more")
            msg.append("")
            msg.append("    If these were skipped as 'private-looking', re-run with")
            msg.append("    --include-private to index them anyway.")
        sys.exit("\n".join(msg))
    dupes = sum(1 for r in records if r.duplicate_of is not None)
    print(f"  images indexed   : {len(records)}")
    print(f"  near-duplicates  : {dupes}  (kept, but demoted at search time)")
    print(f"  skipped          : {len(skipped)}")
    for s in skipped[:8]:
        print(f"      - {s}")
    if len(skipped) > 8:
        print(f"      ... and {len(skipped)-8} more")

    from collections import Counter
    print(f"  sources          : {dict(Counter(r.source for r in records))}")

    # ---- 2. OCR ---------------------------------------------------------
    engine = get_engine(a.ocr)
    print(f"\nOCR engine: {engine.name}")
    t0 = time.perf_counter()
    empty = 0
    for r in tqdm(records, desc="ocr", unit="img"):
        try:
            res = engine.read(Path(r.path))
            r.ocr_text, r.ocr_confidence, r.ocr_engine = (
                res.text, res.mean_confidence, res.engine)
            if not res.text.strip():
                empty += 1
        except Exception as exc:                             # noqa: BLE001
            r.ocr_engine = f"failed:{type(exc).__name__}"
            empty += 1
    t_ocr = time.perf_counter() - t0
    chars = [len(r.ocr_text) for r in records]
    print(f"  {t_ocr:.1f}s  ({len(records)/t_ocr:.1f} img/s)")
    print(f"  images with no text : {empty}")
    print(f"  chars per image     : median {int(np.median(chars))}  max {max(chars)}")
    print(f"  mean OCR confidence : "
          f"{np.mean([r.ocr_confidence for r in records if r.ocr_text]):.3f}")

    # ---- 3. embed -------------------------------------------------------
    print("\nembedding images with SigLIP 2 ...")
    enc = SigLIPEncoder()
    t0 = time.perf_counter()
    imgs = [Image.open(r.path) for r in records]
    emb = enc.encode_images(imgs, batch_size=a.batch_size)
    for im in imgs:
        im.close()
    print(f"  {time.perf_counter()-t0:.1f}s  shape={emb.shape}  device={enc.device}")

    # ---- 4. save --------------------------------------------------------
    bm25 = BM25([r.ocr_text for r in records], [r.id for r in records])
    print(f"  BM25 index: {bm25.stats()}")

    np.save(OUT / "embeddings.npy", emb)
    (OUT / "records.json").write_text(
        json.dumps([r.to_dict() for r in records], indent=2))
    with (OUT / "bm25.pkl").open("wb") as fh:
        pickle.dump(bm25, fh)
    (OUT / "manifest.json").write_text(json.dumps({
        "folder": str(folder), "images": len(records),
        "near_duplicates": dupes, "skipped": len(skipped),
        "ocr_engine": engine.name, "model": enc.ckpt,
        "embedding_dim": int(emb.shape[1]),
    }, indent=2))

    print(f"\nindex written to data/personal/  ({len(records)} images)")
    print(f"\n  try it:\n    python scripts/search.py \"your query here\"")


if __name__ == "__main__":
    main()

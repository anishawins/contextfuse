#!/usr/bin/env python3
"""
Stage 0: assemble Flickr30k and precompute FROZEN SigLIP embeddings.

    python scripts/prepare_flickr30k.py --limit 1500   # smoke test
    python scripts/prepare_flickr30k.py                # full

TWO SOURCES, ON PURPOSE:
  TRAIN POOL  lmms-lab/flickr30k                        ~31.8k images
  TEST SET    nlphuji/flickr_1k_test_image_text_retrieval  1k images

The original nlphuji/flickr30k ships a loading script, which `datasets` 5.x
removed support for. These two are parquet-native.

THE LEAKAGE STEP, and it is the whole reason this script is shaped like this:
the 1k test images are a SUBSET of the 31.8k pool. Train on the pool as-is
and you would train on your own test set. Your Recall@1 would look excellent
and mean nothing.

So we match on filename and DELETE every test image from the training pool
before anything is encoded. The script prints how many it removed. If that
number is not close to 1000, something is wrong with the matching and you
should stop rather than proceed - which is why it refuses to continue when
the overlap looks implausible.

WHY PRECOMPUTE THE EMBEDDINGS:
encoding ~31k images takes ~12 minutes on your GPU. Inside a training loop
that is 12 minutes PER EPOCH. Encode once, cache the vectors, and Stage 1
trains in seconds per epoch - so you can run twenty experiments instead of
three. The cost is that you can only train what sits on top of the frozen
encoder; changing the encoder itself is Stage 2 (LoRA), which necessarily
runs it in the loop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _venv_check import require_venv                      # noqa: E402
require_venv()

from datasets import load_dataset                          # noqa: E402
from contextfuse.hfcompat import load_any                  # noqa: E402
from contextfuse.embed import SigLIPEncoder                # noqa: E402

OUT = ROOT / "data" / "flickr30k"
POOL_REPO = "lmms-lab/flickr30k"
TEST_REPO = "nlphuji/flickr_1k_test_image_text_retrieval"


def die(msg: str) -> None:
    sys.exit(f"\n!!! {msg}\n")


def load_one(repo: str):
    print(f"\nloading {repo} ...")
    ds, how = load_any(repo)
    print(f"  route={how}  rows={len(ds)}")
    print(f"  columns: {ds.column_names}")
    return ds


def flatten_captions(caps) -> list[str]:
    """Captions arrive as list[str] per image, occasionally as a bare str."""
    if isinstance(caps, str):
        return [caps]
    return [c for c in caps if isinstance(c, str) and c.strip()]


def encode_split(enc, ds, name: str, batch_size: int):
    print(f"\nencoding {len(ds)} {name} images ...")
    t0 = time.perf_counter()
    img = enc.encode_images(ds, batch_size=batch_size, column="image")
    dt = time.perf_counter() - t0
    print(f"  {dt/60:.1f} min ({len(ds)/dt:.1f} img/s)  shape={img.shape}")

    flat, owner = [], []
    for i, caps in enumerate(ds["caption"]):
        for c in flatten_captions(caps):
            flat.append(c)
            owner.append(i)
    print(f"  encoding {len(flat)} captions ...")
    txt = enc.encode_texts(flat, batch_size=256)
    return img, txt, np.asarray(owner, np.int32), flat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)

    test = load_one(TEST_REPO)
    pool = load_one(POOL_REPO)
    for need, ds, who in (("caption", test, TEST_REPO), ("caption", pool, POOL_REPO)):
        if need not in ds.column_names:
            die(f"{who} has no {need!r} column - columns are {ds.column_names}")

    # ---- the leakage removal -------------------------------------------
    key = "filename" if "filename" in test.column_names else "img_id"
    if key not in pool.column_names:
        die(f"cannot match the two datasets: {key!r} missing from {POOL_REPO}")
    test_keys = set(test[key])
    pool_keys = pool[key]
    keep = [i for i, k in enumerate(pool_keys) if k not in test_keys]
    removed = len(pool) - len(keep)

    print(f"\n--- leakage check ---")
    print(f"  matched on column        : {key!r}")
    print(f"  test images              : {len(test_keys)}")
    print(f"  pool images before       : {len(pool)}")
    print(f"  REMOVED (test ∩ pool)    : {removed}")
    print(f"  training pool after      : {len(keep)}")
    if removed < 0.5 * len(test_keys):
        die(f"only {removed} overlaps found but the test set has {len(test_keys)} "
            f"images. The filename formats probably differ between the two "
            f"datasets, so the de-duplication did NOT work. Fix the matching "
            f"before training - otherwise you will train on your test set.")

    pool = pool.select(keep)
    if a.limit:
        pool = pool.select(range(min(a.limit, len(pool))))
        print(f"\n  LIMITED to {len(pool)} training images - smoke test, not a result")

    # ---- train / val, by a stable hash ----------------------------------
    def is_val(k: str) -> bool:
        return int(hashlib.sha1(str(k).encode()).hexdigest()[:8], 16) % 1000 \
            < a.val_fraction * 1000

    val_flags = np.array([is_val(k) for k in pool[key]])
    print(f"  train={int((~val_flags).sum())}  val={int(val_flags.sum())}  "
          f"test={len(test)}")

    print(f"\n  example captions:")
    for c in flatten_captions(pool["caption"][0])[:3]:
        print(f"    - {c}")

    enc = SigLIPEncoder()
    print(f"\n  encoder {enc.ckpt}  device={enc.device}  dim={enc.dim}")

    p_img, p_txt, p_owner, p_caps = encode_split(enc, pool, "train/val", a.batch_size)
    t_img, t_txt, t_owner, t_caps = encode_split(enc, test, "test", a.batch_size)

    np.savez_compressed(
        OUT / "siglip_frozen.npz",
        pool_img=p_img, pool_txt=p_txt, pool_owner=p_owner,
        pool_is_val=val_flags,
        test_img=t_img, test_txt=t_txt, test_owner=t_owner,
    )
    (OUT / "captions.json").write_text(json.dumps(
        {"pool": p_caps, "test": t_caps}))
    (OUT / "manifest.json").write_text(json.dumps({
        "pool_repo": POOL_REPO, "test_repo": TEST_REPO, "match_column": key,
        "removed_for_leakage": removed,
        "train_images": int((~val_flags).sum()), "val_images": int(val_flags.sum()),
        "test_images": len(test),
        "pool_captions": len(p_caps), "test_captions": len(t_caps),
        "model": enc.ckpt, "dim": int(p_img.shape[1]), "device": enc.device,
        "limited_to": a.limit,
    }, indent=2))

    # ---- zero-shot baseline, measured BEFORE any training ---------------
    print(f"\n--- zero-shot SigLIP 2, text -> image over {len(test)} candidates ---")
    scores = t_txt @ t_img.T
    order = np.argsort(-scores, axis=1)
    ranks = np.array([np.where(order[r] == t_owner[r])[0][0]
                      for r in range(len(t_owner))])
    res = {f"recall@{k}": float((ranks < k).mean()) for k in (1, 5, 10)}
    for k, v in res.items():
        print(f"    {k:<10}: {v:.4f}")
    print(f"    median rank: {np.median(ranks)+1:.0f}")
    res.update({"tag": "siglip2_zeroshot", "model": enc.ckpt,
                "test_images": len(test), "test_captions": len(t_owner),
                "median_rank": float(np.median(ranks) + 1)})
    (ROOT / "results" / "flickr30k_zeroshot.json").write_text(json.dumps(res, indent=2))

    print(f"\n  cached -> data/flickr30k/siglip_frozen.npz")
    print(f"  baseline -> results/flickr30k_zeroshot.json")
    print("\n  This baseline is what your trained model has to beat.")


if __name__ == "__main__":
    main()

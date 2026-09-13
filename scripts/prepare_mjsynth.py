#!/usr/bin/env python3
"""
Verify the MJSynth archive and extract a documented, reproducible subset.

    python scripts/prepare_mjsynth.py --verify-only
    python scripts/prepare_mjsynth.py --target 200000

WHY NOT JUST `tar -xzf`:
MJSynth is ~9 million JPEGs. Extracting all of them creates 9 million files
(9 million inodes), takes hours, and you will train on maybe 3% of them.
This makes ONE streaming pass and writes out only what you sampled.

HOW THE SUBSET IS CHOSEN, and why it is defensible in a review:
not "the first N files" - the archive is ordered, so the first N would be a
biased slice of the dataset. Instead each filename is hashed and kept if the
hash falls in a bucket. That is:
  - deterministic  (same subset every time, on any machine, no seed to lose)
  - unbiased       (hashing scatters the selection across the whole archive)
  - resumable      (a crash halfway does not change which files get picked)
  - describable    (you can state the exact rule in your report)

Labels live in the filename: .../366_TRAVERSING_80793.jpg -> "TRAVERSING"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = RAW / "mjsynth_subset"
MANIFESTS = ROOT / "data" / "manifests"

LABEL_RE = re.compile(r"^\d+_(.+)_\d+\.jpg$", re.IGNORECASE)
SEARCH = [
    RAW, Path.home() / "Downloads", Path.home() / "Desktop", Path.home(),
]


def find_archive(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            sys.exit(f"!!! not a file: {p}")
        return p
    for d in SEARCH:
        if not d.is_dir():
            continue
        for pat in ("mjsynth*.tar.gz", "*90k*.tar.gz", "*Synth90k*.tar.gz"):
            hits = sorted(d.glob(pat))
            if hits:
                return hits[0]
    sys.exit("!!! could not find the archive. Pass --archive /path/to/mjsynth.tar.gz")


def verify(archive: Path) -> dict:
    print(f"archive : {archive}")
    gb = archive.stat().st_size / 1_073_741_824
    print(f"size    : {gb:.2f} GB")
    if gb < 5:
        print("  !! WARNING: expected roughly 10 GB. This may be a partial download.")

    print("sha256  : hashing (about a minute for 10 GB) ...")
    h = hashlib.sha256()
    t0 = time.perf_counter()
    with archive.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    digest = h.hexdigest()
    print(f"          {digest}   ({time.perf_counter()-t0:.0f}s)")

    print("opening : reading first members to check structure ...")
    names, images = [], []
    with tarfile.open(archive, "r|gz") as tf:          # streaming, no seek
        for m in tf:
            names.append(m.name)
            if m.isfile() and m.name.lower().endswith(".jpg"):
                images.append(m.name)
            # keep going until we have actually SEEN some images - the first
            # ~25 members are annotation/lexicon files and directories.
            if len(images) >= 20 or len(names) >= 8000:
                break
    for n in names[:8]:
        print(f"          {n}")
    labelled = [n for n in images if LABEL_RE.match(Path(n).name)]
    print(f"\n  members scanned      : {len(names)}")
    print(f"  image members seen   : {len(images)}")
    if images:
        print(f"  example image path   : {images[0]}")
    print(f"  parse as labelled jpg: {len(labelled)}")
    if labelled:
        ex = Path(labelled[0]).name
        print(f"  example              : {ex}  ->  label {LABEL_RE.match(ex).group(1)!r}")
    else:
        print("  !! no filename matched the expected pattern. Inspect the names above.")
    return {"path": str(archive), "size_bytes": archive.stat().st_size,
            "sha256": digest, "first_members": names[:25]}


def bucket(name: str, k: int) -> bool:
    """Keep this file? Deterministic, uniform, independent of archive order."""
    return int(hashlib.sha1(name.encode()).hexdigest()[:8], 16) % k == 0


def extract(archive: Path, target: int, total_est: int = 8_900_000) -> dict:
    k = max(1, round(total_est / target))
    print(f"\nextracting ~1 in every {k} images  (target ~{target:,})")
    OUT.mkdir(parents=True, exist_ok=True)

    kept = skipped = bad = 0
    index: list[dict] = []
    t0 = time.perf_counter()

    with tarfile.open(archive, "r|gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            base = Path(m.name).name

            # annotation / lexicon files are small and always worth keeping
            if base.endswith(".txt"):
                try:
                    data = tf.extractfile(m).read()
                    (OUT / base).write_bytes(data)
                    print(f"  kept metadata file: {base} ({len(data)/1024:.0f} KB)")
                except Exception as exc:                       # noqa: BLE001
                    print(f"  [warn] {base}: {exc}")
                continue

            mo = LABEL_RE.match(base)
            if not mo:
                bad += 1
                continue
            if not bucket(m.name, k):
                skipped += 1
                continue

            dest = OUT / "images" / base
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                dest.write_bytes(tf.extractfile(m).read())
            except Exception:                                   # noqa: BLE001
                bad += 1
                continue
            index.append({"file": f"images/{base}", "label": mo.group(1),
                          "source_member": m.name})
            kept += 1

            if kept % 5000 == 0:
                el = time.perf_counter() - t0
                print(f"  kept {kept:,}  scanned {kept+skipped:,}  "
                      f"{el:.0f}s  ({(kept+skipped)/el:,.0f} members/s)")

    el = time.perf_counter() - t0
    print(f"\ndone in {el/60:.1f} min")
    print(f"  kept    : {kept:,}")
    print(f"  skipped : {skipped:,}")
    print(f"  unparsed: {bad:,}")

    # Assign splits from the OFFICIAL annotation files that ship in the
    # archive, not from our own hashing. The dataset authors defined these;
    # inventing our own split would make our numbers incomparable to every
    # published MJSynth result, for no benefit.
    official: dict[str, str] = {}
    for split in ("train", "val", "test"):
        f = OUT / f"annotation_{split}.txt"
        if not f.is_file():
            continue
        for line in f.read_text(errors="replace").splitlines():
            rel = line.split()[0] if line.split() else ""
            if rel:
                official[Path(rel).name] = split
    if official:
        print(f"  official split map loaded: {len(official):,} entries")
        for rec in index:
            rec["split"] = official.get(Path(rec["file"]).name, "unassigned")
    else:
        print("  !! no annotation_*.txt found - falling back to a hash split")
        for rec in index:
            rec["split"] = "val" if bucket(rec["file"] + "#split", 10) else "train"

    from collections import Counter as _C
    print(f"  split counts: {dict(_C(r['split'] for r in index))}")
    print("  NOTE: the CRNN is scored on IIIT-5K, a separate dataset of real")
    print("        photographs. MJSynth's own test split is not used at all,")
    print("        so there is no leakage path between training and scoring.")

    chars = sorted({c for r in index for c in r["label"]})
    print(f"  charset : {len(chars)} distinct -> {''.join(chars)[:70]}")
    lens = sorted(len(r["label"]) for r in index)
    if lens:
        print(f"  label len: min={lens[0]} median={lens[len(lens)//2]} max={lens[-1]}")

    MANIFESTS.mkdir(parents=True, exist_ok=True)
    (MANIFESTS / "mjsynth_subset.json").write_text(json.dumps(
        {"rule": f"sha1(member_name)[:8] % {k} == 0", "bucket_k": k,
         "kept": kept, "skipped": skipped, "unparsed": bad,
         "charset": chars, "records": index}, indent=2))
    print(f"  manifest -> data/manifests/mjsynth_subset.json")
    return {"kept": kept, "bucket_k": k, "charset_size": len(chars)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=None)
    ap.add_argument("--target", type=int, default=200_000)
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()

    arc = find_archive(a.archive)
    info = verify(arc)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    (MANIFESTS / "mjsynth_archive.json").write_text(json.dumps(info, indent=2))
    print("\narchive facts -> data/manifests/mjsynth_archive.json")

    if not a.verify_only:
        extract(arc, a.target)

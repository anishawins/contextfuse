#!/usr/bin/env python3
"""
Download + inspect the two public datasets ContextFuse depends on.

    source .venv/bin/activate
    python scripts/fetch_datasets.py

This script DOWNLOADS and REPORTS FACTS. It preprocesses nothing.
Every number it prints is measured, not assumed. Paste the output back.

Open questions this script exists to answer:
  Q1  Are the ViDoRe qrels binary or graded?   (decides whether nDCG is meaningful)
  Q2  How big is ViDoRe on disk, really?       (the brief claimed ~578 MB)
  Q3  What are the IIIT-5K train/test sizes?   (the project page does not say)
  Q4  What licence does IIIT-5K actually ship? (the project page does not say)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFESTS = ROOT / "data" / "manifests"
RAW.mkdir(parents=True, exist_ok=True)
MANIFESTS.mkdir(parents=True, exist_ok=True)

FACTS: dict = {}


def rule(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def die(msg: str) -> None:
    """Fail loudly. Never let a half-downloaded dataset proceed silently."""
    print(f"\n!!! FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / 1_048_576, 1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------
# 1. ViDoRe V3 - Computer Science
# ----------------------------------------------------------------------
def fetch_vidore() -> None:
    rule("1/2  ViDoRe V3 - Computer Science  (primary retrieval benchmark)")
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError:
        die("`datasets` not installed. Run scripts/setup.sh first.")

    repo = "vidore/vidore_v3_computer_science"

    # Ask the hub what configs exist rather than assuming their names.
    try:
        configs = get_dataset_config_names(repo)
    except Exception as exc:  # noqa: BLE001
        die(f"could not reach {repo}: {exc}\n"
            f"    If this is a 403, your network is blocking Hugging Face.")

    print(f"configs advertised by the hub: {configs}")
    FACTS["vidore_configs"] = configs

    loaded = {}
    for cfg in configs:
        try:
            ds = load_dataset(repo, cfg)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] config {cfg!r}: {exc}")
            continue
        loaded[cfg] = ds
        for split, d in ds.items():
            print(f"  {cfg:>22} / {split:<6}  rows={len(d):<6} cols={list(d.features)}")

    FACTS["vidore_shapes"] = {
        cfg: {sp: len(d) for sp, d in ds.items()} for cfg, ds in loaded.items()
    }

    # --- Q1: are relevance judgements binary or graded? -----------------
    qrels_cfg = next((c for c in loaded if "qrel" in c.lower()), None)
    if qrels_cfg is None:
        print("\n  [warn] no qrels config found - cannot answer Q1 automatically.")
    else:
        split = next(iter(loaded[qrels_cfg]))
        qrels = loaded[qrels_cfg][split]
        score_col = next(
            (c for c in qrels.column_names
             if c.lower() in {"score", "relevance", "rel", "label", "grade"}),
            None,
        )
        print(f"\n  --- Q1: relevance judgements ({qrels_cfg}/{split}) ---")
        print(f"  columns: {qrels.column_names}")
        print(f"  first row: {qrels[0]}")
        if score_col:
            dist = Counter(qrels[score_col])
            print(f"  distribution of {score_col!r}: {dict(sorted(dist.items()))}")
            graded = len(dist) > 2 or set(dist) - {0, 1}
            verdict = "GRADED  -> nDCG is meaningful" if graded else \
                      "BINARY  -> nDCG degenerates; Recall/MRR are the honest metrics"
            print(f"  VERDICT: {verdict}")
            FACTS["vidore_qrels_verdict"] = verdict
            FACTS["vidore_qrels_distribution"] = {str(k): v for k, v in dist.items()}
        else:
            print("  [warn] no obvious score column - inspect the columns above by hand.")

    # --- Q2: real on-disk size -----------------------------------------
    cache = Path.home() / ".cache" / "huggingface" / "datasets"
    size = dir_size_mb(cache)
    print(f"\n  --- Q2: HF datasets cache is now {size} MB total ---")
    print("      (includes anything cached earlier, so treat as an upper bound)")
    FACTS["hf_datasets_cache_mb"] = size


# ----------------------------------------------------------------------
# 2. IIIT 5K-Word
# ----------------------------------------------------------------------
IIIT_URL = "https://cvit.iiit.ac.in/images/Projects/SceneTextUnderstanding/IIIT5K-Word_V3.0.tar.gz"


def fetch_iiit5k() -> None:
    rule("2/2  IIIT 5K-Word  (independent OCR evaluation)")
    tarball = RAW / "IIIT5K-Word_V3.0.tar.gz"

    if tarball.exists():
        print(f"already present: {tarball.name} ({tarball.stat().st_size/1_048_576:.1f} MB)")
    else:
        print(f"downloading {IIIT_URL}")
        rc = subprocess.call(["curl", "-L", "-C", "-", "--fail",
                              "-o", str(tarball), IIIT_URL])
        if rc != 0 or not tarball.exists():
            die("IIIT-5K download failed. Open the CVIT project page in a browser "
                "and check the link is still live.")

    size_mb = round(tarball.stat().st_size / 1_048_576, 1)
    digest = sha256(tarball)
    print(f"  size   : {size_mb} MB   (project page states 106 MB)")
    print(f"  sha256 : {digest}")
    FACTS["iiit5k_size_mb"] = size_mb
    FACTS["iiit5k_sha256"] = digest
    FACTS["iiit5k_source_url"] = IIIT_URL

    extract_dir = RAW / "IIIT5K"
    if not extract_dir.exists():
        print("  extracting ...")
        with tarfile.open(tarball) as tf:
            tf.extractall(RAW)

    members = sorted(p for p in RAW.rglob("*") if p.is_dir() and "IIIT5K" in p.name)
    base = members[0] if members else extract_dir
    print(f"  extracted to: {base}")

    # --- Q3: what is actually inside? -----------------------------------
    top = sorted(p.name for p in base.iterdir())
    print(f"\n  --- Q3: top-level contents ---\n  {top}")
    for sub in ("train", "test"):
        d = base / sub
        if d.is_dir():
            n = sum(1 for _ in d.rglob("*.png")) + sum(1 for _ in d.rglob("*.jpg"))
            print(f"  {sub}/ contains {n} images")
            FACTS[f"iiit5k_{sub}_images"] = n

    # --- Q4: licence / terms -------------------------------------------
    print("\n  --- Q4: README / licence text ---")
    found = False
    for name in ("README", "README.txt", "readme.txt", "LICENSE", "UPDATES"):
        f = base / name
        if f.is_file():
            found = True
            print(f"  ----- {name} -----")
            print("  " + f.read_text(errors="replace")[:1500].replace("\n", "\n  "))
    if not found:
        print("  [warn] no README found - report this, do not assume a licence.")


def main() -> None:
    fetch_vidore()
    fetch_iiit5k()

    out = MANIFESTS / "dataset_facts.json"
    out.write_text(json.dumps(FACTS, indent=2, sort_keys=True))
    rule("DONE")
    print(f"measured facts written to: {out.relative_to(ROOT)}")
    print("Paste this whole terminal output back to Claude.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Three-way OCR comparison on the IIIT 5K-Word test split.

    python scripts/compare_ocr.py
    python scripts/compare_ocr.py --limit 500      # quick pass

  CRNN          trained here on a 162k MJSynth subset (synthetic renderings)
  Apple Vision  production engine, built into macOS
  Tesseract     the classic open-source baseline, if installed

SAME IMAGES, SAME GROUND TRUTH, SAME METRICS. That is the only way the row
comparison means anything.

ONE THING TO EXPECT, and it is the interesting part of this experiment:
Apple Vision and Tesseract are FULL-PAGE engines. They first DETECT where
text is, then recognise it. IIIT-5K images are already-cropped single words,
often small and low-resolution - so the detection stage has almost nothing to
work with and can fail on images the recogniser would have read fine. The
CRNN has no detector at all; it assumes the crop IS the word, which on this
dataset is exactly true.

So this is not "my model versus a production engine" in general. It is a
comparison on pre-cropped word images, which is the CRNN's native task and
not theirs. Say that in the report. A result that favours your model because
the benchmark suits it is only a finding if you name the reason.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _venv_check import require_venv                        # noqa: E402
require_venv()

from contextfuse.crnn import CRNN                           # noqa: E402
from contextfuse.ctc import (Alphabet, beam_decode, cer,     # noqa: E402
                             greedy_decode, edit_distance, wer)
from contextfuse.textdata import IIIT5K, preprocess         # noqa: E402


def run_crnn(items, root, alphabet, dev, beam=False, batch=64):
    ck = torch.load(ROOT / "data" / "crnn_best.pt", map_location=dev)
    model = CRNN(len(alphabet)).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()
    preds, t0 = [], time.perf_counter()
    with torch.no_grad():
        for s in range(0, len(items), batch):
            imgs = []
            for name, _ in items[s:s + batch]:
                p = root / name
                if not p.is_file():
                    p = root / "test" / Path(name).name
                imgs.append(preprocess(Image.open(p)))
            logits = model(torch.stack(imgs).to(dev))
            preds += (beam_decode(logits, alphabet) if beam
                      else greedy_decode(logits, alphabet))
    return preds, time.perf_counter() - t0, ck.get("val_cer")


def run_engine(items, root, alphabet, engine):
    preds, t0 = [], time.perf_counter()
    for name, _ in items:
        p = root / name
        if not p.is_file():
            p = root / "test" / Path(name).name
        try:
            txt = engine.read(p).text
        except Exception:                                    # noqa: BLE001
            txt = ""
        preds.append(alphabet.normalise(txt.replace("\n", "").replace(" ", "")))
    return preds, time.perf_counter() - t0


def failure_table(name, preds, gold):
    """Categorise errors so the report can say HOW each system fails."""
    empty = sum(1 for p in preds if not p)
    exact = sum(p == g for p, g in zip(preds, gold))
    off_by_one = sum(1 for p, g in zip(preds, gold)
                     if p != g and p and edit_distance(p, g) == 1)
    too_short = sum(1 for p, g in zip(preds, gold) if p and len(p) < len(g))
    too_long = sum(1 for p, g in zip(preds, gold) if p and len(p) > len(g))
    n = len(gold)
    print(f"  {name:<22} exact {exact/n:6.1%}   empty {empty/n:6.1%}   "
          f"off-by-1 char {off_by_one/n:6.1%}   short {too_short/n:5.1%}   "
          f"long {too_long/n:5.1%}")
    return {"exact": exact, "empty": empty, "off_by_one": off_by_one,
            "too_short": too_short, "too_long": too_long, "n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-beam", action="store_true")
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    alphabet = Alphabet()
    root = ROOT / "data" / "raw" / "IIIT5K"
    ds = IIIT5K(root, alphabet, "test")
    items = ds.items[:a.limit] if a.limit else ds.items
    gold = [g for _, g in items]
    print(f"IIIT 5K-Word test: {len(items)} images   labels from {ds.source}")
    print(f"device={dev}\n")

    runs, rows = {}, {}

    print("running CRNN (greedy) ...")
    p, t, val_cer = run_crnn(items, root, alphabet, dev, beam=False)
    runs["CRNN (greedy)"] = (p, t)
    if not a.skip_beam:
        print("running CRNN (beam, w=10) ...")
        p2, t2, _ = run_crnn(items, root, alphabet, dev, beam=True)
        runs["CRNN (beam w=10)"] = (p2, t2)

    from contextfuse.ocr import AppleVisionOCR, TesseractOCR
    for label, cls in (("Apple Vision", AppleVisionOCR), ("Tesseract", TesseractOCR)):
        try:
            eng = cls()
        except Exception as exc:                             # noqa: BLE001
            print(f"  [skip] {label}: {type(exc).__name__}: {str(exc)[:70]}")
            continue
        print(f"running {label} ...")
        runs[label] = run_engine(items, root, alphabet, eng)

    print(f"\n{'='*78}\n  OCR COMPARISON - IIIT 5K-Word test, {len(items)} cropped word images\n{'='*78}")
    print(f"  {'system':<22}{'CER':>9}{'WER':>9}{'word acc':>11}{'ms/img':>10}")
    for name, (preds, secs) in runs.items():
        c, w = cer(preds, gold), wer(preds, gold)
        rows[name] = {"cer": float(c), "wer": float(w),
                      "word_accuracy": float(1 - w),
                      "ms_per_image": secs / len(items) * 1000,
                      "total_seconds": secs}
        print(f"  {name:<22}{c:>9.4f}{w:>9.4f}{1-w:>10.1%}{secs/len(items)*1000:>10.1f}")

    print(f"\n{'='*78}\n  FAILURE MODES\n{'='*78}")
    fails = {name: failure_table(name, preds, gold) for name, (preds, _) in runs.items()}

    # ---- digits vs letters ------------------------------------------
    # Observed: the CRNN reads 41km as 'atkm' and 83km as 'e3km'. MJSynth is
    # rendered from a ~90,000-word ENGLISH LEXICON, so digits barely appear in
    # training. If that is the cause, accuracy should collapse specifically on
    # labels containing digits, and only for the CRNN. Measure it.
    print(f"\n{'='*78}\n  DIGITS vs LETTERS - does the training lexicon explain the errors?\n{'='*78}")
    has_digit = [any(c.isdigit() for c in g) for g in gold]
    n_dig = sum(has_digit)
    print(f"  labels containing a digit: {n_dig} of {len(gold)} ({n_dig/len(gold):.1%})")
    digit_rows = {}
    if n_dig >= 10:
        print(f"\n  {'system':<22}{'letters only':>16}{'with digits':>15}{'gap':>9}")
        for name, (preds, _) in runs.items():
            lp = [p for p, d in zip(preds, has_digit) if not d]
            lg = [g for g, d in zip(gold, has_digit) if not d]
            dp = [p for p, d in zip(preds, has_digit) if d]
            dg = [g for g, d in zip(gold, has_digit) if d]
            la, da = 1 - wer(lp, lg), 1 - wer(dp, dg)
            digit_rows[name] = {"word_acc_letters": float(la),
                                "word_acc_with_digits": float(da),
                                "gap": float(la - da), "n_digit_labels": len(dg)}
            print(f"  {name:<22}{la:>15.1%}{da:>15.1%}{la-da:>+9.1%}")
        print("""
  A large gap for the CRNN and a small one for a production engine confirms
  the cause is the TRAINING DISTRIBUTION, not the architecture: MJSynth's
  lexicon is English words, so the model saw almost no digits. The fix is
  data (render digit-bearing samples), not a different network.""")
    else:
        print("  too few digit-bearing labels in this subset to conclude anything")

    # ---- character confusion ------------------------------------------
    print(f"\n{'='*78}\n  CRNN's most common character substitutions\n{'='*78}")
    conf = Counter()
    for pr, g in zip(runs["CRNN (greedy)"][0], gold):
        if pr != g and len(pr) == len(g):          # same length: pure substitution
            for a_c, b_c in zip(g, pr):
                if a_c != b_c:
                    conf[f"{a_c} -> {b_c}"] += 1
    for k, v in conf.most_common(10):
        print(f"      {k:<10} {v}")

    print(f"\n{'='*78}\n  WHERE THEY DISAGREE\n{'='*78}")
    keys = list(runs)
    if len(keys) >= 2:
        a_name, b_name = keys[0], keys[-1]
        A, B = runs[a_name][0], runs[b_name][0]
        a_only = [(g, x, y) for g, x, y in zip(gold, A, B) if x == g and y != g]
        b_only = [(g, x, y) for g, x, y in zip(gold, A, B) if y == g and x != g]
        print(f"  {a_name} right, {b_name} wrong : {len(a_only)}")
        for g, x, y in a_only[:5]:
            print(f"      {g:<14} {a_name[:12]}={x!r:<16} {b_name[:12]}={y!r}")
        print(f"  {b_name} right, {a_name} wrong : {len(b_only)}")
        for g, x, y in b_only[:5]:
            print(f"      {g:<14} {a_name[:12]}={x!r:<16} {b_name[:12]}={y!r}")

    out = ROOT / "results" / "ocr_comparison_iiit5k.json"
    out.write_text(json.dumps({
        "tag": "ocr_comparison_iiit5k",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"name": "IIIT 5K-Word test", "images": len(items),
                    "label_source": ds.source},
        "device": dev, "crnn_mjsynth_val_cer": val_cer,
        "metrics": rows, "failure_modes": fails,
        "digit_analysis": digit_rows,
        "crnn_top_substitutions": dict(conf.most_common(15)),
        "caveat": "IIIT-5K images are pre-cropped single words. The CRNN assumes "
                  "the crop is the word; Apple Vision and Tesseract include a text "
                  "DETECTION stage that this benchmark does not exercise and that "
                  "can fail on small crops. The comparison is on the CRNN's native "
                  "task, not a general claim about the engines.",
    }, indent=2, default=str))
    print(f"\n  saved -> results/{out.name}")


if __name__ == "__main__":
    main()

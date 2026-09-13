#!/usr/bin/env python3
"""
Train the CRNN on MJSynth, evaluate on IIIT 5K-Word.

    python scripts/train_crnn.py --smoke          # 2000 images, ~2 min sanity run
    python scripts/train_crnn.py --epochs 8

TRAIN AND TEST COME FROM DIFFERENT DATASETS, on purpose.
Training is MJSynth (synthetic renderings); evaluation is IIIT-5K (real
photographs). There is no path by which a test image could appear in
training, which is a stronger guarantee than an in-distribution split. It is
also the standard published protocol, so the CER/WER are comparable to the
literature.

The cost is a DOMAIN SHIFT baked into the number: the model reads photographs
having only ever seen synthetic text. Report it as such.

WHAT TO WATCH: validation CER should fall steadily. If the loss falls while
CER stays at 1.0, the network has collapsed to emitting only blanks - the
degenerate solution CTC falls into when the learning rate is too high early,
because predicting "blank everywhere" is a local minimum that satisfies most
of the loss surface.
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _venv_check import require_venv                       # noqa: E402
require_venv()

from contextfuse.crnn import CRNN                          # noqa: E402
from contextfuse.ctc import (Alphabet, beam_decode, cer,    # noqa: E402
                             greedy_decode, wer)
from contextfuse.textdata import IIIT5K, MJSynthSubset, collate  # noqa: E402


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else \
           ("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, loader, alphabet, dev, beam: bool = False,
             limit: int | None = None):
    """
    Returns (CER, WER, predictions, gold, blank_fraction).

    BLANK FRACTION is the diagnostic that matters during early CTC training:
    the proportion of output frames whose argmax is the blank symbol.

    CTC reliably begins by predicting blank everywhere - it is a wide, shallow
    local minimum that satisfies most of the loss surface, and every model
    passes through it. The question is whether it is CLIMBING OUT. Loss alone
    cannot tell you: a model stuck at all-blanks still shows a slowly falling
    loss. Blank fraction can.

        1.00        emitting nothing. Normal for the first few thousand steps.
        falling     climbing out. This is what progress looks like.
        ~0.5-0.7    healthy - a converged CRNN emits blanks between characters,
                    so the fraction never approaches zero.
    """
    model.eval()
    preds, gold = [], []
    blank, total = 0, 0
    for i, (imgs, _, _, labels) in enumerate(loader):
        logits = model(imgs.to(dev))
        arg = logits.argmax(dim=2)
        blank += int((arg == 0).sum())
        total += arg.numel()
        preds += (beam_decode(logits, alphabet) if beam
                  else greedy_decode(logits, alphabet))
        gold += labels
        if limit and len(gold) >= limit:
            break
    return cer(preds, gold), wer(preds, gold), preds, gold, blank / max(1, total)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lstm-hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run to prove the pipeline works end to end")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="continue from data/crnn_best.pt")
    a = ap.parse_args()
    if a.smoke:
        a.epochs, a.max_train = 2, 2000

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = device()
    alphabet = Alphabet()
    print(f"device={dev}  alphabet={len(alphabet)} classes (36 + blank)")

    # ---- data -----------------------------------------------------------
    man = ROOT / "data" / "manifests" / "mjsynth_subset.json"
    root = ROOT / "data" / "raw" / "mjsynth_subset"
    if not man.is_file():
        sys.exit("!!! run scripts/prepare_mjsynth.py --target 200000 first")

    train = MJSynthSubset(root, man, "train", alphabet)
    val = MJSynthSubset(root, man, "val", alphabet)
    if len(val) == 0:                     # official split file was absent
        cut = int(len(train) * 0.95)
        train.items, val.items = train.items[:cut], train.items[cut:]
        print("  [note] no official val split found; carved 5% off train")
    if a.max_train:
        train.items = train.items[:a.max_train]
        val.items = val.items[:max(200, a.max_train // 10)]

    iiit = IIIT5K(ROOT / "data" / "raw" / "IIIT5K", alphabet, "test")
    print(f"  MJSynth train={len(train):,}  val={len(val):,}  "
          f"(dropped {train.skipped:,} unusable labels)")
    print(f"  IIIT-5K test={len(iiit):,}  labels from {iiit.source}")
    if len(iiit) == 0:
        sys.exit("!!! IIIT-5K produced no items - check data/raw/IIIT5K")

    # functools.partial, not a lambda: DataLoader workers are separate
    # processes and their arguments must be picklable (Python 3.14 changed the
    # default start method). A closure defined here cannot be pickled.
    coll = functools.partial(collate, alphabet=alphabet)
    tl = DataLoader(train, batch_size=a.batch_size, shuffle=True,
                    collate_fn=coll, num_workers=a.workers, drop_last=True)
    vl = DataLoader(val, batch_size=a.batch_size, collate_fn=coll,
                    num_workers=a.workers)
    il = DataLoader(iiit, batch_size=a.batch_size, collate_fn=coll,
                    num_workers=a.workers)

    # ---- model ----------------------------------------------------------
    model = CRNN(len(alphabet), lstm_hidden=a.lstm_hidden, dropout=a.dropout).to(dev)
    print(f"  CRNN parameters: {model.n_params():,}")
    if a.resume:
        ck = ROOT / "data" / "crnn_best.pt"
        if ck.is_file():
            saved = torch.load(ck, map_location=dev)
            model.load_state_dict(saved["state"])
            print(f"  resumed from epoch {saved['epoch']} "
                  f"(val CER {saved['val_cer']:.4f})")
        else:
            print("  [warn] --resume given but no checkpoint found; starting fresh")
    with torch.no_grad():
        T = model(torch.zeros(2, 1, 32, 100, device=dev)).shape[0]
    print(f"  output timesteps T={T}  (max emittable label length ~{T//2})")

    # blank=0 matches Alphabet's layout. zero_infinity guards the case where a
    # label is longer than the frame count, which would otherwise give inf loss.
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    c0, w0, _, _, bf0 = evaluate(model, vl, alphabet, dev, limit=500)
    print(f"\n  epoch -1 (untrained)   val CER {c0:.4f}  WER {w0:.4f}  "
          f"blank {bf0:.2f}")

    history, best, best_state = [], float("inf"), None
    t0 = time.perf_counter()
    for ep in range(a.epochs):
        model.train()
        losses = []
        for imgs, targets, tgt_len, _ in tqdm(tl, desc=f"epoch {ep}", unit="b"):
            logits = model(imgs.to(dev))                      # [T, B, C]
            logp = logits.log_softmax(2)
            in_len = torch.full((logits.shape[1],), logits.shape[0],
                                dtype=torch.long)
            # CTCLoss wants log-probs on CPU-friendly layout [T, B, C], the
            # targets CONCATENATED (not padded), and both length vectors.
            loss = criterion(logp, targets.to(dev), in_len, tgt_len)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)   # CTC gradients spike
            opt.step()
            losses.append(loss.item())
        sched.step()

        c, w, ex, gd, bf = evaluate(model, vl, alphabet, dev, limit=1000)
        history.append({"epoch": ep, "loss": float(np.mean(losses)),
                        "val_cer": float(c), "val_wer": float(w),
                        "blank_fraction": float(bf)})
        flag = ""
        if c < best:
            best, flag = c, "  <- best"
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            # Write every improvement to disk. Training runs for hours; losing
            # it to a Ctrl-C, a closed lid or a crash would be avoidable waste.
            torch.save({"state": best_state, "alphabet": alphabet.chars,
                        "epoch": ep, "val_cer": float(c), "history": history},
                       ROOT / "data" / "crnn_best.pt")
        sample = next((f"{g}->{e}" for e, g in zip(ex, gd) if e), "(all blank)")
        print(f"  epoch {ep:2d}  loss {np.mean(losses):7.4f}  "
              f"val CER {c:.4f}  WER {w:.4f}  blank {bf:.2f}  {sample}{flag}")

    print(f"  trained in {(time.perf_counter()-t0)/60:.1f} min")
    if best_state:
        model.load_state_dict(best_state)
        print(f"  restored best checkpoint (val CER {best:.4f})")

    # ---- IIIT-5K, greedy and beam ---------------------------------------
    print("\nevaluating on IIIT 5K-Word (real photographs, unseen distribution)")
    g_cer, g_wer, g_pred, gold, g_bf = evaluate(model, il, alphabet, dev, beam=False)
    print(f"  greedy decoding    CER {g_cer:.4f}   WER {g_wer:.4f}  "
          f"blank fraction {g_bf:.2f}")
    b_cer, b_wer, b_pred, _, _ = evaluate(model, il, alphabet, dev, beam=True)
    print(f"  beam search (w=10) CER {b_cer:.4f}   WER {b_wer:.4f}")
    print(f"  beam changed {sum(x != y for x, y in zip(g_pred, b_pred))} "
          f"of {len(g_pred)} predictions")

    print("\n  --- 10 examples (gold -> greedy) ---")
    for i in range(min(10, len(gold))):
        mark = "ok " if g_pred[i] == gold[i] else "XX "
        print(f"   {mark} {gold[i]:<16} -> {g_pred[i]}")

    out = ROOT / "results" / "crnn_iiit5k.json"
    out.write_text(json.dumps({
        "tag": "crnn_mjsynth_to_iiit5k",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": vars(a), "device": dev, "params": model.n_params(),
        "timesteps": int(T), "alphabet_size": len(alphabet),
        "train_images": len(train), "val_images": len(val),
        "test_images": len(iiit), "iiit_label_source": iiit.source,
        "untrained_val_cer": float(c0),
        "final_blank_fraction": float(g_bf),
        "history": history,
        "iiit5k": {"greedy": {"cer": float(g_cer), "wer": float(g_wer)},
                   "beam10": {"cer": float(b_cer), "wer": float(b_wer)}},
        "note": "trained on synthetic MJSynth, tested on real IIIT-5K photographs; "
                "the reported error includes that domain shift",
    }, indent=2, default=str))
    torch.save({"state": model.state_dict(), "alphabet": alphabet.chars},
               ROOT / "data" / "crnn.pt")
    print(f"\n  saved -> results/{out.name}  and  data/crnn.pt")
    print("  next: scripts/compare_ocr.py for the CRNN vs Apple Vision table")


if __name__ == "__main__":
    main()

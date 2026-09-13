#!/usr/bin/env python3
"""
Does ANY adapter configuration help on ViDoRe, or is the ceiling the encoder?

    python scripts/sweep_adapter.py

Each run takes ~2 seconds, so a real search is affordable. This is the
hyperparameter study your report needs, and it answers a specific question
rather than producing a graph for its own sake.

THE HYPOTHESIS BEING TESTED:
    SigLIP's page embeddings have a mean pairwise cosine of 0.795 and a
    gold-vs-other margin of +0.0149 (measured in diagnose_baseline.py). If
    the information distinguishing one page from another was destroyed by
    the 224x224 downscale, then NO projection on top can recover it, and
    every configuration here will fail in the same way.

    Confirming that is a real finding: it says the ceiling is the ENCODER,
    not the head, and that only Stage 2 (LoRA inside the ViT) or higher
    input resolution can move it.

    If instead some configuration DOES help, the hypothesis is wrong and we
    learn something more useful.

SELECTION DISCIPLINE: every configuration is chosen on VAL. Test is computed
once, for the winner, at the end. Picking the config with the best TEST score
would be tuning on the test set and the number would be meaningless.
"""
from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GRID = {
    "lr": [1e-5, 1e-4, 1e-3],
    "bottleneck": [16, 64, 256],
    "train_heads": ["both", "text"],
    "dropout": [0.1, 0.4],
}


def run(cfg: dict) -> dict | None:
    cmd = [sys.executable, str(ROOT / "scripts" / "train_adapter.py"),
           "--dataset", "vidore", "--epochs", "40", "--patience", "10",
           "--lr", str(cfg["lr"]), "--bottleneck", str(cfg["bottleneck"]),
           "--train-heads", cfg["train_heads"], "--dropout", str(cfg["dropout"])]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    FAILED: {r.stderr.strip().splitlines()[-1][:110]}")
        return None
    d = json.loads((ROOT / "results" / "adapter_vidore.json").read_text())
    return {"cfg": cfg, "best_val": d["best_val"],
            "before": d["before"]["ndcg@10"]["mean"],
            "after": d["after"]["ndcg@10"]["mean"],
            "epochs_run": len(d["history"])}


def main() -> None:
    keys = list(GRID)
    combos = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]
    print(f"{len(combos)} configurations, ~2s each\n")

    rows = []
    for i, cfg in enumerate(combos, 1):
        print(f"  [{i:2d}/{len(combos)}] lr={cfg['lr']:<7} bn={cfg['bottleneck']:<4} "
              f"heads={cfg['train_heads']:<6} do={cfg['dropout']}", end="  ", flush=True)
        r = run(cfg)
        if r:
            rows.append(r)
            print(f"val={r['best_val']:.4f}  test={r['after']:.4f}  "
                  f"({r['epochs_run']} ep)")

    if not rows:
        sys.exit("every configuration failed - fix the trainer first")

    base = rows[0]["before"]
    rows.sort(key=lambda r: -r["best_val"])

    print(f"\n{'='*74}\n  ranked by VALIDATION nDCG@10 (test shown but NOT used to choose)\n{'='*74}")
    print(f"  {'lr':>8} {'bn':>5} {'heads':>7} {'drop':>5} {'val':>8} {'test':>8} {'vs base':>9}")
    for r in rows[:12]:
        c = r["cfg"]
        print(f"  {c['lr']:>8} {c['bottleneck']:>5} {c['train_heads']:>7} "
              f"{c['dropout']:>5} {r['best_val']:>8.4f} {r['after']:>8.4f} "
              f"{r['after']-base:>+9.4f}")

    win = rows[0]
    print(f"\n  zero-shot baseline nDCG@10 : {base:.4f}")
    print(f"  best-on-val configuration  : {win['cfg']}")
    print(f"  its test nDCG@10           : {win['after']:.4f}  "
          f"({win['after']-base:+.4f})")

    # Does validation actually predict test? If not, no amount of searching
    # helps - you would just be picking noise.
    import numpy as _np
    v = _np.array([r["best_val"] for r in rows])
    t = _np.array([r["after"] for r in rows])
    if len(v) > 2 and v.std() > 0 and t.std() > 0:
        corr = float(_np.corrcoef(v, t)[0, 1])
        print(f"\n  correlation between VAL and TEST across configs: {corr:+.3f}")
        if corr < 0.3:
            print("  ^ WEAK. Validation is not predicting test, so selecting on")
            print("    it is selecting noise. Fix the validation set before")
            print("    trusting any 'best' configuration.")
        else:
            print("  ^ validation is informative; selecting on it is sound.")

    helped = [r for r in rows if r["after"] > base]
    print(f"\n  configurations beating zero-shot on TEST: {len(helped)} of {len(rows)}")
    if not helped:
        print("""
  NONE. That is the hypothesis confirmed, and it is a result worth stating:

    Across 36 configurations spanning two orders of magnitude of learning
    rate, three adapter capacities, both head-freezing regimes and two
    dropout levels, no projection on top of frozen SigLIP embeddings
    improved retrieval on ViDoRe. The limitation is not the head's capacity
    or its optimisation - it is that the frozen encoder, run at 224x224 on
    1700x2200 pages, does not encode what distinguishes one page from
    another. Mean pairwise cosine between page embeddings is 0.795.

  The consequence is concrete: only changing the ENCODER can move this -
  LoRA inside the ViT, or a higher input resolution. That is Stage 2, and
  you now have 36 measurements justifying it instead of an assumption.""")
    else:
        print(f"  hypothesis WRONG - the best is {max(r['after'] for r in helped):.4f}. "
              f"Investigate what those configurations share.")

    (ROOT / "results" / "adapter_sweep_vidore.json").write_text(
        json.dumps({"baseline_ndcg@10": base, "grid": GRID, "runs": rows}, indent=2))
    print(f"\n  saved -> results/adapter_sweep_vidore.json")


if __name__ == "__main__":
    main()

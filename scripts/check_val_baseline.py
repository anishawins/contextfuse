#!/usr/bin/env python3
"""
Was the training curve real, or was it recovery from self-inflicted damage?

The trainer logged validation AFTER each epoch, so the first value printed
(0.4941) already included 286 update steps. The true starting point was
never recorded. This measures it.

A learning curve without its zero point is not evidence of learning.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _venv_check import require_venv                    # noqa: E402
require_venv()
from contextfuse.adapter import DualAdapter             # noqa: E402

z = np.load(ROOT / "data" / "flickr30k" / "siglip_frozen.npz")
pool_img, pool_txt = z["pool_img"], z["pool_txt"]
owner, is_val = z["pool_owner"], z["pool_is_val"]
val_mask = is_val[owner]
val_txt, val_owner = pool_txt[val_mask], owner[val_mask]
print(f"val captions {len(val_txt)}   candidates {len(pool_img)}")


def recall_at(txt, img, gold, ks=(1, 5, 10)):
    out = {k: 0 for k in ks}
    for s in range(0, len(txt), 256):
        sc = txt[s:s + 256] @ img.T
        g = gold[s:s + 256]
        rank = (sc > sc[np.arange(len(g)), g][:, None]).sum(1) + 1
        for k in ks:
            out[k] += int((rank <= k).sum())
    return {k: v / len(txt) for k, v in out.items()}


print("\n--- ZERO-SHOT (no adapter at all) ---")
zs = recall_at(val_txt, pool_img, val_owner)
for k, v in zs.items():
    print(f"  Recall@{k:<3}: {v:.4f}")

print("\n--- UNTRAINED adapter (identity init) ---")
m = DualAdapter(pool_img.shape[1], 256, 0.1); m.eval()
with torch.no_grad():
    i0 = m.image(torch.tensor(pool_img)).numpy()
    t0 = m.text(torch.tensor(val_txt)).numpy()
u = recall_at(t0, i0, val_owner)
for k, v in u.items():
    print(f"  Recall@{k:<3}: {v:.4f}")

print("\n--- TRAINED adapter ---")
m.load_state_dict(torch.load(ROOT / "data" / "adapter_flickr.pt", map_location="cpu"))
m.eval()
with torch.no_grad():
    i1 = m.image(torch.tensor(pool_img)).numpy()
    t1 = m.text(torch.tensor(val_txt)).numpy()
tr = recall_at(t1, i1, val_owner)
for k, v in tr.items():
    print(f"  Recall@{k:<3}: {v:.4f}")

print(f"\n{'='*60}")
d = tr[10] - zs[10]
print(f"  zero-shot R@10 {zs[10]:.4f}  ->  trained {tr[10]:.4f}   {d:+.4f}")
if d < 0:
    print("""
  CONFIRMED: the curve was recovery, not learning. The adapter degraded the
  representation early and never returned to its starting point. The trainer
  must log validation BEFORE the first update - otherwise any curve looks
  like progress.""")
else:
    print("\n  The gain is real on this split; the hard-test regression needs another cause.")

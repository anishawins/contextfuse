#!/usr/bin/env python3
"""
De-risk week 7 in week 1.

MPS (Apple's GPU backend) does not implement every PyTorch operation.
The CRNN needs LSTM + CTC loss. If either fails on MPS, the CRNN trains
on CPU and the MJSynth subset has to shrink accordingly.

Find out now, in 20 seconds, rather than in six weeks.

    source .venv/bin/activate
    python scripts/check_mps_ops.py
"""
import time

import torch
import torch.nn as nn

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"torch {torch.__version__} | testing on: {DEV}\n")

results = {}


def probe(name, fn):
    try:
        t0 = time.perf_counter()
        fn()
        if DEV == "mps":
            torch.mps.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        results[name] = "OK"
        print(f"  {name:<34} OK      ({dt:.0f} ms)")
    except Exception as exc:                       # noqa: BLE001
        results[name] = f"FAIL: {type(exc).__name__}"
        print(f"  {name:<34} FAIL    {type(exc).__name__}: {str(exc)[:110]}")


print("--- ops the CRNN depends on -------------------------------------")

# CNN feature extractor
probe("Conv2d forward+backward", lambda: (
    nn.Conv2d(1, 64, 3, padding=1).to(DEV)(torch.randn(8, 1, 32, 100, device=DEV))
).sum().backward())

# feature map -> sequence -> recurrent layer
probe("LSTM (bidirectional) forward", lambda:
      nn.LSTM(64, 128, bidirectional=True, batch_first=True).to(DEV)(
          torch.randn(8, 25, 64, device=DEV)))

def _lstm_bwd():
    m = nn.LSTM(64, 128, bidirectional=True, batch_first=True).to(DEV)
    out, _ = m(torch.randn(8, 25, 64, device=DEV))
    out.sum().backward()
probe("LSTM backward", _lstm_bwd)

# CTC - the one most likely to break
def _ctc(dev):
    T, N, C = 25, 8, 37          # timesteps, batch, classes (36 alnum + blank)
    logp = torch.randn(T, N, C, device=dev).log_softmax(2).requires_grad_()
    targets = torch.randint(1, C, (N, 8), dtype=torch.long, device=dev)
    in_len = torch.full((N,), T, dtype=torch.long)
    tg_len = torch.full((N,), 8, dtype=torch.long)
    loss = nn.CTCLoss(blank=0, zero_infinity=True)(logp, targets, in_len, tg_len)
    loss.backward()
    return loss.item()

probe("CTCLoss forward+backward", lambda: _ctc(DEV))

print("\n--- CPU sanity check (must always pass) -------------------------")
try:
    print(f"  CTCLoss on CPU                     OK      loss={_ctc('cpu'):.4f}")
except Exception as exc:                            # noqa: BLE001
    print(f"  CTCLoss on CPU                     FAIL    {exc}")

print("\n--- ops the retrieval side depends on ----------------------------")
probe("matmul 5000x768 @ 768x1", lambda:
      torch.randn(5000, 768, device=DEV) @ torch.randn(768, 1, device=DEV))
probe("F.normalize", lambda:
      torch.nn.functional.normalize(torch.randn(5000, 768, device=DEV), dim=-1))
probe("topk over 5000", lambda:
      torch.randn(5000, device=DEV).topk(10))

print("\n" + "=" * 66)
bad = [k for k, v in results.items() if v != "OK"]
if bad:
    print("VERDICT: these fall back to CPU or must be worked around:")
    for b in bad:
        print(f"   - {b}  ({results[b]})")
    print("\nThe CRNN plan changes. Paste this to Claude.")
else:
    print("VERDICT: every op the CRNN and retriever need runs on MPS.")
    print("Week 7 can train on the GPU. Paste this to Claude anyway.")

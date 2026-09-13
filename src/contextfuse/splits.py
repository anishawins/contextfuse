"""
DEV / TEST separation.

Fusion weights are HYPERPARAMETERS. Choosing the weight that maximises a
score on the same queries you then report is tuning on your test set. The
number you publish would be optimistically biased and you would not be able
to say by how much.

Split by INFORMATION NEED, not by query row - the six language variants of
one question share the same relevant pages, so putting one in dev and its
translation in test leaks the answer across the boundary.

Hash-based, so the split is identical on every machine and every run, with
no seed file to lose.
"""
from __future__ import annotations

import hashlib


def dev_or_test(need_id: int, dev_fraction: float = 0.5) -> str:
    h = int(hashlib.sha1(f"need-{need_id}".encode()).hexdigest()[:8], 16)
    return "dev" if (h % 1000) < dev_fraction * 1000 else "test"


def split_queries(need_of: dict[int, int], dev_fraction: float = 0.5
                  ) -> tuple[list[int], list[int]]:
    dev, test = [], []
    for qid, need in need_of.items():
        (dev if dev_or_test(need, dev_fraction) == "dev" else test).append(qid)
    return sorted(dev), sorted(test)

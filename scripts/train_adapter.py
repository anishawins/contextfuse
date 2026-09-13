#!/usr/bin/env python3
"""
Stage 1: train a residual adapter on frozen SigLIP embeddings.

    python scripts/train_adapter.py --dataset vidore
    python scripts/train_adapter.py --dataset flickr

WHAT IS BEING TRAINED: only the adapter (a residual bottleneck MLP) plus the
loss's temperature and bias. SigLIP itself is frozen and was run once, so an
epoch costs seconds instead of minutes.

VIDORE SETUP, and why it differs from the textbook caption case:
    every step scores a batch of QUERIES against ALL 1360 page embeddings,
    not just the pages in the batch. The corpus is small and cached, so full
    -corpus negatives are free - and they are strictly better than in-batch
    negatives, because the model sees every distractor it will face at
    evaluation time rather than a random 32 of them.

    A query can have several relevant pages (median 4). The sigmoid loss
    handles that natively: each cell is an independent yes/no question, so
    multiple positives in a row need no special treatment. A softmax loss
    would have to divide probability mass among them.

SPLIT DISCIPLINE: train on DEV information needs, evaluate on TEST needs,
using the same hash split as every other experiment in this project. The
adapter never sees a test query or its relevance labels.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _venv_check import require_venv                          # noqa: E402
require_venv()

from contextfuse.adapter import DualAdapter                   # noqa: E402
from contextfuse.losses import SigmoidContrastiveLoss         # noqa: E402
from contextfuse.metrics import evaluate, format_table        # noqa: E402


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


# ----------------------------------------------------------------------
def load_vidore():
    """
    THREE-way split, and a use for the multilingual structure.

    TRAIN   all SIX language variants of the training needs
    VAL     english only, on held-out needs, for early stopping
    TEST    english only, the SAME 107 queries every other experiment in this
            project used - so the number stays directly comparable

    Why all six languages for training: there are only ~215 english queries,
    which is far too few to fit even a small adapter without memorising them.
    Each need exists as six translations sharing the same relevant pages, so
    training on all of them gives ~6x the supervision for free. It is
    translation-as-augmentation, and it is safe because the split is by NEED -
    a need is entirely in train or entirely in test, so no translation of a
    test question can leak into training.

    (SigLIP 2 is multilingual, which is what makes the non-english variants
    usable as training signal at all rather than noise.)
    """
    from contextfuse.data import load_vidore_cs
    from contextfuse.embed import SigLIPEncoder
    from contextfuse.splits import dev_or_test, split_queries
    import hashlib

    bench_all = load_vidore_cs(language=None)      # all 1290
    bench = load_vidore_cs("english")              # the 215 english
    z = np.load(next((ROOT / "data" / "embeddings").glob("vidore_cs__*.npz")))
    pages, ids = z["emb"], [int(i) for i in z["ids"]]
    pos_of = {cid: i for i, cid in enumerate(ids)}

    qids = sorted(bench_all.queries)               # ALL languages
    cache = ROOT / "data" / "embeddings" / "vidore_cs_queries_all.npz"
    if cache.exists():
        z2 = np.load(cache)
        q_emb, cached = z2["emb"], [int(i) for i in z2["qids"]]
        if cached != qids:
            raise RuntimeError("query cache is stale - delete "
                               f"{cache.name} and rerun")
        print(f"  loaded cached query embeddings {q_emb.shape}")
    else:
        q_emb = SigLIPEncoder().encode_texts([bench_all.queries[q] for q in qids])
        np.savez_compressed(cache, emb=q_emb, qids=np.asarray(qids))
        print(f"  encoded {len(qids)} queries -> {cache.name}")

    row = {q: i for i, q in enumerate(qids)}

    # TEST: exactly the english test queries used by every other experiment
    eng = sorted(bench.queries)
    _, test = split_queries({q: bench.need_of[q] for q in eng})
    test_needs = {bench.need_of[q] for q in test}

    # Carve VAL out of the remaining needs, again by a stable hash.
    def is_val(need: int) -> bool:
        return int(hashlib.sha1(f"val-{need}".encode()).hexdigest()[:8], 16) % 100 < 22

    train_needs, val_needs = set(), set()
    for q in eng:
        n = bench.need_of[q]
        if n in test_needs:
            continue
        (val_needs if is_val(n) else train_needs).add(n)

    train = [q for q in qids if bench_all.need_of.get(q) in train_needs]   # all langs
    # VAL also uses all six languages. With english only this was 26 queries -
    # far too few to rank configurations: in the first sweep the best-on-val
    # configuration scored 0.167 on val and 0.064 on test, while the fourth
    # placed one gave the best test score. That is not overfitting the model,
    # it is a validation SET too small to measure with. Six translations of
    # ~25 needs gives ~150 queries and a usable estimate. Still no leakage:
    # the split is by need, and no test need appears in any language here.
    val = [q for q in qids if bench_all.need_of.get(q) in val_needs]
    print(f"  needs: train={len(train_needs)} val={len(val_needs)} test={len(test_needs)}")
    print(f"  queries: train={len(train)} (6 langs)  val={len(val)} (6 langs)  "
          f"test={len(test)} (en, unchanged from every other experiment)")
    assert not (train_needs & test_needs) and not (val_needs & test_needs), \
        "need-level leakage between splits"

    P = np.zeros((len(qids), len(ids)), dtype=bool)
    for q in qids:
        for cid in bench_all.qrels[q]:
            if cid in pos_of:
                P[row[q], pos_of[cid]] = True

    return dict(q_emb=q_emb, pages=pages, P=P, qids=qids, row=row,
                train=train, val=val, test=test,
                page_ids=ids, bench=bench_all)


def load_flickr():
    z = np.load(ROOT / "data" / "flickr30k" / "siglip_frozen.npz")
    pool_img, pool_txt = z["pool_img"], z["pool_txt"]
    owner, is_val = z["pool_owner"], z["pool_is_val"]
    tr = ~is_val[owner]
    return dict(train_txt=pool_txt[tr], train_owner=owner[tr],
                train_img=pool_img, val_txt=pool_txt[~tr],
                val_owner=owner[~tr],
                test_img=z["test_img"], test_txt=z["test_txt"],
                test_owner=z["test_owner"])


# ----------------------------------------------------------------------
def eval_vidore(model, d, subset, k=10):
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        pages = model.image(torch.tensor(d["pages"], device=dev)).cpu().numpy()
        q = model.text(torch.tensor(d["q_emb"], device=dev)).cpu().numpy()
    rankings = {}
    for qid in subset:
        s = q[d["row"][qid]] @ pages.T
        top = np.argsort(-s)[:k]
        rankings[qid] = [d["page_ids"][j] for j in top]
    return evaluate(rankings, {qid: d["bench"].qrels[qid] for qid in subset})


def eval_flickr(model, img, txt, owner):
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        I = model.image(torch.tensor(img, device=dev)).cpu().numpy()
        T = model.text(torch.tensor(txt, device=dev)).cpu().numpy()
    order = np.argsort(-(T @ I.T), axis=1)
    ranks = np.array([np.where(order[r] == owner[r])[0][0] for r in range(len(owner))])
    return {f"recall@{k}": float((ranks < k).mean()) for k in (1, 5, 10)}


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["vidore", "flickr"], default="vidore")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--bottleneck", type=int, default=64,
                    help="small by default: ~700 training queries "
                         "cannot support a large adapter")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--shared-head", action="store_true")
    ap.add_argument("--train-heads", choices=["both", "text", "image"],
                    default="both",
                    help="'text' freezes the page side: the corpus geometry "
                         "stays exactly as SigLIP produced it and only the "
                         "query projection moves. Far fewer effective degrees "
                         "of freedom, which matters with ~80 training needs.")
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--warmup-steps", type=int, default=200,
                    help="linear LR warmup. The adapter's output layer starts "
                         "at 1e-4 so it acts as an identity; Adam moves a "
                         "weight by ~lr per step REGARDLESS of gradient size, "
                         "so at lr=1e-4 that initialisation is destroyed "
                         "within a few steps. Warmup keeps the first updates "
                         "proportionate to where the weights actually start.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = device()
    print(f"device={dev}  dataset={a.dataset}  seed={a.seed}")

    d = load_vidore() if a.dataset == "vidore" else load_flickr()
    dim = (d["pages"] if a.dataset == "vidore" else d["train_img"]).shape[1]

    model = DualAdapter(dim, a.bottleneck, a.dropout, shared=a.shared_head).to(dev)
    loss_fn = SigmoidContrastiveLoss().to(dev)
    if a.train_heads == "text":
        for q in model.image.parameters():
            q.requires_grad_(False)
    elif a.train_heads == "image":
        for q in model.text.parameters():
            q.requires_grad_(False)
    print(f"  trainable parameters: {model.n_params() + 2:,}  "
          f"(heads: {a.train_heads})")

    opt = torch.optim.AdamW(
        [q for q in model.parameters() if q.requires_grad] + list(loss_fn.parameters()),
        lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    _step = {"n": 0}

    def warm() -> None:
        """Linear warmup applied per OPTIMISER STEP, not per epoch."""
        _step["n"] += 1
        if _step["n"] <= a.warmup_steps:
            f = _step["n"] / a.warmup_steps
            for g in opt.param_groups:
                g["lr"] = a.lr * f

    # ---- baseline, measured through the untrained adapter ---------------
    # It is an identity at init, so this reproduces zero-shot SigLIP exactly.
    # Measuring it through the same code path rules out any difference
    # arising from the harness rather than from training.
    if a.dataset == "vidore":
        base = eval_vidore(model, d, d["test"])
        print("\n" + format_table(base, "BEFORE TRAINING (test)"))
        key = lambda r: r["ndcg@10"]["mean"]                      # noqa: E731
        # VAL, not train. Early-stopping on the training queries would
        # happily let the adapter memorise them and report success.
        val_of = lambda: key(eval_vidore(model, d, d["val"]))      # noqa: E731
    else:
        base = eval_flickr(model, d["test_img"], d["test_txt"], d["test_owner"])
        print(f"\n  BEFORE TRAINING (test): {base}")
        key = lambda r: r["recall@10"]                             # noqa: E731
        val_of = lambda: eval_flickr(                              # noqa: E731
            model, d["train_img"], d["val_txt"], d["val_owner"])["recall@10"]

    # ---- training -------------------------------------------------------
    # Record validation BEFORE any update. Without this the first logged
    # value already contains a full epoch of training, and a curve that is
    # merely RECOVERING from early damage is indistinguishable from one that
    # is learning. This exact mistake made a degradation look like a 33%
    # improvement.
    v0 = val_of()
    print(f"\n  epoch  -1  (BEFORE ANY UPDATE)        val {v0:.4f}   <- the real baseline")
    history, best, best_state, bad = [{"epoch": -1, "train_loss": None,
                                       "val": float(v0)}], v0, None, 0
    t_start = time.perf_counter()

    if a.dataset == "vidore":
        train_rows = np.array([d["row"][q] for q in d["train"]])
        Q = torch.tensor(d["q_emb"], device=dev)
        PAGES = torch.tensor(d["pages"], device=dev)
        POS = torch.tensor(d["P"], device=dev)
    else:
        Tt = torch.tensor(d["train_txt"], device=dev)
        Ti = torch.tensor(d["train_img"], device=dev)
        OW = torch.tensor(d["train_owner"].astype(np.int64), device=dev)

    for epoch in range(a.epochs):
        model.train()
        losses = []
        if a.dataset == "vidore":
            perm = np.random.permutation(train_rows)
            for s in range(0, len(perm), a.batch_size):
                idx = torch.tensor(perm[s:s + a.batch_size], device=dev)
                # The two heads are applied by ROLE, not by argument order:
                # queries go through the TEXT head, pages through the IMAGE
                # head. Calling model(...) would pair them the wrong way.
                q_out = model.text(Q[idx])
                p_out = model.image(PAGES)
                l = loss_fn(p_out, q_out, POS[idx])
                opt.zero_grad(); l.backward(); warm(); opt.step()
                losses.append(l.item())
        else:
            perm = torch.randperm(len(Tt), device=dev)
            for s in range(0, len(perm), a.batch_size):
                idx = perm[s:s + a.batch_size]
                owners = OW[idx]
                img_out = model.image(Ti[owners])
                txt_out = model.text(Tt[idx])
                # two captions of the same image in one batch are both correct
                pos = owners[:, None] == owners[None, :]
                l = loss_fn(img_out, txt_out, pos)
                opt.zero_grad(); l.backward(); warm(); opt.step()
                losses.append(l.item())
        sched.step()

        v = val_of()
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val": float(v),
                        "temperature": float(loss_fn.logit_scale.detach().exp()),
                        "bias": float(loss_fn.logit_bias.detach())})
        flag = ""
        if v > best:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            flag = "  <- best"
        else:
            bad += 1
        if epoch % 5 == 0 or flag or bad >= a.patience:
            print(f"  epoch {epoch:3d}  loss {np.mean(losses):7.4f}  "
                  f"val {v:.4f}{flag}")
        if bad >= a.patience:
            print(f"  early stop: {a.patience} epochs without improvement")
            break

    print(f"  trained in {time.perf_counter()-t_start:.1f}s")
    if best_state is None:
        print("  !! NO epoch ever beat the pre-training baseline. Training hurt.")
    else:
        model.load_state_dict(best_state)

    # ---- final, on TEST -------------------------------------------------
    if a.dataset == "vidore":
        after = eval_vidore(model, d, d["test"])
        print("\n" + format_table(after, "AFTER TRAINING (test)"))
        b, f = base["ndcg@10"]["mean"], after["ndcg@10"]["mean"]
    else:
        after = eval_flickr(model, d["test_img"], d["test_txt"], d["test_owner"])
        print(f"\n  AFTER TRAINING (test): {after}")
        b, f = base["recall@10"], after["recall@10"]

    print(f"\n  {'=' * 52}")
    print(f"  before {b:.4f}  ->  after {f:.4f}   "
          f"absolute {f-b:+.4f}   relative {(f-b)/max(b,1e-9)*100:+.1f}%")
    print(f"  {'=' * 52}")
    print("  A single before/after is not significance. Run")
    print("  scripts/compare_runs.py next for the paired test.")

    out = ROOT / "results" / f"adapter_{a.dataset}.json"
    out.write_text(json.dumps({
        "tag": f"adapter_{a.dataset}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": vars(a), "device": dev,
        "trainable_params": model.n_params() + 2,
        "before": base, "after": after, "history": history,
        "best_val": best,
    }, indent=2, default=float))
    torch.save(model.state_dict(), ROOT / "data" / f"adapter_{a.dataset}.pt")
    print(f"\n  saved -> results/{out.name}  and  data/adapter_{a.dataset}.pt")


if __name__ == "__main__":
    main()

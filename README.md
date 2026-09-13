# ContextFuse

**Local-first multimodal retrieval over personal visual memory — screenshots, slides, documents.**

Search your own images in natural language. ContextFuse combines a vision-language
transformer, OCR, and lexical retrieval, then reranks with a network trained on
preference pairs. Everything runs on your machine; no image or embedding leaves it.

```
$ python scripts/search.py "postgres connection refused"

  1. IMG_3467.PNG
     score 0.844   visual 0.61   text 1.00
     why: exact text match: 'postgres', 'connection'  |  source: screenshot  |  2026-03-19
     text: FATAL: password authentication failed for user "postgres"
```

---

## Results

Measured on **107 held-out queries** from ViDoRe V3 Computer Science (1,360 document
pages, graded relevance). Intervals are 95% percentile bootstrap over queries.

| Configuration | nDCG@10 | Recall@10 | MRR@10 |
|---|---:|---:|---:|
| SigLIP 2 only (dense) | 0.078 `[0.049, 0.112]` | 0.070 | 0.168 |
| BM25 over OCR text | 0.630 `[0.576, 0.685]` | 0.633 | 0.803 |
| Fixed-α fusion | 0.637 `[0.584, 0.691]` | 0.640 | 0.817 |
| **Learned reranker** | **0.640** `[0.587, 0.693]` | **0.643** | 0.807 |

**Learned reranker vs BM25: +0.0094 nDCG@10, 95% CI [+0.0004, +0.0191], p = 0.041**
(paired bootstrap, 10,000 resamples). The effect is small and established; it is
reported as both.

### Two findings worth the report

**Lexical retrieval beats dense visual retrieval by 8.1× on document images.**
Pages are rendered at 1700×2200; SigLIP 2 ingests 224×224. Body text does not
survive that reduction, so the encoder matches page *layout* rather than page
*content*. Mean pairwise cosine between page embeddings is **0.795** — it can
barely tell the pages apart. This is the measured reason the architecture keeps
OCR and lexical search rather than replacing them with a vision model.

**Adapter fine-tuning on frozen embeddings did not work, across ~40 configurations
on two datasets.** Degradation was invariant to a 30× learning-rate range and to a
1000× difference in training-set size, which localises the cause to the objective
rather than to optimisation or data volume. Reported as a negative result; changing
the encoder itself remains untested.

---

## Architecture

```
                    ┌─ OCR (Apple Vision) ──→ BM25 inverted index ──┐
  image files ──────┤                                                ├──→ candidates
                    └─ SigLIP 2 (ViT-B/16) ──→ 768-d unit vectors ──┘      (top-100
                                                                            from each)
                                                        │
                                                        ▼
                            17 within-query features (scores, ranks, IDF stats,
                                    term coverage, length normalisation)
                                                        │
                                                        ▼
                              RankNet reranker · 433 parameters · PyTorch
                                                        │
                                                        ▼
                        ranked results + per-signal explanation + dedup
```

Embeddings are L2-normalised at write time, so cosine similarity reduces to a dot
product and search is a single matrix multiply. At N = 1,360 an approximate index
would add complexity no measurement justifies — exact search runs at **p95 = 0.045 ms**.

---

## Quickstart

```bash
git clone <this repo> && cd contextfuse
./scripts/setup.sh                  # venv, deps, prints GPU availability
source .venv/bin/activate

# index your own images
pip install ocrmac pillow-heif      # Apple Vision OCR + HEIC support (macOS)
python scripts/build_personal_index.py ~/Pictures/Screenshots

# search from the CLI
python scripts/search.py "the slide about CNNs"

# or the web UI
./run_ui.sh                         # http://127.0.0.1:8000
```

Reproduce the benchmark results:

```bash
python scripts/fetch_datasets.py    # downloads + verifies, fails loudly
python scripts/run_baseline.py      # config B — SigLIP only
python scripts/run_bm25.py          # config A — BM25 over OCR
python scripts/run_hybrid.py        # config D — fusion, α tuned on dev
python scripts/train_reranker.py    # config F — learned reranker + ablation
```

Every script writes a JSON record to `results/` including model version, device,
seed, git SHA and platform.

---

## Evaluation protocol

The benchmark has a property that is easy to miss and invalidates naive splits.

**Its 1,290 queries are 213 information needs translated into six languages**, and
the six share identical relevance judgements. Splitting by query row puts a question
in training and its French translation in test. All splits here are **by information
need**, and significance tests bootstrap over needs rather than query rows — treating
six translations as six independent observations would make confidence intervals
about √6 too narrow.

Other protocol commitments:

- **Selection never touches test.** Fusion weight α tuned on dev; reranker
  hyperparameters and feature subset chosen by 5-fold cross-validation over dev.
- **Cross-validation replaced a single validation split** that proved too small to
  measure with — 26 queries ranked a configuration first that scored *below* the
  untrained baseline on test, and validation/test correlation across 36
  configurations was near zero.
- **A test-set improvement was found and rejected.** Feature ablation showed
  removing rank features raised test nDCG@10 to 0.648. Cross-validation on dev did
  not support it, so the full feature set was kept and 0.640 stands.
- **Judgement incompleteness is stated.** ViDoRe judges a median of 4 pages per
  query out of 1,360; unjudged pages are scored irrelevant. Recall is recall over
  *known* relevant pages.

---

## Layout

```
src/contextfuse/
  embed.py        SigLIP 2 encoder, L2 normalisation, batched inference
  lexical.py      BM25 from the formula + inverted index
  fusion.py       score normalisation, weighted fusion, RRF
  adaptive.py     per-query fusion weight from corpus IDF statistics
  features.py     17 within-query learning-to-rank features
  reranker.py     RankNet pairwise objective
  losses.py       SigLIP sigmoid contrastive loss; CLIP InfoNCE for comparison
  adapter.py      residual bottleneck adapter (identity at initialisation)
  metrics.py      Recall@k, P@k, MRR, nDCG@k, bootstrap CIs
  stats.py        paired bootstrap significance testing
  ocr.py          Apple Vision / Tesseract with explicit fallback
  ingest.py       scanning, perceptual dedup, privacy filtering
  api.py          FastAPI service
scripts/          one runnable experiment or utility each
tests/            30 tests — metrics, losses, ingest, adaptive weighting, stats
```

~5,200 lines of Python.

---

## Design notes

**Why SigLIP 2 over CLIP.** Its sigmoid objective treats each image-text pair as an
independent binary decision rather than a softmax over the batch, so it does not
depend on very large batches — which matters on a single laptop. It is also
multilingual, which the benchmark's six languages require.

**Why normalise embeddings.** Cosine already divides by the norms, so pre-normalising
is mathematically redundant — but it turns cosine into a plain dot product (one
matmul over the corpus), prevents high-magnitude vectors from scoring well against
everything, and bounds scores to [-1, 1] so they can be fused with BM25 at all.

**Why scores cannot simply be added.** BM25 is unbounded and scales with query
length and term rarity; cosine here clusters tightly around 0.11. Summing them makes
the dense term arithmetically invisible. Both are min-max normalised within each
query before fusion.

**Why a pairwise loss for reranking.** Ranking depends only on order. A model scoring
every relevant document 0.51 and every irrelevant one 0.49 is a poor classifier and a
perfect ranker, so pointwise classification optimises the wrong objective. RankNet's
loss depends only on score *differences*, formed within a query.

**Why per-query adaptive fusion.** A fixed weight cannot serve both `"vellore"` — a
rare term appearing verbatim in four images — and `"logical reasoning question"`,
which describes what a picture looks like. The weight is set from the document
frequency of the rarest query term that actually occurs in the corpus, so
corpus-common words like "question" in a folder of Q&A screenshots are discounted
the same way stopwords are.

---

## Limitations

- All 1,360 benchmark pages come from two OpenStax textbooks rendered identically.
  Results may not transfer to visually heterogeneous corpora.
- The established gain over BM25 is small (+0.0094); 53 of 107 queries are unchanged
  by reranking.
- 630 of the benchmark's 1,290 queries are machine-generated. BM25 scored *higher*
  on human queries (0.671 vs 0.579), so there is no evidence of leakage — but the
  provenance is disclosed.
- The reranker reorders only what BM25 and dense retrieval already found; its recall
  ceiling is theirs.

## Data

| Dataset | Use | Licence |
|---|---|---|
| [ViDoRe V3 Computer Science](https://huggingface.co/datasets/vidore/vidore_v3_computer_science) | retrieval benchmark | CC BY 4.0 |
| [IIIT 5K-Word](https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset) | OCR evaluation | none stated; citation requested |
| [MJSynth](https://www.robots.ox.ac.uk/~vgg/data/text/) | OCR training | see source |
| [Flickr30k](https://huggingface.co/datasets/lmms-lab/flickr30k) | fine-tuning experiments | see source |

No dataset or personal image is committed to this repository.
Model: [`google/siglip2-base-patch16-224`](https://huggingface.co/google/siglip2-base-patch16-224) (Apache-2.0).

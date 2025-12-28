# RAG Reliability Lab

A small, end-to-end lab for **retrieval → evidence selection → grounded answering → reliability evaluation**.

This repo is deliberately opinionated: it optimizes for **measurable reliability** (correct + grounded + citation-valid) over “fun demos”.

---

## Key Results (Headline)

### Reliability patch (Rep slice, n=120; judge: `gpt-5-mini`)
**Tiered Policy V2** improved answer quality while maintaining high coverage:

- **Correctness:** 0.52 → **0.73**
- **Grounded rate:** 0.72 → **0.99**
- **Citation-valid rate:** 0.77 → **1.00**
- **Answered coverage:** 0.50 → **0.74** (Tier C abstains)

**So what:** V2 largely eliminates “confident but unsupported” answers by enforcing **cite-or-abstain**, producing near-perfect groundedness/citation validity at high coverage.

### Retrieval (Full validation, n=3972)
Hybrid retrieval provides the best candidate pool:

- **Hybrid RRF (BM25 + OpenAI dense), k=30:** recall@30 **0.7059**, MRR **0.3822**, nDCG@30 **0.4519**

### Local reranking (Rep slice, n=120; $0 credits)
Local cross-encoder reranking improves **top-context ordering** in the window RAG actually uses:

- Candidate pool: Hybrid RRF (top-30)
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (GPU, no API calls)
- **At k=5:** recall@5 **0.50 → 0.50** (unchanged), **MRR +0.039**, **nDCG@5 +0.030**

---

## Core contribution (narrative arc)

1) **Hybrid retrieval** builds a **high-recall** candidate pool (coverage).
2) **Reranking** improves **top-k ordering** (quality of the contexts you actually feed to the LLM).
3) **Tiered policy + calibration** makes the system **fail gracefully** (answer when confident; abstain when not), yielding large reliability gains.

---

## Architecture (ASCII)

```
Query
  └─> Retrieve (BM25 + Dense)  ──────────────┐
         └─> Fuse (RRF) → Candidate Pool k=30├─> (optional) Rerank (CE / LLM) → Top-k (k=5..10)
                                             └─> Tier Policy (A/B/C) → Answer or Abstain
                                                      └─> Judge + Calibration → Metrics + Failure reports
```

Tier routing (conceptual):

```
m = score_top1 - score_top2

if m >= 1.0:
  Tier A: strict grounded answer + citations
elif score_top1 >= 3.0:
  Tier B: evidence-first; cite-or-abstain
else:
  Tier C: abstain (show top passages)
```

---

## Repository conventions

- CLI: `raglab ...` (Typer)
- Runs: `runs/<timestamp>__<run_name>/`
  - `config.json`, `metrics.json`, `predictions.jsonl`
  - plus stage-specific outputs (e.g., `answers.jsonl`, `calibration.json`, `failures.csv`)

---

## Data + evaluation

### Data products
- `chunks.jsonl`: chunked document corpus (`chunk_id`, `text`, …)
- `grounding_map_<split>_chunks.jsonl`: (`qid`, `question`, `gold_chunk_ids`, …)

### Retrieval success criterion
A retrieval is “correct” if any retrieved chunk overlaps the gold span(s) mapped to chunks (via `gold_chunk_ids`).

---

## Results

### Retrieval comparison (full validation)

| Retriever | Split | k | Recall@k | MRR | nDCG@k |
|---|---:|---:|---:|---:|---:|
| Hybrid RRF (BM25 + OpenAI dense) | validation | 30 | **0.7059** | **0.3822** | **0.4519** |
| Local Dense Torch (MiniLM bi-encoder) | validation | 30 | 0.5687 | 0.2668 | 0.3300 |

> Interpretation: MiniLM bi-encoder is not competitive as a primary retriever here; hybrid retrieval remains the best candidate pool.

### Local dense retrieval (PyTorch, exact cosine; no FAISS)

We implemented a local dense retriever using SentenceTransformers on GPU:
- embed chunks once (cached to disk)
- embed queries in batches
- L2 normalize and compute cosine via `matmul`
- top-k via `torch.topk`

**Model:** `sentence-transformers/all-MiniLM-L6-v2` (dim=384)

- Rep slice (n=120, k=30): recall@30 **0.6083**, MRR **0.3264**, nDCG@30 **0.3831**
- Full validation (n=3972, k=30): recall@30 **0.5687**, MRR **0.2668**, nDCG@30 **0.3300**

### Reranking comparison (rep slice; candidate pool fixed)

Reranking does **not** change candidate coverage; it changes ordering.

| Candidate Pool | Reranker | k | Recall@k | MRR | nDCG@k |
|---|---|---:|---:|---:|---:|
| Hybrid RRF (top-30) | none | 5 | 0.5000 | 0.3816 | 0.3950 |
| Hybrid RRF (top-30) | **Local cross-encoder** | 5 | 0.5000 | **0.4204** | **0.4248** |

At k=10 (rep slice), reranking improved ordering (MRR/nDCG) but slightly reduced recall@10 (coverage in top-10):
- Baseline: recall@10 0.5917, MRR 0.3816, nDCG@10 0.4224  
- Reranked: recall@10 0.5750, MRR 0.4204, nDCG@10 0.4441

> Practical takeaway: cross-encoder reranking improves “how early the first relevant chunk appears” (MRR) in the top-k window; it does not add new candidates.

Why recall@10 can drop: the cross-encoder sometimes **penalizes a borderline-but-relevant chunk** that the retriever ranked highly, while strongly promoting the best evidence upward—so ordering improves even if a few relevant items fall just outside the cutoff.

---

## Tiered policy and reliability (rep slice, n=120)

### Tiered Policy V2 summary (judge: `gpt-5-mini`)
| Version | Answered Coverage | Correctness (↑) | Grounded Rate (↑) | Citation Valid Rate (↑) |
|---|---:|---:|---:|---:|
| V1 | 0.50 | 0.52 | 0.72 | 0.77 |
| **V2** | **0.74** | **0.73** | **0.99** | **1.00** |

Tier metrics (answered-only within tier):
| Tier | n | Answered Coverage | Correctness | Grounded | Citation Valid |
|---|---:|---:|---:|---:|---:|
| A_high | 50 | 0.92 | 0.73 | 0.98 | 1.00 |
| B_medium | 45 | 0.96 | 0.73 | 1.00 | 1.00 |
| C_low | 25 | 0.00 | — | — | — |

---

## Setup

### Environments
- **Default env (`.venv`)**: core pipeline + OpenAI retrieval/rerank/judge
- **Torch env (`.venv-torch`)**: GPU inference for local dense + cross-encoder reranking  
  - Python **3.11**
  - Torch **2.1.2+cu118** (CUDA 11.8)
  - SentenceTransformers **2.2.2**

Why no FAISS: corpus is small (~1506 chunks), and FAISS GPU wheels for cp311 can be painful; exact cosine is sufficient.

Tip: use `.venv-torch` specifically for **GPU-bound local inference** (local dense retrieval, cross-encoder reranking), and keep `.venv` for the main pipeline and any OpenAI-dependent stages to avoid dependency conflicts.

### Install
```bash
python -m venv .venv
# activate .venv
pip install -e .
```

---

## Reproduce (high level)

### Build processed data
```bash
python scripts/00_download_doc2dial.py
python scripts/01_ingest_doc2dial.py
python scripts/02_build_grounding_map.py
python scripts/03_chunk_docs.py
python scripts/04_map_grounding_to_chunks.py
```

### Retrieval
```bash
raglab run-bm25 --run-name bm25_validation_k30 --split validation --k 30 --tokenizer a3
raglab run-dense --run-name dense_openai_validation_k30 --split validation --k 30 --dense-model text-embedding-3-small
raglab run-hybrid-rrf --run-name hybrid_rrf_validation_k30 --split validation --k 30 --dense-model text-embedding-3-small
```

### Local dense retrieval (Torch)
Run in `.venv-torch`:
```bash
raglab run-local-dense-torch --run-name local_dense_torch_val --split validation --k 30
```

### Local cross-encoder reranking (Torch; no credits)
Rerank an existing candidate pool (e.g., hybrid RRF rep slice) and evaluate at k=5:
```bash
raglab run-rerank-cross-encoder-local \
  --run-name xenc_on_hybrid_rrf_rep_eval5 \
  --input-run-dir runs/<timestamp>__hybrid_rrf_rep_k30 \
  --split validation \
  --top-n 30 \
  --k-eval 5 \
  --use-samples
```

Evaluate any existing `predictions.jsonl` at a different cutoff:
```bash
raglab eval-predictions \
  --run-name eval_only_k10 \
  --predictions-path runs/<timestamp>__hybrid_rrf_rep_k30/predictions.jsonl \
  --split validation \
  --k 10 \
  --use-samples
```

---

## Threats to validity + audit

- **LLM-as-judge bias:** Judge quality can vary by model/prompt; we mitigate with deterministic rep slice and storing all artifacts.
- **No inter-annotator agreement yet:** recommended next step is a small human audit on ~20 examples and compute agreement (e.g., **Cohen’s κ**) between judge vs human labels for **correctness / groundedness / citation-valid**.
- **Rep slice size (n=120):** useful for iteration and ablation; full validation generation is possible but costly.

---

## Roadmap (bounded)
- Try retrieval-optimized bi-encoders (E5/BGE) with correct query/passage formatting
- Cross-encoder model sweep (quality vs latency)
- Distillation: (teacher reranker) → (student bi-encoder) to improve local dense retrieval
- Learned confidence model (beyond margin) + stronger calibration reports

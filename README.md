# RAG Reliability Lab

A small, end-to-end lab for **retrieval + grounded generation + reliability evaluation**.
Focus: measurable improvements via **retrieval baselines**, **hybrid retrieval**, **LLM reranking**, **selective answering**, and **LLM-as-judge** evaluation.

## Why this project
Most RAG demos stop at “it works.” This project focuses on:
- **Retriever quality** (recall/MRR/nDCG)
- **Answer reliability** (correctness, groundedness, citation validity)
- **Calibration + abstention** (coverage vs reliability tradeoffs)
- **Actionable failure analysis** (tiering, recoverability)

---

## Data
- Corpus and dialogues: Doc2Dial (ingested locally)
- We build:
  - `docs.jsonl` and `dialogues_{split}.jsonl`
  - `grounding_map_{split}.jsonl`: maps each query to the gold document spans
  - `chunks.jsonl`: chunked corpus for retrieval

### Grounding
We treat “correct retrieval” as retrieving a chunk that overlaps the **gold span(s)** associated with the query (not just “something vaguely relevant”).

---

## System overview

### Pipeline stages
1) **Ingest → Grounding map → Chunking**
2) **Retrieval**: BM25 baseline, dense embeddings, hybrid fusion (RRF)
3) **(Optional) Rerank**: LLM reranking on a representative slice
4) **RAG generation**: answer + citations + abstain if unsupported
5) **LLM-as-judge evaluation**
6) **Calibration + tiered policy**: improve reliability via selective answering

---

## Retrieval results (full validation, n=3972)
**Hybrid retrieval (BM25 + Dense via RRF)** shows strong recall and ranking quality:

- k=30: recall@30 **0.706**, MRR **0.382**, nDCG@30 **0.452**

This provides a high-recall candidate set for downstream reranking and generation.

---

## End-to-end RAG reliability (Rep Slice, n=120)

We evaluate answer behavior with an LLM judge:
- **Correctness** (0–1)
- **Groundedness** (0/1)
- **Citation validity** (0/1)
- **Answered coverage** (fraction not abstained)

### Tiered policy (V1 → V2 Reliability Patch)
We route queries based on reranker confidence:

- **Tier A (high confidence):** margin(top1 - top2) ≥ 1 → strict grounded answer
- **Tier B (medium confidence):** top1 ≥ 3 and margin < 1 → evidence-first answer (**cite-or-abstain enforced**)
- **Tier C (low confidence):** abstain + show top passages (no LLM call)

#### Overall (answered-only; weighted across tiers)
| Version | Answered Coverage | Correctness (↑) | Grounded Rate (↑) | Citation Valid Rate (↑) |
|---|---:|---:|---:|---:|
| V1 | 0.50 | 0.52 | 0.72 | 0.77 |
| V2 (gpt-5-mini) | 0.74 | 0.73 | 0.99 | 1.00 |

#### V2 Tier Metrics (answered-only within tier)
| Tier | n | Answered Coverage | Correctness | Grounded | Citation Valid |
|---|---:|---:|---:|---:|---:|
| A_high | 50 | 0.92 | 0.73 | 0.98 | 1.00 |
| B_medium | 45 | 0.96 | 0.73 | 1.00 | 1.00 |
| C_low | 25 | 0.00 | — | — | — |

---

## How to reproduce

### Setup
```bash
python -m venv .venv
# activate venv
pip install -e .
```

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
raglab run-bm25 --run-name bm25_a3_prod --split validation --k 10 --tokenizer a3
raglab run-dense --run-name dense_prod --split validation --k 10 --dense-model text-embedding-3-small
raglab run-hybrid-rrf --run-name hybrid_rrf_full_k30 --split validation --k 30 --dense-model text-embedding-3-small
```

### RAG + judge + tier analysis (rep slice)
```bash
raglab run-rag --run-name rag_gen_rep_tiered_v2_q120 --use-samples --policy tiered_v1 --model gpt-5-mini ...
raglab judge-rag --run-name rag_judge_rep_tiered_v2_q120 --use-samples --judge-model gpt-5-mini ...
raglab analyze-tiers --run-name tiers_rep_tiered_v2_q120 --rag-gen-dir <...> --judge-dir <...>
```

---

## Limitations
- Reranking + judging uses an LLM, so scores reflect judge behavior (we mitigate with deterministic slices + reproducible runs).
- Rep slice is small (n=120); full validation generation is possible but costly.
- Current dense embeddings use hosted APIs; a local PyTorch embedding baseline is planned.

---

## Extension roadmap
- Local embeddings + ANN index (PyTorch + FAISS) to compare cost/latency/quality vs hosted embeddings
- Better confidence models (e.g., margin + entropy, or learned calibrator)
- Richer failure taxonomy and auto-generated qualitative reports

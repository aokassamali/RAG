# Design Decisions

This document records key choices, tradeoffs considered, and why we made them.

## 1) Document unit + grounding definition
**Decision:** treat gold as span-level evidence from Doc2Dial; retrieval is “correct” if a retrieved chunk overlaps gold span(s).  
**Why:** prevents “lucky retrieval” of vaguely relevant text; aligns retrieval scoring with evidence needed for generation.

Alternatives:
- document-level gold (too coarse; inflates recall)
- answer-string match (fragile; requires clean answers)

## 2) Chunking strategy (300–400 tokens, ~50 overlap)
**Decision:** chunk docs into ~300–400 tokens with ~50 token overlap.  
**Why:** balances:
- enough context for semantic coherence
- manageable granularity for retrieval
- overlap reduces boundary-splitting where key evidence straddles chunks

Tradeoff: overlap can inflate recall if too large; kept modest.

## 3) Retriever ladder: BM25 → Dense → Hybrid (RRF)
**Decision:** establish BM25 baseline; add dense embeddings; fuse with Reciprocal Rank Fusion.  
**Why:**
- BM25 captures rare keyword matches and exact phrasing
- dense captures semantic paraphrases
- hybrid improves recall/robustness with minimal complexity

## 4) Hybrid operating point (k=30)
**Decision:** use k=30 as candidate set for downstream steps.  
**Why:** higher recall candidate pool for reranking/generation; still computationally manageable.

## 5) Confidence proxy: reranker margin > top1 score
**Decision:** prefer margin(top1-top2) as high-confidence signal.  
**Why:** separation between best and runner-up correlates with unambiguous evidence; top1 alone is often poorly calibrated.

## 6) Tiered answering policy (A/B/C)
**Decision:** route queries based on confidence:
- **A_high:** margin ≥ 1 → strict grounded answer
- **B_medium:** top1 ≥ 3 & margin < 1 → evidence-first + cite-or-abstain
- **C_low:** abstain + show top passages

**Why:** achieves balanced coverage with high reliability, and provides useful fallback for low-confidence cases without hallucination.

## 7) Evidence-first generation + “no citation → abstain”
**Decision:** in Tier B, enforce short evidence-first answers and abstain if no valid citations returned.  
**Why:** removes unsupported fluent answers; improves groundedness and citation validity.

## 8) Evaluation: LLM-as-judge + calibration
**Decision:** use judge scores for correctness/grounded/citation-valid and evaluate coverage–reliability tradeoffs.  
**Why:** scalable, structured evaluation for end-to-end behavior (retrieval metrics alone are insufficient).

## 9) Local dense retrieval (PyTorch, exact cosine; no FAISS)
**Decision:** implement local dense retrieval using SentenceTransformers on GPU with **exact cosine similarity** (L2 normalize + matmul + topk) and cache chunk embeddings to disk; avoid FAISS.  
**Why:**
- corpus is small (~1506 chunks), so exact search is fast and simple
- avoids FAISS GPU wheel issues on cp311
- caching makes repeated runs cheap (embed corpus once; embed queries per run)

**Outcome:** baseline MiniLM bi-encoder underperformed hosted dense + hybrid on this dataset (lower recall/MRR/nDCG), so we treat it as a reproducible local baseline and a foundation for future improvements (E5/BGE, distillation).

Alternatives:
- FAISS ANN index (blocked by packaging constraints; unnecessary at current corpus size)
- quantized ANN (premature optimization)

---

## 10) Local cross-encoder reranking (no credits)
**Decision:** add an inference-only local cross-encoder reranker on top of the hybrid candidate pool.  
**Why:**
- improves **top-context ordering** where RAG is most sensitive (top 3–5 chunks)
- runs with $0 API cost and provides a reproducible offline dev loop
- clean separation of concerns: retrieval for coverage; rerank for ordering

**Measured effect (rep slice):**
- recall@5 unchanged while **MRR and nDCG@5 improved** (better ordering within fixed pool)

Alternatives:
- train/fine-tune a reranker (more scope; save for follow-up work)
- distillation into a bi-encoder (more novel, but larger pipeline and training loop)

---

## Extension roadmap
- Try retrieval-optimized local bi-encoders (E5/BGE) with correct query/passage prefixes
- Distillation: teacher (hosted embeddings or cross-encoder) → student (local bi-encoder)
- Hybrid fusion: BM25 + local dense + RRF (evaluate complementarity)
- Improve confidence models (beyond margin), and add learned calibrators

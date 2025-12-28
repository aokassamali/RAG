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

## Extension: PyTorch local embeddings (planned)
**Goal:** implement local embedding model + ANN index (FAISS) and compare with hosted embeddings on quality/cost/latency.

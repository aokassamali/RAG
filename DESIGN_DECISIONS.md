# Design Decisions (and “why not”)

This document captures the *reasoning* behind choices—especially alternatives we considered and failure modes we tried to avoid.

---

## 1) Gold definition: span-level grounding → chunk-level targets
**Decision:** retrieval is correct if any retrieved chunk overlaps gold span(s) mapped to chunks (`gold_chunk_ids`).  
**Why:** document-level gold inflates recall and hides evidence selection failures.

**Rejected alternatives**
- **Doc-level gold:** too coarse; can “retrieve the doc” but miss the evidence.
- **Answer-string match:** brittle; confounds retrieval with generation phrasing.

**Failure mode avoided:** “looks relevant” retrieval that doesn’t actually support the answer.

---

## 2) Chunking: fixed-window token chunks with overlap
**Decision:** fixed-window token chunker (~300–400 tokens) with ~50-token overlap. To preserve structure, we **prepend the nearest section header (and, if present, parent header)** to each chunk’s text before embedding/indexing.  
**Why:** stable chunk IDs and consistent retrieval behavior; overlap reduces boundary splits.

Header prefixing is a common RAG trick: it injects lightweight global context into local snippets and improves retrieval when questions refer to a section implicitly.

**Rejected alternatives**
- Recursive character chunking: can be good, but introduces non-obvious boundaries and more variance across runs.
- Large chunks: hurts retrieval specificity; more noise in evidence selection.

**Failure mode avoided:** key evidence split across chunk boundaries.

---

## 3) Retriever ladder: BM25 → Dense → Hybrid (RRF)
**Decision:** build a retriever ladder:
- BM25 baseline for lexical exact match
- dense embeddings for semantic match
- RRF fusion for best of both

**Why:** improves recall robustly with minimal complexity.

**Rejected alternatives**
- Learned fusion (train a ranker): more moving parts; harder to debug early.
- “Dense only”: often misses rare terms/entities that BM25 gets.

**Failure mode avoided:** catastrophic misses when a query depends on an exact phrase.

---

## 4) Hybrid fusion method: Reciprocal Rank Fusion (RRF)
**Decision:** use RRF to combine BM25 + dense ranks.

**Formalism**
For document/chunk \(d\):
\[
RRF(d) = \sum_{i \in \{\text{bm25},\text{dense}\}} \frac{1}{k_0 + \text{rank}_i(d)}
\]

**Why:** robust to score scale differences and easy to implement.

**Rejected alternatives**
- Score normalization + weighted sum: brittle because score distributions differ a lot across retrievers.
- Learned weighting: adds training loop and data splits.

**Failure mode avoided:** “dense scores dominate everything” due to scaling.

---

## 5) Confidence proxy: top-1 vs top-2 margin
**Decision:** use margin as the primary confidence proxy.

**Formalism**
Let scores for ranked chunks be \(S_1 \ge S_2 \ge ...\). Define:
\[
m = S_1 - S_2
\]

**Why:** separation is often a better uncertainty signal than absolute top-1 score.

**Rejected alternatives**
- Top-1 score only: poorly calibrated, retriever-specific.
- Entropy over softmax(scores): can be unstable with arbitrary score scales.

**Failure mode avoided:** overconfident answers from “barely better than runner-up” contexts.

---

## 6) Tiered policy: answer when confident; abstain when not
**Decision:** A/B/C tiers:
- **Tier A:** high margin → strict grounded answer
- **Tier B:** medium confidence → evidence-first; cite-or-abstain
- **Tier C:** low confidence → abstain

**Why:** selective answering improves reliability dramatically with only modest coverage loss.

**Rejected alternatives**
- Always answer: maximizes coverage, but hallucination risk is high.
- Always abstain when uncertain without showing passages: unhelpful UX.

**Failure mode avoided:** fluent but unsupported answers (hallucinations).

---

## 7) Evidence-first generation and cite-or-abstain
**Decision:** in medium-confidence cases, force evidence-first outputs and abstain if citations are missing/invalid.

**Why:** “groundedness” is the objective, not fluency.

**Rejected alternatives**
- Allow answer without citations: too easy to hallucinate and still sound plausible.

**Failure mode avoided:** correctness ≈ random under uncertainty while sounding confident.

---

## 8) Evaluation: combine retrieval metrics + end-to-end judge metrics
**Decision:** measure both:
- retrieval: recall@k / MRR / nDCG
- end-to-end: correctness / groundedness / citation validity + coverage

**Why:** retrieval can look good while generation hallucinates; end-to-end can look good while retrieval is hiding misses (small slices).

**Rejected alternatives**
- Retrieval-only evaluation: insufficient for reliability.
- Judge-only evaluation: can hide retrieval regressions.

**Failure mode avoided:** optimizing the wrong layer.

---

## 9) Local dense retrieval in PyTorch (exact cosine; no FAISS)
**Decision:** implement exact cosine retrieval with SentenceTransformers + torch matmul/topk; cache chunk embeddings.  
**Why:** corpus is small (~1506 chunks), exact search is fast; avoids FAISS GPU wheel issues on Python 3.11.

**Formalism**
Embed query \(q\) and chunk \(c\) into vectors \(e_q, e_c\). Cosine similarity:
\[
S(q,c)=\frac{e_q \cdot e_c}{\|e_q\|\|e_c\|}
\]
With L2-normalized embeddings, \(S(q,c)= e_q \cdot e_c\).

**Outcome:** MiniLM bi-encoder underperformed hybrid retrieval; treat it as a reproducible baseline and a platform for E5/BGE or distillation.

**Rejected alternatives**
- ANN index (FAISS): unnecessary at current N; packaging friction.
- CPU-only: slower dev loop on large query sets.

---

## 10) Local cross-encoder reranking (inference-only; $0 credits)
**Decision:** add a local cross-encoder reranker over an existing candidate pool.

**Why:** reranking targets the bottleneck for RAG: the **top few contexts**.

**Outcome (rep slice):** recall@5 unchanged, but MRR/nDCG@5 improved meaningfully—better top-k ordering without changing coverage.

**Rejected alternatives**
- Fine-tuning a reranker: valuable but larger scope (training loop, splits, hyperparams).
- Distillation first: more novel, but best done after establishing a strong teacher and evaluation harness.

**Failure mode avoided:** passing “near miss” passages into the LLM while better evidence exists slightly lower in the pool.

---

## 11) Threats to validity and mitigations
- **Judge bias:** mitigated by deterministic slices and storing artifacts for audit.
- **Small rep slice:** used for iteration; key claims should be revalidated on full validation where possible (retrieval/rerank is cheap; generation is costly).
- **Score scale mismatch:** addressed via RRF (rank-based fusion) and margin-based confidence.

---

## Next upgrades (if expanding scope)
- Swap MiniLM for retrieval-optimized bi-encoders (E5/BGE) with query/passage prefixes
- Distill cross-encoder or hosted embeddings into a local bi-encoder
- Train a lightweight confidence model (features: margin, rank stats, BM25/dense agreement)
- Add a small human audit (n≈20) to quantify judge agreement

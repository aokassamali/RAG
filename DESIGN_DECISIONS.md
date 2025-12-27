Decision: Use BM25Okapi + “safe stopword” tokenizer for lexical baseline (V1)

Context: Need a debuggable retrieval baseline before dense/hybrid methods.

Options tested (validation, K=10):

A1 BM25Okapi + basic tokenizer: Recall@10 0.5365, MRR 0.3468, nDCG@10 0.3913

A3 BM25Okapi + safe stopwords (keep negations/modals): Recall@10 0.5582, MRR 0.3561, nDCG@10 0.4031 (winner)

A4 BM25Plus variants: ~no gain vs Okapi

Decision: Adopt A3 as default lexical baseline.

Why: Best metrics with minimal complexity; improvements come from normalization rather than swapping BM25 variant.

How we’ll revisit: After dense retrieval + reranking, re-run the same eval harness.
# Goal: Build a BM25 retriever over chunks and retrieve top-K for each validation query.
# Why: establishes a strong, debuggable lexical baseline before dense/hybrid retrieval.

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from rank_bm25 import BM25Okapi

REPO_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = REPO_ROOT / "data" / "processed" / "chunks.jsonl"
VAL_GM_PATH = REPO_ROOT / "data" / "processed" / "grounding_map_validation_chunks.jsonl"

RUN_DIR = REPO_ROOT / "runs" / "bm25_v1"
OUT_PRED = RUN_DIR / "val_predictions.jsonl"

TOP_K = 10


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def tokenize(text: str) -> List[str]:
    # Simple tokenization: lowercase whitespace tokens.
    # Later upgrades: stopwords, stemming, better tokenization.
    return [t.lower() for t in text.split() if t.strip()]


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    chunks = read_jsonl(CHUNKS_PATH)
    corpus_tokens = [tokenize(c["text"]) for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    # Predict before running:
    # - Corpus size should be 1506 chunks (based on your earlier output).
    # - BM25 build should be fast (< a few seconds).
    print(f"Loaded {len(chunks)} chunks")

    bm25 = BM25Okapi(corpus_tokens)

    val = read_jsonl(VAL_GM_PATH)
    print(f"Loaded {len(val)} validation queries")

    with OUT_PRED.open("w", encoding="utf-8") as f:
        for ex in val:
            qid = ex["qid"]
            query = ex["question"]
            q_toks = tokenize(query)

            scores = bm25.get_scores(q_toks)  # float score per chunk
            # get top K indices
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K]

            preds = [{"chunk_id": chunk_ids[i], "score": float(scores[i])} for i in top_idx]

            f.write(json.dumps({"qid": qid, "preds": preds}, ensure_ascii=False) + "\n")

    print(f"Wrote predictions -> {OUT_PRED}")


if __name__ == "__main__":
    main()

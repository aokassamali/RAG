# Goal: Provide a production-ish BM25 retriever with a stable interface + optional pickle cache.
# Why: Lets us swap retrievers (dense/hybrid later) while keeping the same runner/eval code.

from __future__ import annotations
import pickle, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Callable, Optional

from rank_bm25 import BM25Okapi

SAFE_STOPWORDS = {
    "the","a","an","and","or","to","of","in","on","for","with","at","by","from",
    "as","it","this","that","these","those","be","is","are","was","were","been",
    "i","you","he","she","they","we","my","your","their","our","me","him","her",
    "do","does","did","done","can","could","would","will","just","about","into",
    "than","then","there","here","also","if","when","where","which","who","whom"
}
KEEP_WORDS = {"no","not","never","without","must","should","may","cannot","can't","won't","n't"}

def tok_basic(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

def tok_a3_safe_stopwords(text: str) -> List[str]:
    toks = tok_basic(text)
    return [t for t in toks if (t in KEEP_WORDS) or (t not in SAFE_STOPWORDS)]

@dataclass
class BM25Retriever:
    bm25: Any
    chunk_ids: List[str]
    tokenizer: Callable[[str], List[str]]

    @staticmethod
    def build(chunks: List[Dict[str, Any]], tokenizer: Callable[[str], List[str]]) -> "BM25Retriever":
        corpus_tokens = [tokenizer(c["text"]) for c in chunks]
        chunk_ids = [c["chunk_id"] for c in chunks]
        bm25 = BM25Okapi(corpus_tokens)
        return BM25Retriever(bm25=bm25, chunk_ids=chunk_ids, tokenizer=tokenizer)

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, float]]:
        q = self.tokenizer(query)
        scores = self.bm25.get_scores(q)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{"chunk_id": self.chunk_ids[i], "score": float(scores[i])} for i in top_idx]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "BM25Retriever":
        path = Path(path)
        with path.open("rb") as f:
            return pickle.load(f)

# Goal: Dense retrieval via OpenAI embeddings: embed chunks once, embed queries on demand, retrieve by cosine similarity.
# Why: Captures semantic similarity (paraphrase/synonyms) where BM25 fails.

from __future__ import annotations
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI

def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom

@dataclass
class DenseIndex:
    chunk_ids: List[str]
    embeddings: np.ndarray  # shape (N, D), normalized

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "DenseIndex":
        with Path(path).open("rb") as f:
            return pickle.load(f)

@dataclass
class OpenAIDenseRetriever:
    client: Any
    model: str
    index: DenseIndex

    @staticmethod
    def build(
        chunks: List[Dict[str, Any]],
        model: str,
        api_key: Optional[str] = None,
        batch_size: int = 128,
    ) -> "OpenAIDenseRetriever":
        # Predict: This will call the embeddings API ~ceil(N/batch_size) times.
        # With N=1506 chunks, that's ~12 calls at batch_size=128.
        client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

        texts = [c["text"] for c in chunks]
        chunk_ids = [c["chunk_id"] for c in chunks]

        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            resp = client.embeddings.create(model=model, input=batch)
            # response order matches input order
            embs.extend([d.embedding for d in resp.data])

        E = np.array(embs, dtype=np.float32)
        E = l2_normalize(E)

        idx = DenseIndex(chunk_ids=chunk_ids, embeddings=E)
        return OpenAIDenseRetriever(client=client, model=model, index=idx)

    def embed_query(self, query: str) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=query)
        q = np.array(resp.data[0].embedding, dtype=np.float32)[None, :]
        q = l2_normalize(q)
        return q[0]

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, float]]:
        q = self.embed_query(query)  # (D,)
        scores = self.index.embeddings @ q  # cosine since normalized
        top_idx = np.argsort(-scores)[:top_k]
        return [{"chunk_id": self.index.chunk_ids[i], "score": float(scores[i])} for i in top_idx]

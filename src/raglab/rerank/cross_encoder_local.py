# What this module does and why:
# We implement a local cross-encoder reranker (no API calls) to rescore (query, passage) pairs.
# Cross-encoders are strong at reranking because they attend over query+passage jointly.
# This improves ordering within an existing candidate pool, which helps RAG when you only use top 5–10 chunks.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

try:
    from sentence_transformers import CrossEncoder
except Exception as e:
    raise ImportError(
        "sentence-transformers is required for CrossEncoder reranking. "
        "Install it in your .venv-torch environment."
    ) from e


@dataclass(frozen=True)
class CrossEncoderLocalConfig:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cuda"  # "cuda" or "cpu"
    batch_size: int = 64
    max_chars_per_passage: int = 900  # truncate for speed


class CrossEncoderLocalReranker:
    def __init__(self, cfg: CrossEncoderLocalConfig):
        self.cfg = cfg
        self.model = CrossEncoder(cfg.model_name, device=cfg.device)

    def rerank(self, question: str, candidates: List[Tuple[str, str]]) -> List[Tuple[str, float]]:
        """
        candidates: list of (chunk_id, chunk_text)
        returns: list of (chunk_id, new_score) sorted desc
        """
        clipped = []
        ids = []
        for cid, txt in candidates:
            ids.append(cid)
            clipped.append(txt[: self.cfg.max_chars_per_passage])

        pairs = [(question, t) for t in clipped]

        # Predict before running:
        # - returns one float score per (question, text) pair (higher = more relevant)
        scores = self.model.predict(pairs, batch_size=self.cfg.batch_size, show_progress_bar=False)
        out = [(cid, float(sc)) for cid, sc in zip(ids, scores)]
        out.sort(key=lambda x: x[1], reverse=True)
        return out

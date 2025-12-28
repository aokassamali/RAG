# Goal: Local dense retrieval using SentenceTransformers on GPU + exact cosine top-k (no FAISS).
# Why: Corpus is small (~1506 chunks), so exact matmul+topk is fast, simple, and avoids faiss wheels.

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise ImportError(
        "sentence-transformers is required for local_dense_torch. "
        "Install it in your .venv-torch environment."
    ) from e


@dataclass(frozen=True)
class TorchDenseConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cuda"  # "cuda" or "cpu"
    chunk_batch_size: int = 128
    query_batch_size: int = 128
    cache_dtype: str = "float16"  # "float16" or "float32"
    normalize: bool = True
    cache_root: str = "artifacts/local_dense_torch"


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _fingerprint_chunks(chunks: List[Dict[str, Any]], *, id_key: str, text_key: str) -> str:
    """
    Fingerprint chunk ids + chunk texts so cache invalidates when chunking changes.
    We hash a compact representation: [(chunk_id, len(text), sha256(text)), ...] in order.
    """
    rows: List[List[Any]] = []
    for c in chunks:
        cid = str(c[id_key])
        txt = str(c[text_key])
        rows.append([cid, len(txt), _sha256_text(txt)])
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return _sha256_text(payload)[:16]


def _safe_model_slug(model_name: str) -> str:
    return model_name.replace("/", "__").replace(":", "_")


class LocalTorchDenseRetriever:
    """
    Local dense retriever with:
      - cached chunk embeddings on disk
      - batched query embedding
      - exact cosine similarity via matmul and torch.topk

    Output format matches your other retrievers:
      [{"chunk_id": <str>, "score": <float>}, ...] in descending score order
    """

    def __init__(self, cfg: TorchDenseConfig):
        self.cfg = cfg

        # Pick device robustly.
        if cfg.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = cfg.device

        self.model = SentenceTransformer(cfg.model_name, device=self.device)

    def _cache_dir(self, fingerprint: str) -> Path:
        # artifacts/local_dense_torch/<model_slug>/<fingerprint>/
        base = Path(self.cfg.cache_root) / _safe_model_slug(self.cfg.model_name) / fingerprint
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _cache_paths(self, fingerprint: str) -> Dict[str, Path]:
        base = self._cache_dir(fingerprint)
        return {
            "base": base,
            "meta": base / "meta.json",
            "chunk_ids": base / "chunk_ids.json",
            "emb": base / "chunk_emb.pt",
        }

    @torch.inference_mode()
    def _encode_texts(self, texts: List[str], batch_size: int) -> torch.Tensor:
        """
        Encodes texts to a torch tensor on self.device.
        We normalize here so cosine = dot product.
        """
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb)

        emb = emb.to(self.device)
        if self.cfg.normalize:
            emb = F.normalize(emb, p=2, dim=1)
        return emb

    @torch.inference_mode()
    def build_or_load_index(
        self,
        chunks: List[Dict[str, Any]],
        *,
        id_key: str = "chunk_id",
        text_key: str = "text",
        force_rebuild: bool = False,
    ) -> Tuple[List[str], torch.Tensor, Dict[str, Any]]:
        """
        Returns:
          chunk_ids: List[str]
          chunk_emb: torch.Tensor [N, D] on self.device (normalized if cfg.normalize)
          meta: dict
        """
        fp = _fingerprint_chunks(chunks, id_key=id_key, text_key=text_key)
        paths = self._cache_paths(fp)

        if (not force_rebuild) and paths["meta"].exists() and paths["chunk_ids"].exists() and paths["emb"].exists():
            meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
            chunk_ids = json.loads(paths["chunk_ids"].read_text(encoding="utf-8"))

            # Load embeddings (stored CPU) then move to device
            emb = torch.load(paths["emb"], map_location="cpu")
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb)

            emb = emb.to(self.device)
            if emb.dtype != torch.float32:
                # We'll compute in float32 (stability), even if cached as fp16.
                emb = emb.float()
            if self.cfg.normalize:
                emb = F.normalize(emb, p=2, dim=1)

            return chunk_ids, emb, meta

        # Build embeddings
        chunk_ids = [str(c[id_key]) for c in chunks]
        texts = [str(c[text_key]) for c in chunks]
        emb = self._encode_texts(texts, batch_size=self.cfg.chunk_batch_size)

        # Save compactly
        store_dtype = torch.float16 if self.cfg.cache_dtype == "float16" else torch.float32
        emb_to_store = emb.detach().to("cpu", dtype=store_dtype)

        torch.save(emb_to_store, paths["emb"])
        paths["chunk_ids"].write_text(json.dumps(chunk_ids, ensure_ascii=False), encoding="utf-8")

        meta = {
            "fingerprint": fp,
            "model_name": self.cfg.model_name,
            "device_built": self.device,
            "normalize": self.cfg.normalize,
            "cache_dtype": self.cfg.cache_dtype,
            "num_chunks": len(chunk_ids),
            "dim": int(emb.shape[1]),
            "cache_dir": str(paths["base"]),
        }
        paths["meta"].write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        # Compute in float32
        if emb.dtype != torch.float32:
            emb = emb.float()
        return chunk_ids, emb, meta

    @torch.inference_mode()
    def retrieve_many(
        self,
        query_texts: List[str],
        *,
        top_k: int = 10,
        chunk_ids: List[str],
        chunk_emb: torch.Tensor,
    ) -> List[List[Dict[str, float]]]:
        """
        Batched exact cosine top-k retrieval for a list of query texts.
        Returns a list aligned with query_texts, each element is ranked list of {"chunk_id","score"}.
        """
        # Always compute matmul in float32
        docs = chunk_emb.to(self.device)
        if docs.dtype != torch.float32:
            docs = docs.float()

        results: List[List[Dict[str, float]]] = []
        bs = self.cfg.query_batch_size

        for start in range(0, len(query_texts), bs):
            batch = query_texts[start : start + bs]
            q = self._encode_texts(batch, batch_size=bs)
            if q.dtype != torch.float32:
                q = q.float()

            sims = q @ docs.T  # [B, N]
            k_eff = min(int(top_k), sims.shape[1])

            vals, idx = torch.topk(sims, k=k_eff, dim=1, largest=True, sorted=True)
            vals = vals.detach().to("cpu")
            idx = idx.detach().to("cpu")

            for i in range(idx.shape[0]):
                rows: List[Dict[str, float]] = []
                for j, sc in zip(idx[i].tolist(), vals[i].tolist()):
                    rows.append({"chunk_id": str(chunk_ids[j]), "score": float(sc)})
                results.append(rows)

        return results

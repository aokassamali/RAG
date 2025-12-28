# This module reranks a list of candidate passages in ONE LLM call per query.
# We use Structured Outputs (json_schema) so the response is guaranteed to match a schema and is easy to parse.
from __future__ import annotations

from typing import Any, Dict, List
from openai import OpenAI

def rerank_passages_one_call(
    client: OpenAI,
    model: str,
    question: str,
    candidates: List[Dict[str, str]],
    max_chars_per_passage: int = 900,
) -> List[Dict[str, Any]]:
    """
    candidates: [{"chunk_id": "...", "text": "..."}, ...] length <= 30
    returns: [{"chunk_id": "...", "score": 0..3}, ...] same length
    """
    def clip(t: str) -> str:
        t = t.strip()
        return t if len(t) <= max_chars_per_passage else (t[:max_chars_per_passage] + "…")

    packed = [
        {"chunk_id": c["chunk_id"], "text": clip(c["text"])}
        for c in candidates
    ]

    schema = {
        "name": "rerank_scores",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "scores": {
                    "type": "array",
                    "minItems": len(packed),
                    "maxItems": len(packed),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "chunk_id": {"type": "string"},
                            "score": {"type": "integer", "minimum": 0, "maximum": 3},
                        },
                        "required": ["chunk_id", "score"],
                    },
                }
            },
            "required": ["scores"],
        },
    }

    prompt = (
        "You are a retrieval reranker.\n"
        "Score each passage for answering the question on a 0-3 scale:\n"
        "3 = directly answers\n"
        "2 = contains key info needed\n"
        "1 = related but not sufficient\n"
        "0 = irrelevant\n\n"
        f"Question:\n{question}\n\n"
        "Passages (JSON list):\n"
        f"{packed}\n\n"
        "Return scores for the same chunk_ids."
    )

    # Structured Outputs is supported in Chat Completions via response_format json_schema. :contentReference[oaicite:5]{index=5}
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_schema", "json_schema": schema},
    )

    # In structured outputs, the content is valid JSON conforming to schema.
    import json
    obj = json.loads(resp.choices[0].message.content)
    scores = obj["scores"]

    # Defensive: keep only chunk_ids we asked for; default missing to 0.
    allowed = {c["chunk_id"] for c in packed}
    out_map = {s["chunk_id"]: int(s["score"]) for s in scores if s["chunk_id"] in allowed}
    return [{"chunk_id": c["chunk_id"], "score": out_map.get(c["chunk_id"], 0)} for c in packed]

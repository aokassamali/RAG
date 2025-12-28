# Goal: Generate grounded answers with selectable style (strict / cautious / evidence_first).
# Why: Tiered policies need different behaviors; evidence_first reduces hallucinations.

from __future__ import annotations

import json
from typing import Any, Dict, List
from openai import OpenAI

SYSTEM_STRICT = (
    "You are a careful assistant. You must answer using ONLY the provided context passages. "
    "If the context does not contain enough information to answer, you MUST abstain. "
    "All factual claims must be supported by the context, and you MUST cite chunk_ids."
)

SYSTEM_CAUTIOUS = (
    "You are a cautious assistant. Use ONLY the provided context passages. "
    "If there is any ambiguity or missing detail, you MUST abstain. "
    "Keep the answer short (<=3 sentences). Avoid extrapolation. "
    "Cite the chunk_ids that directly support your key statement(s)."
)

SYSTEM_EVIDENCE_FIRST = (
    "You are a strict evidence-first assistant for RAG. Use ONLY the provided passages. "
    "If the passages do not clearly contain the answer, you MUST abstain. "
    "If you answer, keep it <=2 sentences and cite chunk_ids. "
    "Do NOT guess, do NOT fill in missing details. "
    "If you cannot provide at least one valid citation, abstain."
)

def _clip(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"

def generate_answer_structured(
    client: OpenAI,
    model: str,
    question: str,
    contexts: List[Dict[str, str]],
    max_chars_per_passage: int = 900,
    style: str = "strict",  # "strict" | "cautious" | "evidence_first"
) -> Dict[str, Any]:
    packed = [{"chunk_id": c["chunk_id"], "text": _clip(c["text"], max_chars_per_passage)} for c in contexts]
    allowed = {c["chunk_id"] for c in packed}

    schema = {
        "name": "rag_answer",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "answer": {"type": "string"},
                "cited_chunk_ids": {"type": "array", "items": {"type": "string"}},
                "abstained": {"type": "boolean"},
            },
            "required": ["answer", "cited_chunk_ids", "abstained"],
        },
    }

    if style == "evidence_first":
        system = SYSTEM_EVIDENCE_FIRST
    elif style == "cautious":
        system = SYSTEM_CAUTIOUS
    else:
        system = SYSTEM_STRICT

    prompt = (
        "Task:\n"
        "1) Answer the question using ONLY the passages below.\n"
        "2) If the passages do NOT contain enough info, set abstained=true and answer should say you can't answer from the provided context.\n"
        "3) If you answer, include citations by returning cited_chunk_ids that support the key facts.\n\n"
        f"Question:\n{question}\n\n"
        "Passages (JSON list):\n"
        f"{packed}\n\n"
        "Return JSON matching the schema."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_schema", "json_schema": schema},
    )

    obj = json.loads(resp.choices[0].message.content)
    cited = [cid for cid in obj.get("cited_chunk_ids", []) if cid in allowed]
    abstained = bool(obj.get("abstained", False))
    answer = (obj.get("answer") or "").strip()

    # Enforce "no citation => abstain" (especially important for Tier B evidence-first)
    if (not abstained) and (len(cited) == 0):
        return {
            "answer": "I can't answer this from the provided context.",
            "cited_chunk_ids": [],
            "abstained": True,
        }

    return {"answer": answer, "cited_chunk_ids": cited, "abstained": abstained}

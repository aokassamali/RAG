# Goal: LLM-as-judge scoring for correctness + groundedness + citation validity + abstention behavior.
# Why: Reliability is about answer quality, not just retrieval metrics.

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from openai import OpenAI

JUDGE_SYSTEM = (
    "You are a strict evaluator for a Retrieval-Augmented Generation (RAG) system. "
    "You must follow the rubric and return ONLY valid JSON."
)

def judge_answer_structured(
    client: OpenAI,
    model: str,
    question: str,
    answer: str,
    abstained: bool,
    cited_chunk_ids: List[str],
    contexts: List[Dict[str, str]],
    gold_answer_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    correctness: -1 if no gold provided; else 0/1/2
    grounded: 0/1
    citation_valid: 0/1
    abstention_correct: 0/1
    """
    schema = {
        "name": "rag_judgment",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "correctness": {"type": "integer", "minimum": -1, "maximum": 2},
                "grounded": {"type": "integer", "minimum": 0, "maximum": 1},
                "citation_valid": {"type": "integer", "minimum": 0, "maximum": 1},
                "abstention_correct": {"type": "integer", "minimum": 0, "maximum": 1},
                "notes": {"type": "string"},
            },
            "required": ["correctness", "grounded", "citation_valid", "abstention_correct", "notes"],
        },
    }

    # Pack contexts with ids for judge to check citations and support
    packed = [{"chunk_id": c["chunk_id"], "text": (c.get("text") or "").strip()} for c in contexts]

    rubric = (
        "Rubric:\n"
        "- correctness: 2=fully correct vs gold, 1=partially correct, 0=incorrect. "
        "If no gold_answer_text provided, set correctness=-1.\n"
        "- grounded: 1 if ALL factual claims in the answer are supported by the provided passages, else 0.\n"
        "- citation_valid: 1 if cited_chunk_ids are sufficient and actually contain the supporting evidence, else 0.\n"
        "- abstention_correct: 1 if the system abstained when context is insufficient OR answered when sufficient; else 0.\n"
        "Be strict.\n"
    )

    prompt = (
        f"{rubric}\n"
        f"Question:\n{question}\n\n"
        f"Gold answer text (may be empty):\n{gold_answer_text or ''}\n\n"
        f"Model abstained: {abstained}\n\n"
        f"Model answer:\n{answer}\n\n"
        f"Cited chunk ids:\n{cited_chunk_ids}\n\n"
        "Passages:\n"
        f"{packed}\n\n"
        "Return JSON matching the schema."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_schema", "json_schema": schema},
    )

    return json.loads(resp.choices[0].message.content)

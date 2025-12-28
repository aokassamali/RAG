# Goal: LLM-based reranker: given (query, chunk), output a relevance score.
# Why: Cross-encoder-like behavior without torch; improves ranking quality (nDCG/MRR).

from __future__ import annotations
import json
from typing import Any, Dict, List
from openai import OpenAI

SYSTEM = (
  "You are a retrieval reranker. Score how relevant a passage is for answering the user question. "
  "Return ONLY valid JSON."
)

def build_prompt(question: str, passage: str) -> str:
    return (
        "Score relevance on a 0-3 scale:\n"
        "3 = directly answers the question\n"
        "2 = contains key info needed to answer\n"
        "1 = related but not sufficient\n"
        "0 = irrelevant\n\n"
        f"Question: {question}\n\n"
        f"Passage:\n{passage}\n\n"
        "Return JSON: {\"score\": <0|1|2|3>}"
    )

def score_passages(
    client: Any,
    model: str,
    question: str,
    passages: List[str],
) -> List[int]:
    scores: List[int] = []
    for p in passages:
        msg = build_prompt(question, p)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": msg},
            ],
            temperature=0.0,
        )
        text = resp.choices[0].message.content.strip()
        try:
            obj = json.loads(text)
            s = int(obj["score"])
            if s < 0 or s > 3:
                raise ValueError("score out of range")
        except Exception:
            # Conservative fallback: treat as low relevance if parsing fails
            s = 0
        scores.append(s)
    return scores

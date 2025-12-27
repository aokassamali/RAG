# Goal: Chunk a document into overlapping passages with stable chunk_ids + character offsets.
# Why: Retrieval operates over chunks, and grounding spans must be mapped into chunk space.

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import re

def _simple_tokenize(text: str) -> List[str]:
    # Simple whitespace-ish tokenization for V1.
    # Later: swap in tiktoken for closer-to-model token counts.
    return re.findall(r"\S+", text)

def chunk_text(text: str, target_tokens: int = 350, overlap_tokens: int = 50) -> List[Tuple[int, int, str]]:
    tokens = _simple_tokenize(text)
    if not tokens:
        return []

    # We reconstruct approximate char offsets by searching progressively.
    # V1 assumption: good enough for mapping spans -> chunk via char overlap later,
    # since spans use true char offsets and we keep chunk start/end by slicing text directly.
    chunks = []

    # Build token->char map by scanning text once
    # (find each token occurrence in order).
    positions = []
    idx = 0
    for tok in tokens:
        m = re.search(re.escape(tok), text[idx:])
        if m is None:
            # fallback: give up mapping, use entire text
            return [(0, len(text), text)]
        start = idx + m.start()
        end = idx + m.end()
        positions.append((start, end))
        idx = end

    step = max(1, target_tokens - overlap_tokens)
    n = len(tokens)

    start_i = 0
    while start_i < n:
        end_i = min(n, start_i + target_tokens)
        start_char = positions[start_i][0]
        end_char = positions[end_i - 1][1]
        chunk = text[start_char:end_char]
        chunks.append((start_char, end_char, chunk))

        if end_i == n:
            break
        start_i += step

    return chunks

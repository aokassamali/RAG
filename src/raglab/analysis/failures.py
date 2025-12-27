# Goal: Create a failure table with heuristic tags to guide what to try next (dense? rerank? chunking?).
# Why: "Error analysis" is a key production/research skill and makes the project feel real.

from __future__ import annotations
from typing import Dict, Any, List, Set
import re

def q_tokens(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))

def failure_rows(gold_rows: List[Dict[str, Any]], pred_rows: List[Dict[str, Any]], k: int, chunk_text_by_id: Dict[str, str]) -> List[Dict[str, Any]]:
    preds_by_qid = {r["qid"]: r for r in pred_rows}
    out = []

    for g in gold_rows:
        qid = g["qid"]
        p = preds_by_qid.get(qid)
        if p is None:
            continue
        pred_ids = [x["chunk_id"] for x in p["preds"]][:k]
        gold_set = set(g["gold_chunk_ids"])
        hit = any(pid in gold_set for pid in pred_ids)
        if hit:
            continue

        qt = q_tokens(g["question"])
        # Aggregate gold chunk text (may be multiple)
        gold_text = " ".join(chunk_text_by_id.get(cid, "") for cid in list(gold_set)[:3])
        gt = q_tokens(gold_text)

        lexical_overlap = len(qt & gt)
        has_digits = any(ch.isdigit() for ch in g["question"])
        short_query = len(qt) <= 4
        multi_gold = len(gold_set) > 1

        # Heuristic tags
        tags = []
        if lexical_overlap == 0:
            tags.append("lexical_gap__dense_candidate")
        if has_digits:
            tags.append("has_digits__exact_match_sensitive")
        if short_query:
            tags.append("short_query__ambiguous")
        if multi_gold:
            tags.append("multi_gold_chunks__boundary_or_long_span")

        out.append({
            "qid": qid,
            "doc_id": g["doc_id"],
            "domain": g.get("domain",""),
            "question": g["question"],
            "primary_gold_chunk_id": g.get("primary_gold_chunk_id"),
            "gold_chunk_count": len(gold_set),
            "top_pred_chunk_id": pred_ids[0] if pred_ids else None,
            "lexical_overlap_q_vs_gold": lexical_overlap,
            "tags": ";".join(tags),
            "human_label": "",  # fill this in manually later
            "notes": "",
        })

    return out

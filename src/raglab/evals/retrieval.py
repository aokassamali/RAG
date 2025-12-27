# Goal: Compute retrieval metrics (Recall@K, MRR, nDCG@K) in a reusable way.
from __future__ import annotations
import math
from typing import Dict, Any, List, Set

def dcg_binary(rels: List[int]) -> float:
    s = 0.0
    for i, r in enumerate(rels, start=1):
        if r:
            s += 1.0 / math.log2(i + 1)
    return s

def ndcg_at_k(pred_ids: List[str], gold_set: Set[str], k: int) -> float:
    rels = [1 if pid in gold_set else 0 for pid in pred_ids[:k]]
    dcg_val = dcg_binary(rels)
    ideal = sorted(rels, reverse=True)
    idcg = dcg_binary(ideal)
    return 0.0 if idcg == 0 else dcg_val / idcg

def mrr(pred_ids: List[str], gold_set: Set[str]) -> float:
    for i, pid in enumerate(pred_ids, start=1):
        if pid in gold_set:
            return 1.0 / i
    return 0.0

def recall_at_k(pred_ids: List[str], gold_set: Set[str], k: int) -> float:
    return 1.0 if any(pid in gold_set for pid in pred_ids[:k]) else 0.0

def evaluate_retrieval(gold_rows: List[Dict[str, Any]], pred_rows: List[Dict[str, Any]], k: int) -> Dict[str, Any]:
    preds_by_qid = {r["qid"]: r for r in pred_rows}
    n = 0
    miss = 0
    r_sum = mrr_sum = ndcg_sum = 0.0

    for g in gold_rows:
        qid = g["qid"]
        p = preds_by_qid.get(qid)
        if p is None:
            miss += 1
            continue
        pred_ids = [x["chunk_id"] for x in p["preds"]]
        gold_set = set(g["gold_chunk_ids"])
        r_sum += recall_at_k(pred_ids, gold_set, k)
        mrr_sum += mrr(pred_ids, gold_set)
        ndcg_sum += ndcg_at_k(pred_ids, gold_set, k)
        n += 1

    return {
        "k": k,
        "n_eval": n,
        "missing_pred": miss,
        "recall_at_k": r_sum / n if n else 0.0,
        "mrr": mrr_sum / n if n else 0.0,
        "ndcg_at_k": ndcg_sum / n if n else 0.0,
    }

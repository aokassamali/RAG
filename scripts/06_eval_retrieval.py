# Goal: Compute retrieval metrics comparing BM25 top-K vs gold_chunk_ids.
# Why: quantitative baseline before dense/hybrid/reranking.

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List

REPO_ROOT = Path(__file__).resolve().parents[1]
VAL_GM_PATH = REPO_ROOT / "data" / "processed" / "grounding_map_validation_chunks.jsonl"
PRED_PATH = REPO_ROOT / "runs" / "bm25_v1" / "val_predictions.jsonl"
OUT_METRICS = REPO_ROOT / "runs" / "bm25_v1" / "val_metrics.json"

K = 10

def read_jsonl_to_dict(path: Path, key: str) -> Dict[str, Dict[str, Any]]:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            out[obj[key]] = obj
    return out

def dcg(rels: List[int]) -> float:
    # rels in rank order
    s = 0.0
    for i, r in enumerate(rels, start=1):
        if r:
            s += 1.0 / ( (i + 1) ** 0.0 )  # placeholder; see below
    return s

def dcg_binary(rels: List[int]) -> float:
    # Standard DCG with log2 discount
    import math
    s = 0.0
    for i, r in enumerate(rels, start=1):
        if r:
            s += 1.0 / math.log2(i + 1)
    return s

def ndcg_at_k(pred_ids: List[str], gold_set: set, k: int) -> float:
    rels = [1 if pid in gold_set else 0 for pid in pred_ids[:k]]
    dcg_val = dcg_binary(rels)
    ideal_rels = sorted(rels, reverse=True)
    idcg = dcg_binary(ideal_rels)
    return 0.0 if idcg == 0 else dcg_val / idcg

def mrr(pred_ids: List[str], gold_set: set) -> float:
    for i, pid in enumerate(pred_ids, start=1):
        if pid in gold_set:
            return 1.0 / i
    return 0.0

def recall_at_k(pred_ids: List[str], gold_set: set, k: int) -> float:
    return 1.0 if any(pid in gold_set for pid in pred_ids[:k]) else 0.0

def main() -> None:
    gold = read_jsonl_to_dict(VAL_GM_PATH, "qid")
    preds = read_jsonl_to_dict(PRED_PATH, "qid")

    # Predict before running:
    # - #preds should equal #gold (3972).
    # - Metrics should be > 0, likely reasonably high since evidence is in-doc.
    n = 0
    r_sum = 0.0
    mrr_sum = 0.0
    ndcg_sum = 0.0

    missing_pred = 0

    for qid, g in gold.items():
        p = preds.get(qid)
        if p is None:
            missing_pred += 1
            continue

        pred_ids = [x["chunk_id"] for x in p["preds"]]
        gold_set = set(g["gold_chunk_ids"])

        r_sum += recall_at_k(pred_ids, gold_set, K)
        mrr_sum += mrr(pred_ids, gold_set)
        ndcg_sum += ndcg_at_k(pred_ids, gold_set, K)
        n += 1

    metrics = {
        "k": K,
        "n_eval": n,
        "missing_pred": missing_pred,
        "recall_at_k": r_sum / n if n else 0.0,
        "mrr": mrr_sum / n if n else 0.0,
        "ndcg_at_k": ndcg_sum / n if n else 0.0,
    }

    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    OUT_METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()

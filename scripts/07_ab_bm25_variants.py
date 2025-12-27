# Goal: Run controlled retrieval experiments: compare tokenization + BM25 variants with fixed eval protocol.
# Why: Reproducible ablation study (A/B) with metrics logged to runs/ for resume-grade evidence.

from __future__ import annotations
import json, re, math
from pathlib import Path
from typing import Dict, Any, List, Callable, Tuple

from rank_bm25 import BM25Okapi, BM25Plus

REPO_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = REPO_ROOT / "data" / "processed" / "chunks.jsonl"
VAL_GM_PATH = REPO_ROOT / "data" / "processed" / "grounding_map_validation_chunks.jsonl"
RUN_DIR = REPO_ROOT / "runs" / "bm25_ablation_v1"
RUN_DIR.mkdir(parents=True, exist_ok=True)

K = 10

SAFE_STOPWORDS = {
    # common glue words (small list is fine for V1)
    "the","a","an","and","or","to","of","in","on","for","with","at","by","from",
    "as","it","this","that","these","those","be","is","are","was","were","been",
    "i","you","he","she","they","we","my","your","their","our","me","him","her",
    "do","does","did","done","can","could","would","will","just","about","into",
    "than","then","there","here","also","if","when","where","which","who","whom"
}
# keep negations / modals OUT of stopwords so we don't erase meaning
KEEP_WORDS = {"no","not","never","without","must","should","may","cannot","can't","won't","n't"}

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out

def tok_basic(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

def tok_safe_stopwords(text: str) -> List[str]:
    toks = tok_basic(text)
    return [t for t in toks if (t in KEEP_WORDS) or (t not in SAFE_STOPWORDS)]

def dcg_binary(rels: List[int]) -> float:
    s = 0.0
    for i, r in enumerate(rels, start=1):
        if r:
            s += 1.0 / math.log2(i + 1)
    return s

def ndcg_at_k(pred_ids: List[str], gold_set: set, k: int) -> float:
    rels = [1 if pid in gold_set else 0 for pid in pred_ids[:k]]
    dcg_val = dcg_binary(rels)
    ideal = sorted(rels, reverse=True)
    idcg = dcg_binary(ideal)
    return 0.0 if idcg == 0 else dcg_val / idcg

def mrr(pred_ids: List[str], gold_set: set) -> float:
    for i, pid in enumerate(pred_ids, start=1):
        if pid in gold_set:
            return 1.0 / i
    return 0.0

def recall_at_k(pred_ids: List[str], gold_set: set, k: int) -> float:
    return 1.0 if any(pid in gold_set for pid in pred_ids[:k]) else 0.0

def eval_preds(gold_rows: List[Dict[str, Any]], pred_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    preds_by_qid = {r["qid"]: r for r in pred_rows}
    n = 0
    r_sum = mrr_sum = ndcg_sum = 0.0
    missing = 0
    for g in gold_rows:
        qid = g["qid"]
        p = preds_by_qid.get(qid)
        if p is None:
            missing += 1
            continue
        pred_ids = [x["chunk_id"] for x in p["preds"]]
        gold_set = set(g["gold_chunk_ids"])
        r_sum += recall_at_k(pred_ids, gold_set, K)
        mrr_sum += mrr(pred_ids, gold_set)
        ndcg_sum += ndcg_at_k(pred_ids, gold_set, K)
        n += 1
    return {
        "k": K,
        "n_eval": n,
        "missing_pred": missing,
        "recall_at_k": r_sum / n if n else 0.0,
        "mrr": mrr_sum / n if n else 0.0,
        "ndcg_at_k": ndcg_sum / n if n else 0.0,
    }

def retrieve(
    chunks: List[Dict[str, Any]],
    gold_rows: List[Dict[str, Any]],
    tokenizer: Callable[[str], List[str]],
    bm25_kind: str,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    corpus_tokens = [tokenizer(c["text"]) for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    if bm25_kind == "okapi":
        bm25 = BM25Okapi(corpus_tokens)
    elif bm25_kind == "plus":
        bm25 = BM25Plus(corpus_tokens)
    else:
        raise ValueError("bm25_kind must be okapi or plus")

    out = []
    for ex in gold_rows:
        qid = ex["qid"]
        q_toks = tokenizer(ex["question"])
        scores = bm25.get_scores(q_toks)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        preds = [{"chunk_id": chunk_ids[i], "score": float(scores[i])} for i in top_idx]
        out.append({"qid": qid, "preds": preds})
    return out

def main() -> None:
    # Predict before running:
    # - It should load 1506 chunks and 3972 validation queries (your latest counts).
    # - A3 should be slightly better or roughly equal to A1; BM25Plus may help a bit.

    chunks = read_jsonl(CHUNKS_PATH)
    gold = read_jsonl(VAL_GM_PATH)

    variants: List[Tuple[str, Callable[[str], List[str]], str]] = [
        ("A1_okapi_basic", tok_basic, "okapi"),
        ("A3_okapi_safe_stopwords", tok_safe_stopwords, "okapi"),
        ("A4_plus_basic", tok_basic, "plus"),
        ("A4_plus_safe_stopwords", tok_safe_stopwords, "plus"),
    ]

    summary = {"chunks": len(chunks), "val_queries": len(gold), "variants": {}}

    for name, tok, kind in variants:
        pred = retrieve(chunks, gold, tok, kind, top_k=K)
        metrics = eval_preds(gold, pred)
        summary["variants"][name] = metrics
        (RUN_DIR / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

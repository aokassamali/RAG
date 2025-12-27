# Goal: Ensure the end-to-end pipeline works on deterministic sample (fast, CI-friendly).
from pathlib import Path
from raglab.data.loaders import load_chunks, load_grounding
from raglab.indexing.bm25 import BM25Retriever, tok_a3_safe_stopwords
from raglab.evals.retrieval import evaluate_retrieval

def test_bm25_sample_end_to_end():
    repo_root = Path(__file__).resolve().parents[1]
    chunks = load_chunks(repo_root, use_samples=True)
    gold = load_grounding(repo_root, split="validation", use_samples=True)

    retriever = BM25Retriever.build(chunks, tokenizer=tok_a3_safe_stopwords)

    preds = [{"qid": ex["qid"], "preds": retriever.retrieve(ex["question"], top_k=10)} for ex in gold]
    metrics = evaluate_retrieval(gold, preds, k=10)

    assert metrics["n_eval"] == len(gold)
    assert 0.0 <= metrics["recall_at_k"] <= 1.0
    assert 0.0 <= metrics["mrr"] <= 1.0
    assert 0.0 <= metrics["ndcg_at_k"] <= 1.0

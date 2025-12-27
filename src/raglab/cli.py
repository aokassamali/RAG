# Goal: One-command runs that write all artifacts (predictions, metrics, calibration, failures, manifest).
# Why: This is the "production-grade" backbone that makes improvements measurable and reproducible.

from __future__ import annotations
from pathlib import Path
import typer
import pandas as pd
from openai import OpenAI

from raglab.data.loaders import load_chunks, load_grounding
from raglab.indexing.bm25 import BM25Retriever, tok_a3_safe_stopwords, tok_basic
from raglab.utils.run_artifacts import make_run_dir, write_manifest
from raglab.utils.io import write_jsonl, write_json
from raglab.evals.retrieval import evaluate_retrieval, recall_at_k
from raglab.evals.calibration import confidence_from_scores, ece, coverage_accuracy_curve
from raglab.analysis.failures import failure_rows
from raglab.indexing.dense_openai import OpenAIDenseRetriever, DenseIndex

app = typer.Typer(add_completion=False)

def chunk_text_map(chunks):
    return {c["chunk_id"]: c["text"] for c in chunks}

@app.command()
def run_bm25(
    run_name: str = typer.Option("bm25_a3", help="Name used in runs/ folder"),
    split: str = typer.Option("validation", help="train|validation"),
    k: int = typer.Option(10, help="Top-K retrieval"),
    tokenizer: str = typer.Option("a3", help="basic|a3"),
    use_samples: bool = typer.Option(False, help="Use deterministic sample in data/samples"),
    sample_queries_only: bool = typer.Option(False, help="Use sample queries but FULL chunk corpus"),
    cache_index: bool = typer.Option(True, help="Cache BM25 index pickle in run dir"),
):
    repo_root = Path(__file__).resolve().parents[2]
    runs_root = repo_root / "runs"
    run_dir = make_run_dir(runs_root, run_name)

    # Predict before running:
    # - For full validation: ~3972 queries
    # - For sample: 120 queries
    chunks = load_chunks(repo_root, use_samples=use_samples, sample_queries_only=sample_queries_only)
    gold = load_grounding(repo_root, split=split, use_samples=use_samples)

    tok = tok_a3_safe_stopwords if tokenizer.lower() == "a3" else tok_basic

    index_path = run_dir / "bm25_index.pkl"
    if cache_index and index_path.exists():
        retriever = BM25Retriever.load(index_path)
    else:
        retriever = BM25Retriever.build(chunks, tokenizer=tok)
        if cache_index:
            retriever.save(index_path)

    # Retrieve
    preds = []
    confs = []
    accs = []

    for ex in gold:
        p = retriever.retrieve(ex["question"], top_k=k)
        preds.append({"qid": ex["qid"], "preds": p})

        # Calibration target = hit@K
        pred_ids = [x["chunk_id"] for x in p]
        gold_set = set(ex["gold_chunk_ids"])
        hit = recall_at_k(pred_ids, gold_set, k)
        accs.append(float(hit))
        confs.append(float(confidence_from_scores(p)))

    # Metrics
    metrics = evaluate_retrieval(gold, preds, k=k)

    # Calibration artifacts
    cal = {
        "confidence_proxy": "sigmoid(score_top1 - score_top2)",
        "ece": ece(confs, accs, n_bins=10),
        "coverage_accuracy": coverage_accuracy_curve(confs, accs, points=20),
    }

    # Failure taxonomy CSV
    chunk_map = chunk_text_map(chunks)
    fails = failure_rows(gold, preds, k=k, chunk_text_by_id=chunk_map)
    fails_df = pd.DataFrame(fails)

    # Write artifacts
    write_json(run_dir / "config.json", {
        "retriever": "bm25_okapi",
        "tokenizer": tokenizer,
        "k": k,
        "split": split,
        "use_samples": use_samples,
        "cache_index": cache_index,
    })
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "calibration.json", cal)
    write_jsonl(run_dir / "predictions.jsonl", preds)
    fails_df.to_csv(run_dir / "failures.csv", index=False)

    write_manifest(run_dir, extra={"dataset": {"chunks": len(chunks), "queries": len(gold)}})

    typer.echo(f"Run dir: {run_dir}")
    typer.echo(f"Metrics: {metrics}")
    typer.echo(f"Failures: {len(fails_df)} -> {run_dir / 'failures.csv'}")

@app.command()
def explain_next_steps():
    typer.echo("Next: add dense retrieval (API embeddings), then hybrid, then reranking, then end-to-end RAG + judge.")

@app.command()
def run_dense(
    run_name: str = typer.Option("dense_v1", help="Run name"),
    split: str = typer.Option("validation", help="train|validation"),
    k: int = typer.Option(10, help="Top-K"),
    use_samples: bool = typer.Option(False),
    sample_queries_only: bool = typer.Option(False),
    model: str = typer.Option("text-embedding-3-small", help="OpenAI embedding model"),
    cache_index: bool = typer.Option(True),
    batch_size: int = typer.Option(128),
):
    repo_root = Path(__file__).resolve().parents[2]
    runs_root = repo_root / "runs"
    run_dir = make_run_dir(runs_root, run_name)

    chunks = load_chunks(repo_root, use_samples=use_samples, sample_queries_only=sample_queries_only)
    gold = load_grounding(repo_root, split=split, use_samples=use_samples)

    index_path = run_dir / "dense_index.pkl"
    if cache_index and index_path.exists():
        idx = DenseIndex.load(index_path)
        retriever = OpenAIDenseRetriever(client=OpenAI(), model=model, index=idx)
    else:
        retriever = OpenAIDenseRetriever.build(chunks, model=model, batch_size=batch_size)
        if cache_index:
            retriever.index.save(index_path)

    preds = []
    confs = []
    accs = []
    for ex in gold:
        p = retriever.retrieve(ex["question"], top_k=k)
        preds.append({"qid": ex["qid"], "preds": p})

        pred_ids = [x["chunk_id"] for x in p]
        gold_set = set(ex["gold_chunk_ids"])
        hit = recall_at_k(pred_ids, gold_set, k)
        accs.append(float(hit))
        confs.append(float(confidence_from_scores(p)))

    metrics = evaluate_retrieval(gold, preds, k=k)
    cal = {
        "confidence_proxy": "sigmoid(score_top1 - score_top2)",
        "ece": ece(confs, accs, n_bins=10),
        "coverage_accuracy": coverage_accuracy_curve(confs, accs, points=20),
    }

    chunk_map = chunk_text_map(chunks)
    fails = failure_rows(gold, preds, k=k, chunk_text_by_id=chunk_map)
    fails_df = pd.DataFrame(fails)

    write_json(run_dir / "config.json", {
        "retriever": "openai_dense",
        "embedding_model": model,
        "k": k,
        "split": split,
        "use_samples": use_samples,
        "sample_queries_only": sample_queries_only,
        "cache_index": cache_index,
        "batch_size": batch_size,
    })
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "calibration.json", cal)
    write_jsonl(run_dir / "predictions.jsonl", preds)
    fails_df.to_csv(run_dir / "failures.csv", index=False)

    write_manifest(run_dir, extra={"dataset": {"chunks": len(chunks), "queries": len(gold)}})

    typer.echo(f"Run dir: {run_dir}")
    typer.echo(f"Metrics: {metrics}")


def rrf_fuse(bm25_preds, dense_preds, rrf_k: int = 60):
    # bm25_preds/dense_preds: list of {"chunk_id":..., "score":...} in ranked order
    scores = {}
    for rank, p in enumerate(bm25_preds, start=1):
        cid = p["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    for rank, p in enumerate(dense_preds, start=1):
        cid = p["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"chunk_id": cid, "score": float(sc)} for cid, sc in fused]

@app.command()
def run_hybrid_rrf(
    run_name: str = typer.Option("hybrid_rrf_v1"),
    split: str = typer.Option("validation"),
    k: int = typer.Option(10),
    use_samples: bool = typer.Option(False),
    sample_queries_only: bool = typer.Option(False),
    bm25_tokenizer: str = typer.Option("a3"),
    dense_model: str = typer.Option("text-embedding-3-small"),
    top_bm25: int = typer.Option(50),
    top_dense: int = typer.Option(50),
    rrf_k: int = typer.Option(60),
):
    repo_root = Path(__file__).resolve().parents[2]
    runs_root = repo_root / "runs"
    run_dir = make_run_dir(runs_root, run_name)

    chunks = load_chunks(repo_root, use_samples=use_samples, sample_queries_only=sample_queries_only)
    gold = load_grounding(repo_root, split=split, use_samples=use_samples)

    # Build/load BM25
    tok = tok_a3_safe_stopwords if bm25_tokenizer.lower() == "a3" else tok_basic
    bm25 = BM25Retriever.build(chunks, tokenizer=tok)

    # Build/load dense index (cache per run for now; later we can move to a shared cache dir)
    index_path = run_dir / "dense_index.pkl"
    if index_path.exists():
        idx = DenseIndex.load(index_path)
        dense = OpenAIDenseRetriever(client=OpenAI(), model=dense_model, index=idx)
    else:
        dense = OpenAIDenseRetriever.build(chunks, model=dense_model, batch_size=128)
        dense.index.save(index_path)

    preds, confs, accs = [], [], []
    for ex in gold:
        b = bm25.retrieve(ex["question"], top_k=top_bm25)
        d = dense.retrieve(ex["question"], top_k=top_dense)
        fused = rrf_fuse(b, d, rrf_k=rrf_k)[:k]

        preds.append({"qid": ex["qid"], "preds": fused})

        pred_ids = [x["chunk_id"] for x in fused]
        gold_set = set(ex["gold_chunk_ids"])
        hit = recall_at_k(pred_ids, gold_set, k)
        accs.append(float(hit))
        confs.append(float(confidence_from_scores(fused)))

    metrics = evaluate_retrieval(gold, preds, k=k)
    cal = {
        "confidence_proxy": "sigmoid(score_top1 - score_top2) [note: fused score]",
        "ece": ece(confs, accs, n_bins=10),
        "coverage_accuracy": coverage_accuracy_curve(confs, accs, points=20),
    }

    chunk_map = chunk_text_map(chunks)
    fails = failure_rows(gold, preds, k=k, chunk_text_by_id=chunk_map)
    pd.DataFrame(fails).to_csv(run_dir / "failures.csv", index=False)

    write_json(run_dir / "config.json", {
        "retriever": "hybrid_rrf",
        "bm25_tokenizer": bm25_tokenizer,
        "dense_model": dense_model,
        "k": k,
        "top_bm25": top_bm25,
        "top_dense": top_dense,
        "rrf_k": rrf_k,
        "use_samples": use_samples,
        "sample_queries_only": sample_queries_only,
    })
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "calibration.json", cal)
    write_jsonl(run_dir / "predictions.jsonl", preds)
    write_manifest(run_dir, extra={"dataset": {"chunks": len(chunks), "queries": len(gold)}})

    typer.echo(f"Run dir: {run_dir}")
    typer.echo(f"Metrics: {metrics}")

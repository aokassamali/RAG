# Goal: One-command runs that write all artifacts (predictions, metrics, calibration, failures, manifest).
# Why: This is the "production-grade" backbone that makes improvements measurable and reproducible.

from __future__ import annotations
from pathlib import Path
import typer
import pandas as pd
import json
from openai import OpenAI

from raglab.data.loaders import load_chunks, load_grounding
from raglab.indexing.bm25 import BM25Retriever, tok_a3_safe_stopwords, tok_basic
from raglab.utils.run_artifacts import make_run_dir, write_manifest
from raglab.utils.io import write_jsonl, write_json, read_jsonl
from raglab.evals.retrieval import evaluate_retrieval, recall_at_k
from raglab.evals.calibration import confidence_from_scores, ece, coverage_accuracy_curve
from raglab.analysis.failures import failure_rows
from raglab.indexing.dense_openai import OpenAIDenseRetriever, DenseIndex
from raglab.rerank.llm_openai import score_passages
from raglab.rerank.llm_batch import rerank_passages_one_call

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

@app.command()
def run_rerank_llm_batch(
    run_name: str = typer.Option("rerank_llm_batch_v1"),
    input_run_dir: str = typer.Option(..., help="Hybrid run dir with predictions.jsonl (k>=top_n)"),
    split: str = typer.Option("validation"),
    top_n: int = typer.Option(30, help="How many candidates per query to rerank from the input run"),
    k_eval: int = typer.Option(10, help="Evaluate at K after reranking"),
    model: str = typer.Option("gpt-5.2-2025-12-11", help="Reranker model"),
    max_chars_per_passage: int = typer.Option(350, help="Truncate each passage to this many characters"),
    use_samples: bool = typer.Option(False, help="Use deterministic sample queries (120)"),
    failed_only: bool = typer.Option(False, help="Only rerank queries that fail@k_eval in the input run"),
    recoverable_only: bool = typer.Option(False, help="Only rerank queries that fail@k_eval BUT hit@top_n in the input run (recoverable by reranking)"),
    max_queries: int = typer.Option(0, help="Max queries to process (0 = no cap)"),
    resume: bool = typer.Option(True, help="Resume from rerank_partial.jsonl if present"),
):
    # Goal: Budget-safe, resumable LLM reranker with optional filters (samples, failed-only).
    # Why: Long-running API jobs must be resumable and controllable; we use 1 call per query via structured outputs.

    from pathlib import Path
    import json
    import pandas as pd
    from openai import OpenAI

    from raglab.utils.io import read_jsonl, write_jsonl, write_json
    from raglab.utils.run_artifacts import make_run_dir, write_manifest
    from raglab.data.loaders import load_grounding, load_chunks
    from raglab.evals.retrieval import evaluate_retrieval
    from raglab.analysis.failures import failure_rows
    from raglab.rerank.llm_batch import rerank_passages_one_call  # adjust if your module path differs

    repo_root = Path(__file__).resolve().parents[2]
    runs_root = repo_root / "runs"
    run_dir = make_run_dir(runs_root, run_name)

    # Load data
    gold = load_grounding(repo_root, split=split, use_samples=use_samples)
    chunks = load_chunks(repo_root, use_samples=False, sample_queries_only=False)
    text_by_id = chunk_text_map(chunks)  # you said this already exists at top of cli.py

    # Load input candidates
    input_dir = Path(input_run_dir)
    base_preds = read_jsonl(input_dir / "predictions.jsonl")
    base_by_qid = {r["qid"]: r for r in base_preds}

    # Resume support
    partial_path = run_dir / "rerank_partial.jsonl"
    done_qids = set()
    out_preds = []

    if resume and partial_path.exists():
        for r in read_jsonl(partial_path):
            done_qids.add(r["qid"])
            out_preds.append(r)

    gold_by_qid = {g["qid"]: set(g["gold_chunk_ids"]) for g in gold}

    def hit_at_k(pred_ids, gold_set, k):
        return any(pid in gold_set for pid in pred_ids[:k])

    # Optional: only rerank failed queries under the *input run's* current ordering
    failed_qids = None
    if failed_only or recoverable_only:
        failed_qids = set()
        recoverable_count = 0

        for g in gold:
            qid = g["qid"]
            cand = base_by_qid[qid]["preds"]  # input run's ordering
            pred_ids = [p["chunk_id"] for p in cand]
            gset = gold_by_qid[qid]

            hit10 = hit_at_k(pred_ids, gset, k_eval)
            hitN = hit_at_k(pred_ids, gset, top_n)

            if not hit10:
                if recoverable_only:
                    if hitN:
                        failed_qids.add(qid)
                        recoverable_count += 1
                else:
                    failed_qids.add(qid)
                    if hitN:
                        recoverable_count += 1

        # Nice debug print: ceiling for reranking on this subset
        typer.echo(f"Failed@{k_eval}: {len(failed_qids)}; recoverable within top_{top_n}: {recoverable_count}")


    client = OpenAI()

    processed = 0
    for ex in gold:
        qid = ex["qid"]

        if qid in done_qids:
            continue
        if failed_qids is not None and qid not in failed_qids:
            continue

        cand = base_by_qid[qid]["preds"][:top_n]
        candidates = [{"chunk_id": p["chunk_id"], "text": text_by_id.get(p["chunk_id"], "")} for p in cand]

        scored = rerank_passages_one_call(
            client=client,
            model=model,
            question=ex["question"],
            candidates=candidates,
            max_chars_per_passage=max_chars_per_passage,
        )
        # scored is aligned to candidates; each item {"chunk_id":..., "score": int 0..3}

        # Stable sort: higher score first; break ties by original rank (earlier wins)
        order = sorted(
            range(len(scored)),
            key=lambda i: (scored[i]["score"], -i),
            reverse=True,
        )
        reranked = [scored[i] for i in order][:k_eval]

        row = {"qid": qid, "preds": reranked}
        out_preds.append(row)

        # Checkpoint immediately so Ctrl+C is safe
        with partial_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        processed += 1
        if max_queries and processed >= max_queries:
            break

    # Evaluate only on qids we produced predictions for (important for failed_only / max_queries)
    out_qids = {r["qid"] for r in out_preds}
    eval_gold = [g for g in gold if g["qid"] in out_qids]

    metrics = evaluate_retrieval(eval_gold, out_preds, k=k_eval)

    # Failure taxonomy after rerank (only for eval set)
    fails = failure_rows(eval_gold, out_preds, k=k_eval, chunk_text_by_id=text_by_id)
    pd.DataFrame(fails).to_csv(run_dir / "failures.csv", index=False)

    # Write artifacts
    write_json(run_dir / "config.json", {
        "reranker": "llm_batch_structured_outputs",
        "model": model,
        "input_run_dir": str(input_dir),
        "top_n": top_n,
        "k_eval": k_eval,
        "split": split,
        "use_samples": use_samples,
        "failed_only": failed_only,
        "max_queries": max_queries,
        "resume": resume,
        "max_chars_per_passage": max_chars_per_passage,
    })
    write_json(run_dir / "metrics.json", metrics)
    write_jsonl(run_dir / "predictions.jsonl", out_preds)
    write_manifest(run_dir, extra={"dataset": {"queries_evaluated": len(eval_gold)}})

    typer.echo(f"Run dir: {run_dir}")
    typer.echo(f"Metrics: {metrics}")
    typer.echo(f"Evaluated queries: {len(eval_gold)} (processed this run: {processed})")
    if failed_only or recoverable_only:
        typer.echo(f"Filtered mode: {len(failed_qids)} queries (within {len(gold)} total)")

@app.command()
def run_rag(
    run_name: str = typer.Option("rag_gen_v2"),
    input_run_dir: str = typer.Option(..., help="Run dir with predictions.jsonl to use as context (e.g., rerank_rep_all120)"),
    split: str = typer.Option("validation"),
    k_context: int = typer.Option(5, help="How many top chunks to feed to the generator"),
    model: str = typer.Option("gpt-5.2-2025-12-11", help="Generator model"),
    max_chars_per_passage: int = typer.Option(900, help="Truncate context passages to control cost"),
    use_samples: bool = typer.Option(False, help="Use deterministic sample queries (120)"),
    max_queries: int = typer.Option(0, help="Max queries to process (0 = no cap)"),
    resume: bool = typer.Option(True, help="Resume from answers_partial.jsonl if present"),
    policy: str = typer.Option("none", help="Answering policy: none | tiered_v1"),


    # NEW: confidence gating using rerank scores
    confidence_rerank_run_dir: str = typer.Option("", help="If set, read rerank scores from this run dir to gate answering"),
    conf_threshold_top1: float = typer.Option(-1.0, help="If >=0, only answer when top1 rerank score >= threshold; else abstain"),
    conf_threshold_margin: float = typer.Option(-1.0, help="If >=0, only answer when (top1-top2) >= threshold"),
):
    # Goal: End-to-end RAG generation with citations and optional confidence-gated abstention.
    # Why: Reliability in production means selectively answering when evidence is strong.

    from pathlib import Path
    import json
    from openai import OpenAI
    import subprocess, sys
    from raglab.utils.io import read_jsonl, write_jsonl, write_json
    from raglab.utils.run_artifacts import make_run_dir, write_manifest
    from raglab.data.loaders import load_grounding, load_chunks
    from raglab.rag.generate import generate_answer_structured

    repo_root = Path(__file__).resolve().parents[2]
    run_dir = make_run_dir(repo_root / "runs", run_name)

    gold = load_grounding(repo_root, split=split, use_samples=use_samples)
    chunks = load_chunks(repo_root, use_samples=False, sample_queries_only=False)
    text_by_id = chunk_text_map(chunks)

    input_dir = Path(input_run_dir)
    preds = read_jsonl(input_dir / "predictions.jsonl")
    preds_by_qid = {r["qid"]: r for r in preds}

    # Optional: load confidence scores from rerank run
    conf_by_qid = {}
    if confidence_rerank_run_dir:
        conf_dir = Path(confidence_rerank_run_dir)
        conf_preds = read_jsonl(conf_dir / "predictions.jsonl")
        for r in conf_preds:
            scores = []
            for p in r.get("preds", []):
                try:
                    scores.append(float(p.get("score", 0.0)))
                except Exception:
                    scores.append(0.0)
            top1 = scores[0] if len(scores) >= 1 else 0.0
            top2 = scores[1] if len(scores) >= 2 else 0.0
            conf_by_qid[r["qid"]] = {"top1": top1, "top2": top2, "margin": top1 - top2}


    client = OpenAI()

    partial_path = run_dir / "answers_partial.jsonl"
    done_qids = set()
    answers = []

    if resume and partial_path.exists():
        for r in read_jsonl(partial_path):
            done_qids.add(r["qid"])
            answers.append(r)

    processed = 0
    for ex in gold:
        qid = ex["qid"]
        if qid in done_qids:
            continue

        pred_row = preds_by_qid[qid]
        top = pred_row["preds"][:k_context]
        contexts = [{"chunk_id": p["chunk_id"], "text": text_by_id.get(p["chunk_id"], "")} for p in top]

        # Confidence gate: abstain without calling the generator (saves money)
        gated = False
        conf = conf_by_qid.get(qid, {})
        top1_score = conf.get("top1", None)
        margin = conf.get("margin", None)

        tier = "none"
        if policy == "tiered_v1":
            if (margin is not None) and (margin >= 1):
                tier = "A_high"
            elif (top1_score is not None) and (top1_score >= 3):
                tier = "B_medium"
            else:
                tier = "C_low"


        if confidence_rerank_run_dir:
            if conf_threshold_top1 >= 0:
                if (top1_score is None) or (top1_score < conf_threshold_top1):
                    gated = True
            if conf_threshold_margin >= 0:
                if (margin is None) or (margin < conf_threshold_margin):
                    gated = True

        # Build contexts first
        top = pred_row["preds"][:k_context]
        contexts = [{"chunk_id": p["chunk_id"], "text": text_by_id.get(p["chunk_id"], "")} for p in top]

        def _fallback_snippets(contexts, max_chars=240, top_m=3):
            snaps = []
            for c in contexts[:top_m]:
                t = (c.get("text") or "").strip().replace("\n", " ")
                if len(t) > max_chars:
                    t = t[:max_chars] + "…"
                snaps.append({"chunk_id": c["chunk_id"], "snippet": t})
            return snaps

        # --- Decide tier first (you already have this) ---
        # tier = ...

        # --- Produce output: ALWAYS assign out + fallback_passages ---
        if tier == "C_low":
            fallback_passages = _fallback_snippets(contexts, top_m=3)
            out = {
                "answer": "I can't answer this from the provided context. Here are the most relevant passages I found.",
                "cited_chunk_ids": [],
                "abstained": True,
            }
        else:
            fallback_passages = []
            style = "evidence_first" if tier == "B_medium" else "strict"
            out = generate_answer_structured(
                client=client,
                model=model,
                question=ex["question"],
                contexts=contexts,
                max_chars_per_passage=max_chars_per_passage,
                style=style,
            )


        row = {
            "qid": qid,
            "question": ex["question"],
            "context": [{"chunk_id": c["chunk_id"]} for c in contexts],
            "answer": out["answer"],
            "cited_chunk_ids": out["cited_chunk_ids"],
            "abstained": out["abstained"],
            "fallback_passages": fallback_passages,
            "run_meta": {
                "input_run_dir": str(input_dir),
                "k_context": k_context,
                "model": model,
                "split": split,
                "use_samples": use_samples,
                "max_chars_per_passage": max_chars_per_passage,
                "confidence_rerank_run_dir": confidence_rerank_run_dir or None,
                "conf_threshold_top1": conf_threshold_top1,
                "conf_threshold_margin": conf_threshold_margin,
                "top1_score": top1_score,
                "margin": margin,
                "gated": gated,
                "policy": policy,
                "tier": tier,
            },
        }

        answers.append(row)
        with partial_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        processed += 1
        if max_queries and processed >= max_queries:
            break

    write_json(run_dir / "config.json", {
        "stage": "rag_generation",
        "input_run_dir": str(input_dir),
        "k_context": k_context,
        "model": model,
        "split": split,
        "use_samples": use_samples,
        "max_queries": max_queries,
        "resume": resume,
        "max_chars_per_passage": max_chars_per_passage,
        "confidence_rerank_run_dir": confidence_rerank_run_dir or None,
        "conf_threshold_top1": conf_threshold_top1,
    })
    write_jsonl(run_dir / "answers.jsonl", answers)
    write_manifest(run_dir, extra={"dataset": {"queries_generated": len(answers)}})

    typer.echo(f"Run dir: {run_dir}")
    typer.echo(f"Wrote answers: {len(answers)} (processed this run: {processed})")



@app.command()
def judge_rag(
    run_name: str = typer.Option("rag_judge_v1"),
    input_answers_dir: str = typer.Option(..., help="Run dir with answers.jsonl from run_rag"),
    split: str = typer.Option("validation"),
    judge_model: str = typer.Option("gpt-5.2-2025-12-11", help="Judge model"),
    use_samples: bool = typer.Option(False, help="Use deterministic sample queries (120)"),
    max_queries: int = typer.Option(0, help="Max answers to judge (0 = no cap)"),
    resume: bool = typer.Option(True, help="Resume from judge_partial.jsonl if present"),
):
    # Goal: LLM-as-judge scoring for correctness + groundedness + citations + abstention.
    # Why: Makes the project about reliability outcomes, not just retrieval metrics.

    from pathlib import Path
    import json
    import pandas as pd
    from openai import OpenAI

    from raglab.utils.io import read_jsonl, write_jsonl, write_json
    from raglab.utils.run_artifacts import make_run_dir, write_manifest
    from raglab.data.loaders import load_grounding, load_chunks
    from raglab.rag.judge import judge_answer_structured

    repo_root = Path(__file__).resolve().parents[2]
    run_dir = make_run_dir(repo_root / "runs", run_name)

    gold = load_grounding(repo_root, split=split, use_samples=use_samples)
    gold_by_qid = {g["qid"]: g for g in gold}

    chunks = load_chunks(repo_root, use_samples=False, sample_queries_only=False)
    text_by_id = chunk_text_map(chunks)

    input_dir = Path(input_answers_dir)
    answers = read_jsonl(input_dir / "answers.jsonl")

    client = OpenAI()

    partial_path = run_dir / "judge_partial.jsonl"
    done_qids = set()
    judgments = []

    if resume and partial_path.exists():
        for r in read_jsonl(partial_path):
            done_qids.add(r["qid"])
            judgments.append(r)

    def pick_gold_text(g: dict) -> str:
        # Try common field names; okay if empty.
        for key in ["gold_text", "gold_span_text", "answer_text", "answer", "gold_answer"]:
            if key in g and isinstance(g[key], str) and g[key].strip():
                return g[key].strip()
        return ""

    processed = 0
    for row in answers:
        qid = row["qid"]
        if qid in done_qids:
            continue

        g = gold_by_qid.get(qid, {})
        question = row.get("question", g.get("question", ""))
        answer = row.get("answer", "")
        abstained = bool(row.get("abstained", False))
        cited = row.get("cited_chunk_ids", []) or []

        # Reconstruct full context text for judging
        ctx_ids = [c["chunk_id"] for c in (row.get("context") or [])]
        contexts = [{"chunk_id": cid, "text": text_by_id.get(cid, "")} for cid in ctx_ids]

        gold_text = pick_gold_text(g)

        j = judge_answer_structured(
            client=client,
            model=judge_model,
            question=question,
            answer=answer,
            abstained=abstained,
            cited_chunk_ids=cited,
            contexts=contexts,
            gold_answer_text=gold_text if gold_text else None,
        )
        out = {"qid": qid, "judgment": j}
        judgments.append(out)

        with partial_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

        processed += 1
        if max_queries and processed >= max_queries:
            break

    # Summarize metrics
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    correctness_vals = []
    grounded_vals = []
    citation_vals = []
    abst_vals = []

    for r in judgments:
        j = r["judgment"]
        correctness = int(j.get("correctness", -1))
        if correctness >= 0:
            correctness_vals.append(correctness / 2.0)  # normalize to 0..1
        grounded_vals.append(int(j.get("grounded", 0)))
        citation_vals.append(int(j.get("citation_valid", 0)))
        abst_vals.append(int(j.get("abstention_correct", 0)))

    summary = {
        "n_judged": len(judgments),
        "correctness_mean_0_1": mean(correctness_vals),   # None if no gold text available
        "grounded_rate": mean(grounded_vals),
        "citation_valid_rate": mean(citation_vals),
        "abstention_correct_rate": mean(abst_vals),
        "judge_model": judge_model,
        "input_answers_dir": str(input_dir),
        "split": split,
        "use_samples": use_samples,
    }

    write_json(run_dir / "config.json", {
        "stage": "rag_judging",
        "input_answers_dir": str(input_dir),
        "judge_model": judge_model,
        "split": split,
        "use_samples": use_samples,
        "max_queries": max_queries,
        "resume": resume,
    })
    write_jsonl(run_dir / "judge.jsonl", judgments)
    write_json(run_dir / "judge_metrics.json", summary)
    write_manifest(run_dir, extra={"dataset": {"queries_judged": len(judgments)}})

    typer.echo(f"Run dir: {run_dir}")
    typer.echo(f"Judge metrics: {summary}")


# Goal: CLI wrapper to generate calibration artifacts (per-example + threshold sweep table).
# Why: Portfolio polish: one command produces the key reliability curve data.

@app.command()
def analyze_calibration(
    run_name: str = typer.Option("calibration_rep_v1"),
    rerank_run_dir: str = typer.Option(..., help="Rerank run dir (predictions.jsonl with scores)"),
    rag_gen_dir: str = typer.Option(..., help="RAG generation run dir (answers.jsonl)"),
    judge_dir: str = typer.Option(..., help="Judge run dir (judge.jsonl)"),
):
    from pathlib import Path
    import pandas as pd

    from raglab.utils.run_artifacts import make_run_dir, write_manifest
    from raglab.utils.io import write_json
    from raglab.analysis.calibration import build_per_example_df, make_calibration_tables

    repo_root = Path(__file__).resolve().parents[2]
    run_dir = make_run_dir(repo_root / "runs", run_name)

    df = build_per_example_df(
        rerank_run_dir=Path(rerank_run_dir),
        rag_gen_dir=Path(rag_gen_dir),
        judge_dir=Path(judge_dir),
    )
    per_example, sweep = make_calibration_tables(df)

    per_example_path = run_dir / "per_example.csv"
    sweep_path = run_dir / "calibration_sweep.csv"
    per_example.to_csv(per_example_path, index=False)
    sweep.to_csv(sweep_path, index=False)

    # Pick a couple "nice" operating points to print:
    # - strict: top1>=3
    # - medium: top1>=2
    def row_for(conf_col, threshold):
        sub = sweep[(sweep["conf_col"] == conf_col) & (sweep["threshold"] == threshold)]
        return sub.iloc[0].to_dict() if len(sub) else None

    summary = {
        "n_total": int(len(per_example)),
        "strict_top1_ge_3": row_for("conf_top1", 3),
        "medium_top1_ge_2": row_for("conf_top1", 2),
        "margin_ge_1": row_for("conf_margin", 1.0),
        "margin_ge_2": row_for("conf_margin", 2.0),

        "files": {
            "per_example_csv": str(per_example_path),
            "calibration_sweep_csv": str(sweep_path),
        },
        "inputs": {
            "rerank_run_dir": rerank_run_dir,
            "rag_gen_dir": rag_gen_dir,
            "judge_dir": judge_dir,
        },
    }
    write_json(run_dir / "summary.json", summary)
    write_manifest(run_dir, extra={"dataset": {"rows": int(len(per_example))}})

    typer.echo(f"Run dir: {run_dir}")
    typer.echo(f"Wrote: {per_example_path}")
    typer.echo(f"Wrote: {sweep_path}")
    typer.echo(f"Summary: {summary}")

@app.command()
def analyze_tiers(
    run_name: str = typer.Option("tiers_rep_v1"),
    rag_gen_dir: str = typer.Option(...),
    judge_dir: str = typer.Option(...),
):
    from pathlib import Path
    from raglab.utils.run_artifacts import make_run_dir, write_manifest
    from raglab.analysis.tiers import analyze_tiers as _analyze

    repo_root = Path(__file__).resolve().parents[2]
    run_dir = make_run_dir(repo_root / "runs", run_name)

    df = _analyze(Path(rag_gen_dir), Path(judge_dir))
    out_csv = run_dir / "tier_metrics.csv"
    df.to_csv(out_csv, index=False)

    write_manifest(run_dir, extra={"files": {"tier_metrics_csv": str(out_csv)}})
    typer.echo(f"Run dir: {run_dir}")
    typer.echo(f"Wrote: {out_csv}")
    typer.echo(df.to_string(index=False))

#sys.executable -m raglab ... is portable and avoids relying on raglab.exe
#The glob(f"*__{gen_name}") assumes your run dirs look like YYYYMMDD_HHMMSS__<run_name>. Adjust the glob if your naming differs

@app.command()
def run_tiered_e2e(
    run_name: str = typer.Option("tiered_e2e_v2"),
    input_run_dir: str = typer.Option(..., help="Run dir with predictions.jsonl (e.g. rerank run)"),
    confidence_rerank_run_dir: str = typer.Option(..., help="Run dir with rerank scores (often same as input)"),
    split: str = typer.Option("validation"),
    use_samples: bool = typer.Option(True),
    k_context: int = typer.Option(5),
    gen_model: str = typer.Option("gpt-5-mini"),
    judge_model: str = typer.Option("gpt-5-mini"),
    max_chars_per_passage: int = typer.Option(600),
):
    repo_root = Path(__file__).resolve().parents[2]
    runs_root = repo_root / "runs"

    # We suffix runs so you can find them easily.
    gen_name = f"{run_name}__gen"
    judge_name = f"{run_name}__judge"
    tiers_name = f"{run_name}__tiers"

    # 1) Generate
    subprocess.run(
        [
            sys.executable, "-m", "raglab", "run-rag",
            "--run-name", gen_name,
            "--input-run-dir", input_run_dir,
            "--split", split,
            "--k-context", str(k_context),
            "--confidence-rerank-run-dir", confidence_rerank_run_dir,
            "--policy", "tiered_v1",
            "--model", gen_model,
            "--max-chars-per-passage", str(max_chars_per_passage),
            "--use-samples" if use_samples else "--no-use-samples",
        ],
        check=True,
    )

    # Find the newest run dir matching gen_name
    gen_dirs = sorted(runs_root.glob(f"*__{gen_name}"))
    gen_dir = str(gen_dirs[-1])

    # 2) Judge
    subprocess.run(
        [
            sys.executable, "-m", "raglab", "judge-rag",
            "--run-name", judge_name,
            "--input-answers-dir", gen_dir,
            "--split", split,
            "--judge-model", judge_model,
            "--use-samples" if use_samples else "--no-use-samples",
        ],
        check=True,
    )

    judge_dirs = sorted(runs_root.glob(f"*__{judge_name}"))
    judge_dir = str(judge_dirs[-1])

    # 3) Analyze tiers
    subprocess.run(
        [
            sys.executable, "-m", "raglab", "analyze-tiers",
            "--run-name", tiers_name,
            "--rag-gen-dir", gen_dir,
            "--judge-dir", judge_dir,
        ],
        check=True,
    )

    typer.echo(f"Gen dir:   {gen_dir}")
    typer.echo(f"Judge dir: {judge_dir}")
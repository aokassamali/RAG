# Goal: Join (rerank scores) + (RAG answers) + (judge outcomes) and produce selective-answering curves.
# Why: This turns "RAG quality" into a production-style policy: answer only when confidence is high.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

from raglab.utils.io import read_jsonl


def _load_rerank_scores(preds_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    rows = read_jsonl(preds_path)
    by_qid = {}
    for r in rows:
        # preds: [{"chunk_id": "...", "score": <float or int>}]
        by_qid[r["qid"]] = r["preds"]
    return by_qid


def _load_answers(answers_path: Path) -> Dict[str, Dict[str, Any]]:
    rows = read_jsonl(answers_path)
    return {r["qid"]: r for r in rows}


def _load_judgments(judge_path: Path) -> Dict[str, Dict[str, Any]]:
    rows = read_jsonl(judge_path)
    out = {}
    for r in rows:
        out[r["qid"]] = r["judgment"]
    return out


def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def build_per_example_df(
    rerank_run_dir: Path,
    rag_gen_dir: Path,
    judge_dir: Path,
) -> pd.DataFrame:
    rerank_preds = _load_rerank_scores(rerank_run_dir / "predictions.jsonl")
    answers = _load_answers(rag_gen_dir / "answers.jsonl")
    judgments = _load_judgments(judge_dir / "judge.jsonl")

    qids = sorted(set(rerank_preds.keys()) & set(answers.keys()) & set(judgments.keys()))
    rows = []
    for qid in qids:
        preds = rerank_preds[qid]
        ans = answers[qid]
        j = judgments[qid]

        # Confidence proxies from rerank scores (top-10 list in your rerank run)
        scores = [_safe_float(p.get("score", 0.0)) for p in preds]
        top1 = scores[0] if len(scores) >= 1 else 0.0
        top2 = scores[1] if len(scores) >= 2 else 0.0
        margin = top1 - top2
        count3 = sum(1 for s in scores if s >= 3.0)

        # Outcomes from judge
        correctness_raw = _safe_int(j.get("correctness", -1), default=-1)  # -1 means no-gold
        correctness_01 = None if correctness_raw < 0 else correctness_raw / 2.0
        grounded = _safe_int(j.get("grounded", 0))
        citation_valid = _safe_int(j.get("citation_valid", 0))
        abstention_correct = _safe_int(j.get("abstention_correct", 0))

        rows.append({
            "qid": qid,
            "abstained_model": bool(ans.get("abstained", False)),
            "conf_top1": top1,
            "conf_margin": margin,
            "conf_count3": count3,

            "correctness_01": correctness_01,
            "grounded": grounded,
            "citation_valid": citation_valid,
            "abstention_correct": abstention_correct,
        })

    return pd.DataFrame(rows)


def _sweep_thresholds(
    df: pd.DataFrame,
    conf_col: str,
    thresholds: List[float],
) -> pd.DataFrame:
    """
    We define the policy:
      answer_if = (confidence >= threshold)
    Then compute metrics on answered subset.
    """
    rows = []
    for t in thresholds:
        answered = df[(df[conf_col] >= t) & (~df["abstained_model"])]
        coverage = len(answered) / max(1, len(df))

        # correctness_01 may have Nones if no gold; we compute mean over available
        corr = answered["correctness_01"].dropna()
        correctness_mean = float(corr.mean()) if len(corr) else None

        grounded_rate = float(answered["grounded"].mean()) if len(answered) else None
        citation_rate = float(answered["citation_valid"].mean()) if len(answered) else None

        # "hallucination-ish" proxy among answered = 1 - grounded
        halluc_rate = None if grounded_rate is None else (1.0 - grounded_rate)

        rows.append({
            "conf_col": conf_col,
            "threshold": t,
            "coverage": coverage,
            "correctness_mean_0_1": correctness_mean,
            "grounded_rate": grounded_rate,
            "citation_valid_rate": citation_rate,
            "hallucination_proxy_rate": halluc_rate,
            "n_answered": int(len(answered)),
            "n_total": int(len(df)),
        })

    return pd.DataFrame(rows)


def make_calibration_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - per_example (df)
      - sweep_table (threshold sweeps for multiple conf proxies)
    """
    sweep_frames = []

    # For top1 score, thresholds are natural: 0,1,2,3
    sweep_frames.append(_sweep_thresholds(df, "conf_top1", thresholds=[0, 1, 2, 3]))

    # For margin, thresholds: 0, 0.5, 1, 1.5, 2, 2.5
    sweep_frames.append(_sweep_thresholds(df, "conf_margin", thresholds=[0, 0.5, 1, 1.5, 2, 2.5]))

    # For count3, thresholds: 0..10 (top-10 list)
    sweep_frames.append(_sweep_thresholds(df, "conf_count3", thresholds=list(range(0, 11))))

    sweep = pd.concat(sweep_frames, ignore_index=True)
    return df, sweep

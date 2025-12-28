# Goal: Summarize judge metrics by tier (A/B/C).
# Why: This is the production-grade artifact: reliability by confidence bucket.

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import pandas as pd

from raglab.utils.io import read_jsonl

def analyze_tiers(rag_gen_dir: Path, judge_dir: Path) -> pd.DataFrame:
    answers = {r["qid"]: r for r in read_jsonl(rag_gen_dir / "answers.jsonl")}
    judge = {r["qid"]: r["judgment"] for r in read_jsonl(judge_dir / "judge.jsonl")}

    rows = []
    for qid, a in answers.items():
        j = judge.get(qid)
        if not j:
            continue
        tier = (a.get("run_meta") or {}).get("tier", "unknown")
        abst = bool(a.get("abstained", False))

        correctness_raw = int(j.get("correctness", -1))
        correctness_01 = None if correctness_raw < 0 else correctness_raw / 2.0

        rows.append({
            "qid": qid,
            "tier": tier,
            "abstained": abst,
            "correctness_01": correctness_01,
            "grounded": int(j.get("grounded", 0)),
            "citation_valid": int(j.get("citation_valid", 0)),
            "abstention_correct": int(j.get("abstention_correct", 0)),
        })

    df = pd.DataFrame(rows)

    def summarize(g: pd.DataFrame) -> Dict[str, Any]:
        answered = g[~g["abstained"]]
        corr = answered["correctness_01"].dropna()

        return {
            "n_total": len(g),
            "answered_coverage": len(answered) / max(1, len(g)),
            "correctness_mean_0_1_answered": float(corr.mean()) if len(corr) else None,
            "grounded_rate_answered": float(answered["grounded"].mean()) if len(answered) else None,
            "citation_valid_rate_answered": float(answered["citation_valid"].mean()) if len(answered) else None,
            "halluc_proxy_rate_answered": None if len(answered) == 0 else float(1.0 - answered["grounded"].mean()),
            "abstention_correct_rate_all": float(g["abstention_correct"].mean()) if len(g) else None,
        }

    out = []
    for tier, grp in df.groupby("tier"):
        s = summarize(grp)
        s["tier"] = tier
        out.append(s)

    return pd.DataFrame(out).sort_values("tier")

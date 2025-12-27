# Goal: Retrieval calibration: turn scores into a confidence in "hit@K", compute ECE + coverage/accuracy curve.
# Why: Reliability signal—knowing when retrieval is likely correct is valuable in production RAG.

from __future__ import annotations
import math
from typing import Dict, Any, List, Tuple

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def confidence_from_scores(preds: List[Dict[str, float]]) -> float:
    # Use score gap between top-1 and top-2 as confidence proxy (mapped to [0,1]).
    if not preds:
        return 0.0
    s1 = preds[0]["score"]
    s2 = preds[1]["score"] if len(preds) > 1 else (s1 - 1.0)
    gap = s1 - s2
    return sigmoid(gap)

def ece(conf: List[float], acc: List[float], n_bins: int = 10) -> Dict[str, Any]:
    # Expected Calibration Error with uniform bins over [0,1].
    bins = [(i / n_bins, (i + 1) / n_bins) for i in range(n_bins)]
    total = len(conf)
    ece_val = 0.0
    per_bin = []

    for lo, hi in bins:
        idx = [i for i, c in enumerate(conf) if (c >= lo and (c < hi if hi < 1.0 else c <= hi))]
        if not idx:
            per_bin.append({"bin": [lo, hi], "count": 0, "avg_conf": None, "avg_acc": None})
            continue
        avg_conf = sum(conf[i] for i in idx) / len(idx)
        avg_acc = sum(acc[i] for i in idx) / len(idx)
        w = len(idx) / total
        ece_val += w * abs(avg_acc - avg_conf)
        per_bin.append({"bin": [lo, hi], "count": len(idx), "avg_conf": avg_conf, "avg_acc": avg_acc})

    return {"ece": ece_val, "bins": per_bin, "n_bins": n_bins}

def coverage_accuracy_curve(conf: List[float], acc: List[float], points: int = 20) -> List[Dict[str, Any]]:
    # Sort by confidence descending and compute accuracy at different coverages.
    pairs = sorted(zip(conf, acc), key=lambda x: x[0], reverse=True)
    out = []
    n = len(pairs)
    for p in range(1, points + 1):
        cut = max(1, int(n * (p / points)))
        subset = pairs[:cut]
        cov = cut / n
        a = sum(x[1] for x in subset) / cut
        out.append({"coverage": cov, "accuracy": a})
    return out

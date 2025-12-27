# Goal: Map each gold evidence char-span to the chunk(s) that overlap it.
# Why: Retrieval is evaluated in chunk space (Recall@K, MRR, nDCG), so we need gold_chunk_ids.

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from raglab.utils.io import write_jsonl, ensure_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = REPO_ROOT / "data" / "processed" / "chunks.jsonl"
GM_TRAIN = REPO_ROOT / "data" / "processed" / "grounding_map_train.jsonl"
GM_VAL = REPO_ROOT / "data" / "processed" / "grounding_map_validation.jsonl"

OUT_TRAIN = REPO_ROOT / "data" / "processed" / "grounding_map_train_chunks.jsonl"
OUT_VAL = REPO_ROOT / "data" / "processed" / "grounding_map_validation_chunks.jsonl"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def interval_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    # Overlap length in characters between [a0,a1) and [b0,b1)
    lo = max(a0, b0)
    hi = min(a1, b1)
    return max(0, hi - lo)


def build_chunk_index(chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    # doc_id -> list of chunks sorted by start_char
    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for c in chunks:
        by_doc.setdefault(c["doc_id"], []).append(c)
    for doc_id in by_doc:
        by_doc[doc_id].sort(key=lambda x: x["start_char"])
    return by_doc


def map_one_example(ex: Dict[str, Any], chunks_for_doc: List[Dict[str, Any]]) -> Dict[str, Any]:
    gold_spans = ex["gold_char_spans"]

    gold_chunk_ids_set = set()
    primary_chunk = None
    primary_overlap = -1
    overlap_details: List[Tuple[str, int]] = []

    for span in gold_spans:
        s0, s1 = int(span["start"]), int(span["end"])

        for ch in chunks_for_doc:
            c0, c1 = int(ch["start_char"]), int(ch["end_char"])
            ov = interval_overlap(s0, s1, c0, c1)
            if ov > 0:
                cid = ch["chunk_id"]
                gold_chunk_ids_set.add(cid)
                overlap_details.append((cid, ov))
                if ov > primary_overlap:
                    primary_overlap = ov
                    primary_chunk = cid

    # stable ordering: sort by chunk index embedded in chunk_id
    gold_chunk_ids = sorted(gold_chunk_ids_set)

    ex2 = dict(ex)
    ex2["gold_chunk_ids"] = gold_chunk_ids
    ex2["primary_gold_chunk_id"] = primary_chunk
    ex2["primary_gold_overlap_chars"] = primary_overlap
    # keep only top few overlaps for debugging (avoid huge rows)
    overlap_details.sort(key=lambda t: t[1], reverse=True)
    ex2["overlap_debug_top"] = overlap_details[:10]
    return ex2


def map_split(gm_path: Path, out_path: Path, chunk_index: Dict[str, List[Dict[str, Any]]], split_name: str) -> None:
    gm = read_jsonl(gm_path)
    out = []
    missing_doc = 0
    missing_any_gold_chunk = 0

    for ex in gm:
        doc_id = ex["doc_id"]
        chunks_for_doc = chunk_index.get(doc_id)
        if not chunks_for_doc:
            missing_doc += 1
            continue

        ex2 = map_one_example(ex, chunks_for_doc)
        if not ex2["gold_chunk_ids"]:
            # This would mean gold spans don't overlap any chunk (should not happen if chunk offsets are correct)
            missing_any_gold_chunk += 1
            continue

        out.append(ex2)

    write_jsonl(out_path, out)
    print(f"[{split_name}] wrote {len(out)} rows -> {out_path}")
    print(f"[{split_name}] missing_doc={missing_doc} missing_any_gold_chunk={missing_any_gold_chunk}")


def main() -> None:
    ensure_dir(REPO_ROOT / "data" / "processed")

    # Predict before running:
    # - missing_any_gold_chunk should be 0.
    # - output row counts should be close to original grounding_map counts (maybe a tiny drop if something is off).

    chunks = read_jsonl(CHUNKS_PATH)
    chunk_index = build_chunk_index(chunks)

    map_split(GM_TRAIN, OUT_TRAIN, chunk_index, "train")
    map_split(GM_VAL, OUT_VAL, chunk_index, "validation")


if __name__ == "__main__":
    main()

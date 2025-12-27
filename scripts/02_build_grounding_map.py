# Goal: Convert dialogues + document spans into a grounding_map.jsonl for retrieval evaluation.
# Why: This gives us (question, gold evidence spans) so we can score retrieval (Recall@K, MRR, nDCG) later.

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List

from raglab.utils.io import write_jsonl, ensure_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_PATH = REPO_ROOT / "data" / "processed" / "docs.jsonl"
DTRAIN_PATH = REPO_ROOT / "data" / "processed" / "dialogues_train.jsonl"
DVAL_PATH = REPO_ROOT / "data" / "processed" / "dialogues_validation.jsonl"

OUT_TRAIN = REPO_ROOT / "data" / "processed" / "grounding_map_train.jsonl"
OUT_VAL = REPO_ROOT / "data" / "processed" / "grounding_map_validation.jsonl"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_doc_index(docs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    # doc_id -> doc row (includes spans_by_id)
    return {d["doc_id"]: d for d in docs}


def extract_gold_sp_ids(turn: Dict[str, Any]) -> List[str]:
    # references look like: [{"sp_id": "...", "label": "precondition|solution", ...}, ...]
    refs = turn.get("references") or []
    sp_ids = []
    for r in refs:
        sp = r.get("sp_id")
        if sp:
            sp_ids.append(sp)
    # de-dup but preserve order
    seen = set()
    out = []
    for s in sp_ids:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def build_grounding_rows(dialogues: List[Dict[str, Any]], doc_index: Dict[str, Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    out = []
    skipped_no_prev_user = 0
    skipped_no_gold = 0
    skipped_missing_doc = 0
    skipped_missing_span = 0

    for dial in dialogues:
        doc_id = dial["doc_id"]
        doc = doc_index.get(doc_id)
        if doc is None:
            skipped_missing_doc += 1
            continue

        turns = dial["turns"]
        for i, t in enumerate(turns):
            role = t.get("role")
            if role != "agent":
                continue

            gold_sp_ids = extract_gold_sp_ids(t)
            if not gold_sp_ids:
                skipped_no_gold += 1
                continue

            # find previous user utterance
            if i == 0 or turns[i - 1].get("role") != "user":
                skipped_no_prev_user += 1
                continue

            question = turns[i - 1].get("utterance", "").strip()
            answer = t.get("utterance", "").strip()

            spans_by_id = doc.get("spans_by_id", {})
            gold_char_spans = []
            gold_texts = []

            missing_any = False
            for sp_id in gold_sp_ids:
                sp = spans_by_id.get(sp_id)
                if sp is None:
                    skipped_missing_span += 1
                    missing_any = True
                    break
                start = sp.get("start_sp")
                end = sp.get("end_sp")
                text_sp = sp.get("text_sp", "")
                if start is None or end is None:
                    skipped_missing_span += 1
                    missing_any = True
                    break
                gold_char_spans.append({"start": int(start), "end": int(end)})
                gold_texts.append(text_sp)

            if missing_any:
                continue

            qid = f"{dial['dial_id']}_turn{t.get('turn_id', i)}"

            out.append(
                {
                    "qid": qid,
                    "split": split,
                    "dial_id": dial["dial_id"],
                    "turn_index": i,
                    "doc_id": doc_id,
                    "domain": dial.get("domain", ""),
                    "question": question,
                    "answer": answer,
                    "gold_sp_ids": gold_sp_ids,
                    "gold_char_spans": gold_char_spans,
                    "gold_texts": gold_texts,
                }
            )

    print(f"[{split}] built {len(out)} examples")
    print(f"[{split}] skipped_no_gold={skipped_no_gold} skipped_no_prev_user={skipped_no_prev_user} skipped_missing_doc={skipped_missing_doc} skipped_missing_span={skipped_missing_span}")
    return out


def main() -> None:
    ensure_dir(REPO_ROOT / "data" / "processed")

    # Predict before running:
    # - You should get non-zero grounding examples for both train and validation.
    # - skipped_missing_span should be very low (ideally 0). If it's high, our span schema mismatches.

    docs = read_jsonl(DOCS_PATH)
    doc_index = build_doc_index(docs)

    dtrain = read_jsonl(DTRAIN_PATH)
    dval = read_jsonl(DVAL_PATH)

    train_rows = build_grounding_rows(dtrain, doc_index, split="train")
    val_rows = build_grounding_rows(dval, doc_index, split="validation")

    write_jsonl(OUT_TRAIN, train_rows)
    write_jsonl(OUT_VAL, val_rows)

    print(f"Wrote -> {OUT_TRAIN}")
    print(f"Wrote -> {OUT_VAL}")


if __name__ == "__main__":
    main()

# Goal: Load Doc2Dial from local extracted JSON files robustly (no assumptions about zip folder nesting).
# Why: zip extraction paths can differ (extra top-level folder), and relative paths depend on cwd.

from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, Any, Iterable

def _find_one(root: Path, filename: str) -> Path:
    hits = list(root.rglob(filename))
    if not hits:
        raise FileNotFoundError(f"Could not find {filename} under {root.resolve()}")
    if len(hits) > 1:
        # pick the shortest path (usually the 'real' one) but print options for debugging
        hits = sorted(hits, key=lambda p: len(str(p)))
    return hits[0]

def load_documents_local(extract_dir: str | Path) -> Iterable[Dict[str, Any]]:
    extract_dir = Path(extract_dir)
    doc_path = _find_one(extract_dir, "doc2dial_doc.json")

    # Predict: doc_path should point to .../doc2dial_doc.json and file size should be > 0 bytes.
    payload = json.loads(doc_path.read_text(encoding="utf-8"))
    doc_data = payload["doc_data"]

    for domain, docs in doc_data.items():
        for doc_id, doc in docs.items():
            spans = doc["spans"]
            spans_by_id = (
                {s["id_sp"]: s for s in spans.values()} if isinstance(spans, dict)
                else {s["id_sp"]: s for s in spans}
            )

            yield {
                "doc_id": doc_id,
                "title": doc.get("title", ""),
                "domain": domain,
                "text": doc.get("doc_text", ""),
                "spans_by_id": spans_by_id,
                "source": "doc2dial",
                "split": "train",
                "raw_path": str(doc_path),
            }

def load_dialogues_local(extract_dir: str | Path, split: str) -> Iterable[Dict[str, Any]]:
    extract_dir = Path(extract_dir)
    if split == "train":
        dial_path = _find_one(extract_dir, "doc2dial_dial_train.json")
    elif split in {"validation", "valid", "dev"}:
        dial_path = _find_one(extract_dir, "doc2dial_dial_validation.json")
    else:
        raise ValueError("split must be train or validation")

    payload = json.loads(dial_path.read_text(encoding="utf-8"))
    dial_data = payload["dial_data"]

    for domain, by_doc in dial_data.items():
        for doc_id, dials in by_doc.items():
            for dial in dials:
                yield {
                    "dial_id": dial["dial_id"],
                    "doc_id": doc_id,
                    "domain": domain,
                    "turns": dial["turns"],
                    "source": "doc2dial",
                    "split": "validation" if "validation" in dial_path.name else "train",
                    "raw_path": str(dial_path),
                }


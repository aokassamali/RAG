# Goal: Read docs.jsonl and write chunks.jsonl with stable IDs and char offsets.
# Why: Retrieval indexes chunks, and grounding spans must map into these chunks.

from __future__ import annotations
import json
from pathlib import Path
from raglab.utils.io import write_jsonl, ensure_dir
from raglab.chunking.chunkers import chunk_text

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_PATH = REPO_ROOT / "data" / "processed" / "docs.jsonl"
OUT_CHUNKS = REPO_ROOT / "data" / "processed" / "chunks.jsonl"

TARGET_TOKENS = 350
OVERLAP_TOKENS = 50

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main() -> None:
    ensure_dir(OUT_CHUNKS.parent)

    # Predict: 488 docs should become a few thousand chunks (roughly doc_len/300).
    out = []
    for d in read_jsonl(DOCS_PATH):
        doc_id = d["doc_id"]
        text = d["text"]
        chunks = chunk_text(text, target_tokens=TARGET_TOKENS, overlap_tokens=OVERLAP_TOKENS)
        for j, (s, e, ctext) in enumerate(chunks):
            out.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}::c{j:04d}",
                "start_char": s,
                "end_char": e,
                "text": ctext,
                "target_tokens": TARGET_TOKENS,
                "overlap_tokens": OVERLAP_TOKENS,
            })

    write_jsonl(OUT_CHUNKS, out)
    print(f"Wrote {len(out)} chunks -> {OUT_CHUNKS}")

if __name__ == "__main__":
    main()

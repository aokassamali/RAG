# Goal: Create a deterministic, stratified mini dataset for fast iteration + tests.
# Why: Lets you run the pipeline quickly and catch regressions without full data.

from __future__ import annotations
import json, random
from pathlib import Path
from typing import Dict, Any, List

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "data" / "processed" / "docs.jsonl"
CHUNKS = REPO_ROOT / "data" / "processed" / "chunks.jsonl"
GMV = REPO_ROOT / "data" / "processed" / "grounding_map_validation_chunks.jsonl"

OUT_DIR = REPO_ROOT / "data" / "samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 1337
PER_DOMAIN = 30
EDGECASE_PER_DOMAIN = 5  # requires gold_chunk_ids > 1

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main() -> None:
    rng = random.Random(SEED)

    docs = read_jsonl(DOCS)
    chunks = read_jsonl(CHUNKS)
    gmv = read_jsonl(GMV)

    # group by domain
    by_domain: Dict[str, List[Dict[str, Any]]] = {}
    for ex in gmv:
        by_domain.setdefault(ex.get("domain",""), []).append(ex)

    chosen = []
    for dom, rows in by_domain.items():
        rng.shuffle(rows)

        edge = [r for r in rows if len(r.get("gold_chunk_ids", [])) > 1]
        rng.shuffle(edge)
        edge = edge[:EDGECASE_PER_DOMAIN]

        remaining_needed = max(0, PER_DOMAIN - len(edge))
        rest = [r for r in rows if r not in edge][:remaining_needed]

        chosen.extend(edge + rest)

    # dedupe by qid
    seen = set()
    chosen2 = []
    for r in chosen:
        if r["qid"] not in seen:
            seen.add(r["qid"])
            chosen2.append(r)

    # include only referenced docs/chunks
    doc_ids = {r["doc_id"] for r in chosen2}
    chunk_ids = set()
    for r in chosen2:
        chunk_ids.update(r.get("gold_chunk_ids", []))

    docs_s = [d for d in docs if d["doc_id"] in doc_ids]
    chunks_s = [c for c in chunks if c["chunk_id"] in chunk_ids]

    write_jsonl(OUT_DIR / "mini_docs.jsonl", docs_s)
    write_jsonl(OUT_DIR / "mini_chunks.jsonl", chunks_s)
    write_jsonl(OUT_DIR / "mini_grounding_map_validation_chunks.jsonl", chosen2)

    print(f"Wrote {len(docs_s)} docs, {len(chunks_s)} chunks, {len(chosen2)} queries to {OUT_DIR}")

if __name__ == "__main__":
    main()

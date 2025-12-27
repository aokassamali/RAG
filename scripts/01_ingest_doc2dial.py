# Goal: Run ingestion from anywhere (cwd-independent) by resolving paths relative to repo root.

from pathlib import Path
from raglab.ingestion.doc2dial import load_documents_local, load_dialogues_local
from raglab.utils.io import write_jsonl, ensure_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACT_DIR = REPO_ROOT / "data" / "raw" / "doc2dial" / "extracted"

OUT_DOCS = REPO_ROOT / "data" / "processed" / "docs.jsonl"
OUT_DIALOGUES_TRAIN = REPO_ROOT / "data" / "processed" / "dialogues_train.jsonl"
OUT_DIALOGUES_VALID = REPO_ROOT / "data" / "processed" / "dialogues_validation.jsonl"

def main() -> None:
    ensure_dir(OUT_DOCS.parent)

    docs = list(load_documents_local(EXTRACT_DIR))
    write_jsonl(OUT_DOCS, docs)

    dtrain = list(load_dialogues_local(EXTRACT_DIR, "train"))
    write_jsonl(OUT_DIALOGUES_TRAIN, dtrain)

    dval = list(load_dialogues_local(EXTRACT_DIR, "validation"))
    write_jsonl(OUT_DIALOGUES_VALID, dval)

    print(f"Wrote {len(docs)} docs -> {OUT_DOCS}")
    print(f"Wrote {len(dtrain)} train dialogues -> {OUT_DIALOGUES_TRAIN}")
    print(f"Wrote {len(dval)} validation dialogues -> {OUT_DIALOGUES_VALID}")

if __name__ == "__main__":
    main()

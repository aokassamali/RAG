from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
from raglab.utils.io import read_jsonl

def load_chunks(repo_root: Path, use_samples: bool = False, sample_queries_only: bool = False) -> List[Dict[str, Any]]:
    # If we're using the mini sample ONLY for queries, keep the FULL chunk corpus.
    if use_samples and not sample_queries_only:
        path = repo_root / "data/samples/mini_chunks.jsonl"
    else:
        path = repo_root / "data/processed/chunks.jsonl"
    return read_jsonl(path)

def load_grounding(repo_root: Path, split: str, use_samples: bool = False) -> List[Dict[str, Any]]:
    if use_samples:
        if split != "validation":
            raise ValueError("mini sample currently supports validation only")
        return read_jsonl(repo_root / "data/samples/mini_grounding_map_validation_chunks.jsonl")

    if split == "validation":
        return read_jsonl(repo_root / "data/processed/grounding_map_validation_chunks.jsonl")
    if split == "train":
        return read_jsonl(repo_root / "data/processed/grounding_map_train_chunks.jsonl")
    raise ValueError("split must be train or validation")

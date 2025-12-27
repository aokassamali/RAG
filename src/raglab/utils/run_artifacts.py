# Goal: Create a "run directory" with manifest metadata so results are reproducible and comparable.
from __future__ import annotations
import os, platform, subprocess, sys, time
from pathlib import Path
from typing import Dict, Any
from raglab.utils.io import write_json, ensure_dir

def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"

def make_run_dir(root: str | Path, run_name: str) -> Path:
    root = Path(root)
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{ts}__{run_name}"
    ensure_dir(run_dir)
    return run_dir

def write_manifest(run_dir: Path, extra: Dict[str, Any] | None = None) -> None:
    manifest: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": get_git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "env": {
            "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", ""),
        },
    }
    if extra:
        manifest.update(extra)
    write_json(run_dir / "manifest.json", manifest)

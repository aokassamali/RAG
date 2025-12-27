# Goal: Download the official Doc2Dial v1.0.1 zip and extract it into data/raw/doc2dial/.
# Why: Hugging Face's download_and_extract is failing on Windows for you (missing extracted JSON file),
#      but the dataset itself is a normal zip with JSON files, so we can fetch/unzip deterministically.

from pathlib import Path
import zipfile
import requests

URL = "https://doc2dial.github.io/file/doc2dial_v1.0.1.zip"  # same URL used by the HF dataset script :contentReference[oaicite:1]{index=1}

RAW_DIR = Path("data/raw/doc2dial")
ZIP_PATH = RAW_DIR / "doc2dial_v1.0.1.zip"
EXTRACT_DIR = RAW_DIR / "extracted"

NEEDED = [
    EXTRACT_DIR / "doc2dial" / "v1.0.1" / "doc2dial_doc.json",
    EXTRACT_DIR / "doc2dial" / "v1.0.1" / "doc2dial_dial_train.json",
    EXTRACT_DIR / "doc2dial" / "v1.0.1" / "doc2dial_dial_validation.json",
]

def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

def extract(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)

def main() -> None:
    # Predict before you run:
    # - You should end up with data/raw/doc2dial/extracted/doc2dial/v1.0.1/
    # - And the 3 JSON files in NEEDED should exist. (Those are the canonical filenames for Doc2Dial.) :contentReference[oaicite:2]{index=2}

    if not ZIP_PATH.exists():
        print(f"Downloading -> {ZIP_PATH}")
        download(URL, ZIP_PATH)
    else:
        print(f"Zip already exists -> {ZIP_PATH}")

    print(f"Extracting -> {EXTRACT_DIR}")
    extract(ZIP_PATH, EXTRACT_DIR)

    missing = [p for p in NEEDED if not p.exists()]
    if missing:
        print("ERROR: Missing expected files after extraction:")
        for p in missing:
            print("  -", p)
        raise SystemExit(1)

    print("OK: Doc2Dial files present:")
    for p in NEEDED:
        print("  -", p)

if __name__ == "__main__":
    main()

"""Find the dataset zips and extract them into the work folder.

Usage:
  python organize_data.py [data_dir]

The zips are found by name pattern, so the exact file names do not matter.
Expected datasets:
  Bangla LPDB-A         vehicle images plus cropped plates with YOLO labels
  NightLPD-BD           night images with YOLO labels (a zip inside a zip)
  Plate crops 2.5k      optional, has NO labels, used only as an unlabeled check set

Extraction is safe against path tricks in zip files and is skipped when a
dataset was already extracted, so it can be re-run.
"""
import re
import sys
import zipfile
from pathlib import Path

import config

# (key, regex on the lower case zip name). The night pattern is tested first.
PATTERNS = [
    ("night", re.compile(r"night")),
    ("lpdb", re.compile(r"lpdb")),
    ("crops", re.compile(r"licen[cs]e.*plate|plate.*crop|2\.?5k")),
]
REQUIRED = ("lpdb", "night")


def find_zips(data_dir):
    found = {}
    for p in sorted(Path(data_dir).glob("*.zip")):
        name = p.name.lower()
        for key, rx in PATTERNS:
            if rx.search(name):
                found.setdefault(key, p)
                break
    return found


def safe_extract(zip_path, dest):
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            target = (dest / info.filename).resolve()
            if dest != target and dest not in target.parents:
                raise RuntimeError("Unsafe path in zip: %s" % info.filename)
        zf.extractall(dest)


def extract_nested(folder, depth=2):
    """Extract zips found inside an extracted folder, next to themselves."""
    if depth <= 0:
        return
    for inner in sorted(Path(folder).rglob("*.zip")):
        target = inner.parent / (inner.stem + "_unzipped")
        if target.exists():
            continue
        print("  nested zip:", inner.name)
        safe_extract(inner, target)
        extract_nested(target, depth - 1)


def organize(data_dir=None):
    data_dir = Path(data_dir or config.DATA_DIR)
    if not data_dir.is_dir():
        raise SystemExit("Data folder not found: %s" % data_dir)
    zips = find_zips(data_dir)
    print("Data folder:", data_dir)
    for key in ("lpdb", "night", "crops"):
        print("  %-6s %s" % (key, zips[key].name if key in zips else "not found"))
    missing = [k for k in REQUIRED if k not in zips]
    if missing:
        raise SystemExit("Missing required zip(s): %s" % ", ".join(missing))

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for key, zp in zips.items():
        dest = config.RAW_DIR / key
        marker = dest / ".extracted"
        if marker.exists():
            print("Already extracted:", key)
            continue
        print("Extracting %s (%.0f MB) ..." % (zp.name, zp.stat().st_size / 1e6))
        safe_extract(zp, dest)
        extract_nested(dest)
        marker.write_text(zp.name, encoding="utf-8")
    print("Done. Raw data in", config.RAW_DIR)
    return zips


if __name__ == "__main__":
    organize(sys.argv[1] if len(sys.argv) > 1 else None)

"""Find the dataset archives (.zip and .tar) and extract them into the work folder.

Usage:
  python organize_data.py [data_dir]

The archives are found by name pattern, so the exact file names do not matter.
Expected datasets:
  Bangla LPDB-A         vehicle images plus cropped plates with YOLO labels (zip)
  NightLPD-BD           night images with YOLO labels (a zip inside a zip)
  DatasetNinja          character level plate images, Supervisely format (tar), optional
  Plate crops 2.5k      optional, has NO labels, used only as an unlabeled check set

Extraction is safe against path tricks in archives, each dataset goes into its
own folder, and it is skipped when a dataset was already extracted, so the
script can be re-run.
"""
import re
import sys
import tarfile
import zipfile
from pathlib import Path

import config

# (key, regex on the lower case file name). Order matters: the first match wins.
# ninja comes before crops because the Ninja file name also contains
# 'license-plate'.
PATTERNS = [
    ("night", re.compile(r"night")),
    ("ninja", re.compile(r"ninja|character-recognition|character recognition")),
    ("lpdb", re.compile(r"lpdb")),
    ("crops", re.compile(r"licen[cs]e.*plate|plate.*crop|2\.?5k")),
]
REQUIRED = ("lpdb", "night")
ARCHIVE_SUFFIXES = (".zip", ".tar")


def find_archives(data_dir):
    found = {}
    for p in sorted(Path(data_dir).iterdir()):
        if p.suffix.lower() not in ARCHIVE_SUFFIXES:
            continue
        name = p.name.lower()
        for key, rx in PATTERNS:
            if rx.search(name):
                found.setdefault(key, p)
                break
    return found


def _check_inside(dest, name):
    target = (dest / name).resolve()
    if dest != target and dest not in target.parents:
        raise RuntimeError("Unsafe path in archive: %s" % name)


def safe_extract(archive, dest):
    """Extract a .zip or .tar into dest, refusing paths that escape dest."""
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    archive = Path(archive)
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                _check_inside(dest, info.filename)
            zf.extractall(dest)
    else:
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                _check_inside(dest, member.name)
                if member.issym() or member.islnk() or member.isdev():
                    raise RuntimeError("Unsafe member type in archive: %s" % member.name)
            try:
                tf.extractall(dest, filter="data")  # Python 3.12 and newer (also patched older versions)
            except TypeError:
                tf.extractall(dest)  # older Python, members were already checked above


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
    found = find_archives(data_dir)
    print("Data folder:", data_dir)
    for key in ("lpdb", "night", "ninja", "crops"):
        print("  %-6s %s" % (key, found[key].name if key in found else "not found"))
    missing = [k for k in REQUIRED if k not in found]
    if missing:
        raise SystemExit("Missing required archive(s): %s" % ", ".join(missing))
    if "ninja" not in found:
        print("  note: no DatasetNinja tar found, stage 2 will use LPDB-A only")

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for key, path in found.items():
        dest = config.RAW_DIR / key
        marker = dest / ".extracted"
        if marker.exists():
            print("Already extracted:", key)
            continue
        print("Extracting %s (%.0f MB) ..." % (path.name, path.stat().st_size / 1e6))
        safe_extract(path, dest)
        extract_nested(dest)
        marker.write_text(path.name, encoding="utf-8")
    print("Done. Raw data in", config.RAW_DIR)
    return found


if __name__ == "__main__":
    organize(sys.argv[1] if len(sys.argv) > 1 else None)

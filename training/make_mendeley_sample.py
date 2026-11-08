"""Rebuild the held out test photos from the public Mendeley dataset.

Dataset: "Vehicle License Plate Detection Dataset V1", Mendeley Data,
DOI 10.17632/nntnnzffw4, license CC BY 4.0. 2,947 photos of Dhaka plates, 640 x 640,
no labels, none of them used for training.

Usage:
  python make_mendeley_sample.py DOWNLOADED_ZIP OUT_DIR

Steps:
  1. Extract the nested zips and keep only the .jpg files, with a path check.
  2. Draw the fixed sample of 300 photos (seed 7): 120 private vehicles, 90 trucks and
     pickups, 40 buses, 30 bikes, 20 CNGs. Copied to OUT_DIR/sample300.
  3. Copy the 64 hand labeled photos listed in eval_data/mendeley_v1_truth64.csv, with
     that file as truth.csv, to OUT_DIR/scored64.

Then score them: python pipeline_eval.py OUT_DIR/scored64 (PC, needs ai-edge-litert and
pillow) or python phone_selftest.py OUT_DIR/scored64 (a USB phone with the app).
The labels are only for plates a person could read, so scores are slightly optimistic.
"""
import csv
import os
import random
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

QUOTA = {"Private Vehicles": 120, "Trucks & Pickups": 90, "Buses": 40, "Bikes": 30, "Cngs": 20}
HERE = Path(__file__).resolve().parent


def safe_extract(z, dest, keep):
    """Extract the entries that pass keep(), refusing absolute paths and .. parts."""
    os.makedirs(dest, exist_ok=True)
    for info in z.infolist():
        if not keep(info.filename):
            continue
        name = Path(info.filename)
        if name.is_absolute() or ".." in name.parts:
            raise SystemExit("Unsafe path in zip: " + info.filename)
        z.extract(info, dest)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    outer, out = Path(sys.argv[1]), Path(sys.argv[2])
    work = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(outer) as z:
        safe_extract(z, work, lambda n: n.lower().endswith(".zip"))
    inner = next(work.rglob("*.zip"))
    by_type = {}
    with zipfile.ZipFile(inner) as z:
        safe_extract(z, work / "photos", lambda n: n.lower().endswith(".jpg") and not n.startswith("__MACOSX"))
    for p in (work / "photos").rglob("*.jpg"):
        by_type.setdefault(p.parent.name.strip(), []).append(p)

    sample = out / "sample300"
    sample.mkdir(parents=True, exist_ok=True)
    random.seed(7)
    for kind, count in QUOTA.items():
        for p in random.sample(sorted(by_type[kind], key=lambda q: q.name), count):
            shutil.copy2(p, sample / (kind.split()[0].lower() + "_" + p.name))
    print("sample300:", len(list(sample.iterdir())), "photos")

    scored = out / "scored64"
    scored.mkdir(parents=True, exist_ok=True)
    truth = HERE / "eval_data" / "mendeley_v1_truth64.csv"
    with open(truth, encoding="utf-8", newline="") as f:
        names = [r["file"] for r in csv.DictReader(f)]
    for name in names:
        shutil.copy2(sample / name, scored / name)
    shutil.copy2(truth, scored / "truth.csv")
    print("scored64:", len(names), "photos with truth.csv in", scored)
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()

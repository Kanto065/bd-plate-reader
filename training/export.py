"""Export trained models to TFLite and copy them to models/.

Usage:
  python export.py            both stages
  python export.py plate      one stage
  python export.py chars

Run this on Colab or Linux. The TFLite export pulls in TensorFlow and related
packages, which are heavy and awkward on Windows. Ultralytics installs them on
first use. The calibration images for int8 come from the val split listed in
the stage data.yaml.

Set BD_EXPORT_DTYPE to choose the precision (default int8):
  int8     full int8, about 3 MB per model. The activations are int8 too, which
           squeezes the confidence scores into a few levels (the plate score
           tops out at 0.50 on every photo).
  dynamic  int8 weights, float activations (Ultralytics quantize w8a32). About
           the same size as int8, and the scores keep their precision.
  fp32     full float, the accuracy reference. About 10 MB per model, so about
           21 MB for both, over the 15 MB target.
The newer Ultralytics LiteRT exporter has no fp16 option.

Output names (copied to models/ and also to OUT_DIR/models for safekeeping):
  plate_detector_int8.tflite    char_detector_int8.tflite
  plate_detector_dynamic.tflite char_detector_dynamic.tflite
  plate_detector_fp32.tflite    char_detector_fp32.tflite
Only int8 uses the names the app loads. To try another mode in the app, copy its
files over the int8 names.
"""
import os
import shutil
import sys
import time
from pathlib import Path

import config

TARGET_TOTAL_MB = 15.0

MODE = os.environ.get("BD_EXPORT_DTYPE", "int8").lower()
if MODE not in ("int8", "dynamic", "fp32"):
    raise SystemExit("BD_EXPORT_DTYPE must be int8, dynamic or fp32, got %r" % MODE)


def find_tflite(weights, returned=None, started=0.0):
    """Locate the .tflite that this export just wrote.

    Newer Ultralytics (LiteRT exporter) writes next to the weights, older
    versions wrote into <name>_saved_model/. Check the path returned by
    export() first, then both folders. For int8 the file name has "int8" in it.
    For the other modes take the newest .tflite written since the export began,
    skipping int8 files.
    """
    if returned:
        r = Path(str(returned))
        if r.is_file() and r.suffix == ".tflite":
            return r
    folders = [weights.parent, weights.parent / (weights.stem + "_saved_model")]
    candidates = []
    for folder in folders:
        if folder.is_dir():
            candidates += sorted(folder.glob("*.tflite"))
    if MODE == "int8":
        candidates = [c for c in candidates if "int8" in c.name]
    else:
        candidates = [c for c in candidates if "int8" not in c.name and c.stat().st_mtime >= started - 2]
        candidates.sort(key=lambda c: c.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit("No %s .tflite found in %s after export" % (MODE, " or ".join(str(f) for f in folders)))
    return candidates[0]


def export_stage(stage):
    from ultralytics import YOLO

    s = config.stage_settings(stage)
    weights = config.best_weights(stage)
    if not weights.exists():
        raise SystemExit("Missing %s. Train this stage first." % weights)
    print("Exporting %s (imgsz %d) to %s TFLite ..." % (s["name"], s["imgsz"], MODE))
    started = time.time()
    model = YOLO(str(weights))
    if MODE == "int8":
        returned = model.export(
            format="tflite",
            int8=True,
            imgsz=s["imgsz"],
            data=str(config.data_yaml(stage)),
            batch=1,
        )
    else:
        # quantize 32 is float, "w8a32" is int8 weights with float activations.
        returned = model.export(
            format="litert",
            quantize=32 if MODE == "fp32" else "w8a32",
            imgsz=s["imgsz"],
            batch=1,
        )
    src = find_tflite(weights, returned, started)
    name = s["export_name"].replace("_int8", "_" + MODE)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUT_DIR / "models").mkdir(parents=True, exist_ok=True)
    for dest_dir in (config.MODELS_DIR, config.OUT_DIR / "models"):
        shutil.copy2(src, dest_dir / name)
    size_mb = src.stat().st_size / 1e6
    print("  %s: %.2f MB (from %s)" % (name, size_mb, src.name))
    return size_mb


def main(stages):
    sizes = {st: export_stage(st) for st in stages}
    total = sum(sizes.values())
    print("\nTotal size of exported models: %.2f MB (target under %.0f MB)" % (total, TARGET_TOTAL_MB))
    if len(stages) == 2 and total > TARGET_TOTAL_MB:
        print("WARNING: over the size target.")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args if args else ["plate", "chars"])

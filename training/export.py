"""Export trained models to int8 TFLite and copy them to models/.

Usage:
  python export.py            both stages
  python export.py plate      one stage
  python export.py chars

Set BD_EXPORT_DTYPE=fp16 to export half precision instead. The int8 export
squeezes the confidence scores into a few levels (the plate score tops out at
0.50 on every photo), so fp16 is the way to compare. It is about twice the size
(roughly 12 MB for both stages, still under the 15 MB target) and writes
plate_detector_fp16.tflite and char_detector_fp16.tflite, so it never replaces
the int8 files. To try one in the app, copy the fp16 files over the int8 names.

Run this on Colab or Linux. The TFLite export pulls in TensorFlow and related
packages, which are heavy and awkward on Windows. Ultralytics installs them on
first use. The calibration images for int8 come from the val split listed in
the stage data.yaml.

Output names (copied to models/ and also to OUT_DIR/models for safekeeping):
  plate_detector_int8.tflite
  char_detector_int8.tflite
"""
import os
import shutil
import sys

import config

TARGET_TOTAL_MB = 15.0


FP16 = os.environ.get("BD_EXPORT_DTYPE", "int8").lower() == "fp16"


def find_tflite(weights):
    folder = weights.parent / (weights.stem + "_saved_model")
    kind = "float16" if FP16 else "int8"
    candidates = sorted(folder.glob("*%s*.tflite" % kind)) if folder.is_dir() else []
    if not candidates:
        raise SystemExit("No %s .tflite found in %s after export" % (kind, folder))
    return candidates[0]


def export_stage(stage):
    from ultralytics import YOLO

    s = config.stage_settings(stage)
    weights = config.best_weights(stage)
    if not weights.exists():
        raise SystemExit("Missing %s. Train this stage first." % weights)
    print("Exporting %s (imgsz %d) to %s TFLite ..." % (s["name"], s["imgsz"], "fp16" if FP16 else "int8"))
    YOLO(str(weights)).export(
        format="tflite",
        int8=not FP16,
        half=FP16,
        imgsz=s["imgsz"],
        data=str(config.data_yaml(stage)),
        batch=1,
    )
    src = find_tflite(weights)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUT_DIR / "models").mkdir(parents=True, exist_ok=True)
    name = s["export_name"].replace("_int8", "_fp16") if FP16 else s["export_name"]
    for dest_dir in (config.MODELS_DIR, config.OUT_DIR / "models"):
        shutil.copy2(src, dest_dir / name)
    size_mb = src.stat().st_size / 1e6
    print("  %s: %.2f MB" % (name, size_mb))
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

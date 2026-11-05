"""Export trained models to int8 TFLite and copy them to models/.

Usage:
  python export.py            both stages
  python export.py plate      one stage
  python export.py chars

Run this on Colab or Linux. The TFLite export pulls in TensorFlow and related
packages, which are heavy and awkward on Windows. Ultralytics installs them on
first use. The calibration images for int8 come from the val split listed in
the stage data.yaml.

Output names (copied to models/ and also to OUT_DIR/models for safekeeping):
  plate_detector_int8.tflite
  char_detector_int8.tflite
"""
import shutil
import sys

import config

TARGET_TOTAL_MB = 15.0


def find_tflite(weights):
    folder = weights.parent / (weights.stem + "_saved_model")
    candidates = sorted(folder.glob("*int8*.tflite")) if folder.is_dir() else []
    if not candidates:
        raise SystemExit("No int8 .tflite found in %s after export" % folder)
    return candidates[0]


def export_stage(stage):
    from ultralytics import YOLO

    s = config.stage_settings(stage)
    weights = config.best_weights(stage)
    if not weights.exists():
        raise SystemExit("Missing %s. Train this stage first." % weights)
    print("Exporting %s (imgsz %d) to int8 TFLite ..." % (s["name"], s["imgsz"]))
    YOLO(str(weights)).export(
        format="tflite",
        int8=True,
        imgsz=s["imgsz"],
        data=str(config.data_yaml(stage)),
        batch=1,
    )
    src = find_tflite(weights)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUT_DIR / "models").mkdir(parents=True, exist_ok=True)
    for dest_dir in (config.MODELS_DIR, config.OUT_DIR / "models"):
        shutil.copy2(src, dest_dir / s["export_name"])
    size_mb = src.stat().st_size / 1e6
    print("  %s: %.2f MB" % (s["export_name"], size_mb))
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

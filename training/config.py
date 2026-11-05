"""Shared settings for the BD Plate Reader training scripts.

Everything can be overridden with environment variables, so the same scripts
run on Colab (Drive paths) and on a local machine.

Environment variables:
  BD_DATA_DIR  folder holding the dataset zips
  BD_WORK_DIR  scratch folder for extracted and merged datasets
  BD_OUT_DIR   folder for training runs, weights and exports
  BD_EPOCHS    override epochs for both stages (useful for smoke tests)
  BD_BATCH     override batch size for both stages
  BD_FRACTION  fraction of the training set to use (smoke tests only)
  BD_DEVICE    training device, for example 0 or cpu (default: auto)
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

IN_COLAB = (
    "google.colab" in sys.modules
    or "COLAB_RELEASE_TAG" in os.environ
    or os.path.isdir("/content/drive/MyDrive")
)

DRIVE_BASE = Path("/content/drive/MyDrive/bd-plate-reader")

DATA_DIR = Path(
    os.environ.get("BD_DATA_DIR")
    or (DRIVE_BASE / "data" if IN_COLAB else ROOT / "data" / "zips")
)
WORK_DIR = Path(
    os.environ.get("BD_WORK_DIR")
    or ("/content/work" if IN_COLAB else ROOT / "data" / "work")
)
OUT_DIR = Path(
    os.environ.get("BD_OUT_DIR")
    or (DRIVE_BASE / "outputs" if IN_COLAB else ROOT / "runs")
)
MODELS_DIR = REPO / "models"

RAW_DIR = WORK_DIR / "raw"
DATASETS_DIR = WORK_DIR / "datasets"

SEED = 42
SPLIT = (0.8, 0.1, 0.1)  # train, val, test

# Per stage settings. The character model uses a smaller input size because the
# input is already a tight plate crop (see README for the reasoning).
STAGES = {
    "plate": {
        "name": "plate_detector",
        "dataset": "plate_detector",
        "imgsz": 640,
        "epochs": 100,
        "batch": 16,
        "patience": 30,
        "export_name": "plate_detector_int8.tflite",
    },
    "chars": {
        "name": "char_detector",
        "dataset": "char_detector",
        "imgsz": 416,
        "epochs": 150,
        "batch": 32,
        "patience": 40,
        "export_name": "char_detector_int8.tflite",
    },
}

PRETRAINED = "yolo11n.pt"


def stage_settings(stage):
    """Return the settings for a stage, with environment overrides applied."""
    if stage not in STAGES:
        raise SystemExit("Unknown stage '%s'. Use one of: %s" % (stage, ", ".join(STAGES)))
    s = dict(STAGES[stage])
    if os.environ.get("BD_EPOCHS"):
        s["epochs"] = int(os.environ["BD_EPOCHS"])
    if os.environ.get("BD_BATCH"):
        s["batch"] = int(os.environ["BD_BATCH"])
    return s


def data_yaml(stage):
    return DATASETS_DIR / STAGES[stage]["dataset"] / "data.yaml"


def best_weights(stage):
    return OUT_DIR / STAGES[stage]["name"] / "weights" / "best.pt"

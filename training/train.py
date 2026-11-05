"""Train one stage with Ultralytics YOLO11n.

Usage:
  python train.py plate     stage 1, plate detector, imgsz 640
  python train.py chars     stage 2, character detector, imgsz 416

Weights and logs go to OUT_DIR/<stage name>/ (Drive on Colab). If a run was
interrupted, running the same command again resumes from last.pt. If the run
already finished, it is skipped.

Why no horizontal flip: Bangla letters and digits are not mirror symmetric, so a
flipped plate teaches the model glyphs that never occur. Flips are switched off
for both stages. Vertical flips are off too.

Why imgsz 416 for characters: the input is already a tight crop of a plate, a
few hundred pixels wide, with up to about 11 characters. At 416 pixels the
smallest character is still roughly 30 pixels wide, enough for the model to tell
similar letters apart. 320 loses stroke detail, 640 only adds cost on a phone.
"""
import os
import sys

import config

AUGMENT = {
    "plate": dict(
        fliplr=0.0, flipud=0.0, mosaic=1.0, close_mosaic=10, degrees=5.0,
        translate=0.1, scale=0.4, shear=2.0, perspective=0.0005,
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.5, mixup=0.0,
    ),
    "chars": dict(
        fliplr=0.0, flipud=0.0, mosaic=0.3, close_mosaic=20, degrees=3.0,
        translate=0.05, scale=0.3, shear=2.0, perspective=0.0003,
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.5, mixup=0.0,
    ),
}


def run_finished(last):
    """Ultralytics strips the optimizer and sets epoch to -1 when training completes."""
    import torch

    ckpt = torch.load(str(last), map_location="cpu", weights_only=False)
    return ckpt.get("epoch", 0) == -1 or ckpt.get("optimizer") is None


def train(stage):
    from ultralytics import YOLO

    s = config.stage_settings(stage)
    data = config.data_yaml(stage)
    if not data.exists():
        raise SystemExit("Missing %s. Run organize_data.py and merge_datasets.py first." % data)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    last = config.OUT_DIR / s["name"] / "weights" / "last.pt"

    if last.exists():
        if run_finished(last):
            print("Stage '%s' already finished: %s" % (stage, config.best_weights(stage)))
            return config.best_weights(stage)
        print("Resuming interrupted run from", last)
        YOLO(str(last)).train(resume=True)
        return config.best_weights(stage)

    extra = {}
    if os.environ.get("BD_FRACTION"):
        extra["fraction"] = float(os.environ["BD_FRACTION"])
    if os.environ.get("BD_DEVICE"):
        extra["device"] = os.environ["BD_DEVICE"]
    model = YOLO(config.PRETRAINED)
    model.train(
        data=str(data),
        epochs=s["epochs"],
        imgsz=s["imgsz"],
        batch=s["batch"],
        patience=s["patience"],
        project=str(config.OUT_DIR),
        name=s["name"],
        exist_ok=True,
        workers=2 if config.IN_COLAB else 4,
        seed=config.SEED,
        plots=True,
        **AUGMENT[stage],
        **extra,
    )
    print("Done. Best weights:", config.best_weights(stage))
    return config.best_weights(stage)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python train.py plate|chars")
    train(sys.argv[1])

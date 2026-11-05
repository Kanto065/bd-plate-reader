"""Evaluate both models on the test split.

Usage:
  python evaluate.py [plate_weights] [char_weights]

With no arguments the best.pt files from OUT_DIR are used. You can also pass
exported .tflite files to check the quantized models.

Reports:
  1. Detection mAP50 and mAP50-95 for the plate detector and the character
     detector (Ultralytics validation on the test split). The character detector
     has two independent held out sets, reported separately: the LPDB-A test
     split, and the DatasetNinja photos held out by source photo. Ninja is a
     different distribution, so its number is the honest generalization check.
  2. Plate level exact match for the character stage, for both held out sets:
     run the character model on each test plate crop, rebuild the plate with
     postprocess.py, and compare the character sequence with the ground truth
     rebuilt from the label file. Also prints the mean character error rate
     (edit distance over length).
  3. Optional full pipeline run (plate detector, crop, character model) on a
     folder of vehicle photos given by BD_PIPELINE_DIR. If the folder contains
     truth.csv with the columns file,text (text in the same English token form
     as the model output, see README) the exact match is computed, otherwise a
     predictions.csv is written for manual review.
  4. Optional quick look at unlabeled plate crops (the 2.5k dataset) if it was
     extracted. Predictions are written to a csv for manual review.

Note on end to end accuracy: the vehicle photos in LPDB-A carry no plate text,
so true two stage exact match needs your own labeled photos (truth.csv). The
crop level exact match in step 2 is the best number available from the dataset.
"""
import csv
import json
import os
import sys
from pathlib import Path

import config
import postprocess as pp

CONF = 0.25
IOU = 0.5
PAD = 0.05
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def load_names():
    return pp.load_class_names(config.ROOT / "classes.txt")


def read_gt(label_path):
    dets = []
    for line in Path(label_path).read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) == 5:
            dets.append(pp.Detection(int(p[0]), *[float(v) for v in p[1:]]))
    return dets


def predict_dets(model, image, imgsz):
    r = model.predict(str(image), imgsz=imgsz, conf=CONF, iou=IOU, agnostic_nms=True, verbose=False)[0]
    out = []
    for (x, y, w, h), c, p in zip(r.boxes.xywhn.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        out.append(pp.Detection(int(c), x, y, w, h, p))
    return out


def map_scores(model, stage, yaml_path=None):
    """mAP on the 'test' entry of a data yaml (default: the stage data.yaml)."""
    s = config.STAGES[stage]
    batch = 1 if str(model.ckpt_path or "").endswith(".tflite") else 16
    data = str(yaml_path or config.data_yaml(stage))
    r = model.val(data=data, split="test", imgsz=s["imgsz"], batch=batch, plots=False, verbose=False)
    return {"mAP50": float(r.box.map50), "mAP50_95": float(r.box.map)}


def crop_exact_match(model, names, part="test"):
    """Plate level exact match on a held out folder of the character dataset.

    part is 'test' (the LPDB-A test split) or 'test_ninja' (the DatasetNinja
    photos held out, split by source photo).
    """
    s = config.STAGES["chars"]
    root = config.DATASETS_DIR / "char_detector"
    images = sorted(p for p in (root / "images" / part).iterdir() if p.suffix.lower() in IMG_EXTS)
    exact, cer_sum, wrong = 0, 0.0, []
    for img in images:
        gt = pp.plate_tokens(read_gt(root / "labels" / part / (img.stem + ".txt")), names)
        pred = pp.plate_tokens(predict_dets(model, img, s["imgsz"]), names)
        if gt == pred:
            exact += 1
        else:
            wrong.append((img.name, " ".join(gt), " ".join(pred)))
        cer_sum += pp.edit_distance(gt, pred) / max(1, len(gt))
    n = max(1, len(images))
    for name, g, p in wrong[:10]:
        print("  mismatch %s\n    truth: %s\n    pred : %s" % (name, g, p))
    return {"plates": len(images), "exact_match": exact / n, "mean_char_error_rate": cer_sum / n}


def crop_with_padding(image_path, box):
    import cv2

    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    x, y, bw, bh = box
    x0 = max(0, int((x - bw / 2 - PAD) * w))
    x1 = min(w, int((x + bw / 2 + PAD) * w))
    y0 = max(0, int((y - bh / 2 - PAD) * h))
    y1 = min(h, int((y + bh / 2 + PAD) * h))
    return img[y0:y1, x0:x1]


def run_pipeline(folder, plate_model, char_model, names, display):
    import cv2

    folder = Path(folder)
    truth = {}
    tp = folder / "truth.csv"
    if tp.exists():
        with open(tp, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                truth[row["file"]] = row["text"].strip()
    rows, exact = [], 0
    tmp = config.WORK_DIR / "pipeline_crop.png"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    for img in sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS):
        found = predict_dets(plate_model, img, config.STAGES["plate"]["imgsz"])
        if not found:
            rows.append((img.name, "", "", ""))
            continue
        best = max(found, key=lambda d: d.conf)
        crop = crop_with_padding(img, (best.x, best.y, best.w, best.h))
        cv2.imwrite(str(tmp), crop)
        dets = predict_dets(char_model, tmp, config.STAGES["chars"]["imgsz"])
        tokens = " ".join(pp.plate_tokens(dets, names))
        text = pp.read_plate(dets, names, display)
        rows.append((img.name, tokens, text, truth.get(img.name, "")))
        if img.name in truth and truth[img.name] == tokens:
            exact += 1
    out = config.OUT_DIR / "predictions.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "tokens", "text", "truth"])
        w.writerows(rows)
    print("  wrote", out)
    if truth:
        n = sum(1 for r in rows if r[0] in truth)
        return {"labeled_photos": n, "exact_match": exact / max(1, n)}
    return {"photos": len(rows), "note": "no truth.csv, predictions only"}


def check_unlabeled_crops(char_model, names, display, limit=50):
    base = config.RAW_DIR / "crops"
    if not base.is_dir():
        return None
    files = sorted(p for p in base.rglob("*") if p.suffix.lower() in IMG_EXTS and p.parent.name == "testing")[:limit]
    if not files:
        return None
    out = config.OUT_DIR / "crops_check.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "tokens", "text"])
        for p in files:
            dets = predict_dets(char_model, p, config.STAGES["chars"]["imgsz"])
            w.writerow([p.name, " ".join(pp.plate_tokens(dets, names)), pp.read_plate(dets, names, display)])
    print("  wrote", out, "(", len(files), "unlabeled crops, check by eye )")
    return str(out)


def main(argv):
    from ultralytics import YOLO

    plate_w = argv[0] if len(argv) > 0 else str(config.best_weights("plate"))
    char_w = argv[1] if len(argv) > 1 else str(config.best_weights("chars"))
    names = load_names()
    display = pp.load_display(config.ROOT / "class_display_bn.json")
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    plate_model, char_model = YOLO(plate_w), YOLO(char_w)
    results = {}

    ninja_yaml = config.DATASETS_DIR / "char_detector" / "data_test_ninja.yaml"
    has_ninja = ninja_yaml.exists()

    print("\n1. Detection mAP on the held out test sets (reported separately)")
    results["plate_detector"] = map_scores(plate_model, "plate")
    results["char_detector_lpdb_test"] = map_scores(char_model, "chars")
    if has_ninja:
        results["char_detector_ninja_test"] = map_scores(char_model, "chars", ninja_yaml)
    print("  ", json.dumps(results, indent=2))

    print("\n2. Plate level exact match on test crops (reported separately)")
    print("  LPDB-A test split:")
    results["crop_exact_match_lpdb_test"] = crop_exact_match(char_model, names, "test")
    print("  ", results["crop_exact_match_lpdb_test"])
    if has_ninja:
        print("  DatasetNinja held out photos (a different distribution, an honest generalization check):")
        results["crop_exact_match_ninja_test"] = crop_exact_match(char_model, names, "test_ninja")
        print("  ", results["crop_exact_match_ninja_test"])
    else:
        print("  No DatasetNinja test set found, skipped.")

    folder = os.environ.get("BD_PIPELINE_DIR")
    if folder:
        print("\n3. Full pipeline on", folder)
        results["pipeline"] = run_pipeline(folder, plate_model, char_model, names, display)
        print("  ", results["pipeline"])
    else:
        print("\n3. Full pipeline skipped. Set BD_PIPELINE_DIR to a folder of vehicle photos (optional truth.csv).")

    print("\n4. Unlabeled crops")
    results["crops_check"] = check_unlabeled_crops(char_model, names, display)

    out = config.OUT_DIR / "eval_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nSaved", out)


if __name__ == "__main__":
    main(sys.argv[1:])

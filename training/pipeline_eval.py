"""Run the two int8 models the app ships on a folder of photos, like the app does.

Usage:
  python pipeline_eval.py PHOTO_DIR [OUT_CSV] [MODELS_DIR]

Needs only: pip install ai-edge-litert pillow numpy (no PyTorch or Ultralytics).

It mirrors app/src/ml/plateReader.ts: letterbox to 640, plate detector, best box
with 5 percent margin, letterbox the crop to 416, character detector, then
postprocess.py for the reading order. MODELS_DIR defaults to app/assets/models.
Set BD_MODEL_SUFFIX=dynamic (or fp32) to score plate_detector_dynamic.tflite and
char_detector_dynamic.tflite (from export.py with BD_EXPORT_DTYPE=dynamic) instead of the int8 files.

Writes OUT_CSV (default predictions.csv) with one row per photo:
  file, tokens, text, plate_score, min_char_score, truth, match
The tokens are class names in reading order, for example "Dhaka Metro Ga 2 7 8 4 7 6".

Faster labeling: run once, open the csv, fix the wrong rows, save the fixed
tokens as PHOTO_DIR/truth.csv with columns file,text. Run again to get the score.
Photos named in truth.csv are scored. The csv shows plate_score and min_char_score,
so the confidence thresholds can be tuned from the numbers.
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import postprocess as pp

HERE = Path(__file__).resolve().parent
PLATE_SIZE, CHAR_SIZE = 640, 416
SCORE_MIN, IOU_MAX = 0.25, 0.5
PLATE_PAD = 0.05
CHAR_MIN_CONF = 0.25
MAX_CHARS = 20


def load_model(path):
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=str(path), num_threads=4)
    it.allocate_tensors()
    return it


def run(it, chw):
    """Run one float32 [1,3,H,W] input and return the [C, A] output."""
    inp = it.get_input_details()[0]
    it.set_tensor(inp["index"], chw[None])
    it.invoke()
    return it.get_tensor(it.get_output_details()[0]["index"])[0]


def letterbox(img, size):
    """Resize keeping the aspect ratio, pad with gray 114. Returns CHW float32 and the mapping."""
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (pad_x, pad_y))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.ascontiguousarray(arr.transpose(2, 0, 1)), scale, pad_x, pad_y


def nms(boxes, scores, iou_max, limit):
    """Plain agnostic NMS on [x1,y1,x2,y2] boxes. Returns kept indices, best first."""
    order = np.argsort(-scores)
    keep = []
    while len(order) and len(keep) < limit:
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
        iou = inter / (area(boxes[i]) + area(boxes[rest]) - inter + 1e-9)
        order = rest[iou <= iou_max]
    return keep


def detect(out):
    """Decode a [4+nc, A] output (cx,cy,w,h normalized, then class scores) into kept detections."""
    cls_scores = out[4:]
    conf = cls_scores.max(axis=0)
    cls = cls_scores.argmax(axis=0)
    m = conf >= SCORE_MIN
    cx, cy, w, h, conf, cls = out[0][m], out[1][m], out[2][m], out[3][m], conf[m], cls[m]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return cx, cy, w, h, conf, cls, boxes


def read_photo(path, plate_model, char_model, names, display):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    chw, scale, pad_x, pad_y = letterbox(img, PLATE_SIZE)
    cx, cy, w, h, conf, _, boxes = detect(run(plate_model, chw))
    if len(conf) == 0:
        return [], "", 0.0, 0.0
    best = nms(boxes, conf, IOU_MAX, 1)[0]
    # Back to pixels of the photo, grow by the margin, clamp, crop.
    px = lambda n, pad: (n * PLATE_SIZE - pad) / scale  # noqa: E731
    x1, x2 = px(cx[best] - w[best] / 2, pad_x), px(cx[best] + w[best] / 2, pad_x)
    y1, y2 = px(cy[best] - h[best] / 2, pad_y), px(cy[best] + h[best] / 2, pad_y)
    dx, dy = (x2 - x1) * PLATE_PAD, (y2 - y1) * PLATE_PAD
    W, H = img.size
    box = (max(0, x1 - dx), max(0, y1 - dy), min(W, x2 + dx), min(H, y2 + dy))
    crop = img.crop(tuple(int(round(v)) for v in box))
    chw, _, _, _ = letterbox(crop, CHAR_SIZE)
    ccx, ccy, cw, ch, cconf, ccls, cboxes = detect(run(char_model, chw))
    keep = nms(cboxes, cconf, IOU_MAX, MAX_CHARS)
    dets = [
        pp.Detection(int(ccls[i]), float(ccx[i]), float(ccy[i]), float(cw[i]), float(ch[i]), float(cconf[i]))
        for i in keep
    ]
    tokens = pp.plate_tokens(dets, names, CHAR_MIN_CONF)
    text = pp.read_plate(dets, names, display, CHAR_MIN_CONF)
    kept = [d.conf for d in dets if d.conf >= CHAR_MIN_CONF]
    return tokens, text, float(conf[best]), (min(kept) if kept else 0.0)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    folder = Path(sys.argv[1])
    out_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("predictions.csv")
    models = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE.parent / "app" / "assets" / "models"
    names = pp.load_class_names(HERE / "classes.txt")
    display = pp.load_display(HERE / "class_display_bn.json")
    suffix = os.environ.get("BD_MODEL_SUFFIX", "int8")
    plate_model = load_model(models / ("plate_detector_%s.tflite" % suffix))
    char_model = load_model(models / ("char_detector_%s.tflite" % suffix))

    truth = {}
    if (folder / "truth.csv").exists():
        with open(folder / "truth.csv", encoding="utf-8", newline="") as f:
            truth = {r["file"]: r["text"].strip() for r in csv.DictReader(f)}

    photos = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    rows, scored, exact, edits, total = [], 0, 0, 0, 0
    for p in photos:
        tokens, text, plate_score, min_char = read_photo(p, plate_model, char_model, names, display)
        pred = " ".join(tokens)
        t = truth.get(p.name)
        match = ""
        if t is not None:
            truth_tokens = t.split()
            match = "yes" if tokens == truth_tokens else "no"
            scored += 1
            exact += match == "yes"
            edits += pp.edit_distance(tokens, truth_tokens)
            total += len(truth_tokens)
            if match == "no":
                print("mismatch %s\n  truth: %s\n  pred : %s" % (p.name, t, pred))
        rows.append([p.name, pred, text, "%.2f" % plate_score, "%.2f" % min_char, t or "", match])

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "tokens", "text", "plate_score", "min_char_score", "truth", "match"])
        w.writerows(rows)
    print("photos %d, no plate found %d" % (len(photos), sum(1 for r in rows if not r[1])))
    if scored:
        print("scored %d: exact match %.1f%%, character error rate %.1f%%" % (scored, 100 * exact / scored, 100 * edits / max(1, total)))
    print("wrote", out_csv)


if __name__ == "__main__":
    main()

"""Read the DatasetNinja (Supervisely format) character dataset and split it by source photo.

The dataset has train, test and validation folders, each with img/ and ann/.
Every image has one JSON file in ann/ named image name plus .json, with
rectangle objects: classTitle and points.exterior [[x1, y1], [x2, y2]].

Image names come from Roboflow exports: <source>.rf.<hash>.jpg. The same source
photo can appear several times as augmented copies. To avoid leakage, copies are
grouped by the text before '.rf.' and a whole group goes to one side of any split.
"""
import json
import random
from pathlib import Path

import config

MAP_PATH = config.ROOT / "ninja_class_map.json"
IMG_EXTS = {".jpg", ".jpeg", ".png"}
FOLDERS = ("train", "test", "validation")


def load_class_map(path=MAP_PATH):
    """Return {ninja class title: unified class name}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(data["classes"])


def source_key(file_name):
    """Group key: the part of the file name before '.rf.'.

    '105_229_1_jpg.rf.abc123.jpg' gives '105_229_1_jpg'. A name without '.rf.'
    falls back to its stem. A trailing '.json' is ignored.
    """
    name = file_name[:-5] if file_name.lower().endswith(".json") else file_name
    if ".rf." in name:
        return name.split(".rf.")[0]
    return Path(name).stem


def find_ninja_root(raw):
    """Folder that holds the train/test/validation folders (next to meta.json)."""
    for meta in sorted(Path(raw).rglob("meta.json")):
        base = meta.parent
        if all((base / f / "img").is_dir() and (base / f / "ann").is_dir() for f in FOLDERS):
            return base
    raise SystemExit("Could not find the DatasetNinja folders (train/test/validation with img and ann) under %s" % raw)


def parse_annotation(path):
    """Return (width, height, [(class_title, x1, y1, x2, y2), ...]) from one JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    size = data.get("size", {})
    boxes = []
    for obj in data.get("objects", []):
        if obj.get("geometryType") != "rectangle":
            boxes.append((obj.get("classTitle", ""), None, None, None, None))
            continue
        (x1, y1), (x2, y2) = obj["points"]["exterior"]
        boxes.append((obj["classTitle"], float(x1), float(y1), float(x2), float(y2)))
    return size.get("width", 0), size.get("height", 0), boxes


def to_yolo_box(x1, y1, x2, y2, width, height):
    """Pixel corners to normalized (cx, cy, w, h), clipped to the image.

    Returns (box, clipped) where box is None if nothing is left after clipping.
    """
    if width <= 0 or height <= 0:
        return None, False
    lo_x, hi_x = sorted((x1, x2))
    lo_y, hi_y = sorted((y1, y2))
    cx1, cx2 = min(max(lo_x, 0.0), width), min(max(hi_x, 0.0), width)
    cy1, cy2 = min(max(lo_y, 0.0), height), min(max(hi_y, 0.0), height)
    clipped = (cx1, cx2, cy1, cy2) != (lo_x, hi_x, lo_y, hi_y)
    if cx2 - cx1 < 1.0 or cy2 - cy1 < 1.0:  # less than one pixel wide or high
        return None, clipped
    box = (
        (cx1 + cx2) / 2.0 / width,
        (cy1 + cy2) / 2.0 / height,
        (cx2 - cx1) / width,
        (cy2 - cy1) / height,
    )
    return box, clipped


def convert_annotation(path, class_ids, image_size=None):
    """Convert one Supervisely JSON file to YOLO rows.

    class_ids maps a Ninja class title to the unified class id. image_size is
    (width, height) of the real image, used when it differs from the JSON size.

    Returns (rows, problems) where rows is a list of (cls, cx, cy, w, h) and
    problems is a dict of counters: unmapped, bad_geometry, clipped, size_mismatch.
    Callers should drop the whole image if unmapped or bad_geometry is not zero.
    """
    w, h, boxes = parse_annotation(path)
    problems = {"unmapped": 0, "bad_geometry": 0, "clipped": 0, "size_mismatch": 0}
    if image_size and tuple(image_size) != (w, h):
        problems["size_mismatch"] = 1
        w, h = image_size
    rows = []
    for title, x1, y1, x2, y2 in boxes:
        if title not in class_ids:
            problems["unmapped"] += 1
            continue
        if x1 is None:
            problems["bad_geometry"] += 1
            continue
        box, clipped = to_yolo_box(x1, y1, x2, y2, w, h)
        if box is None:
            problems["bad_geometry"] += 1
            continue
        problems["clipped"] += int(clipped)
        rows.append((class_ids[title],) + box)
    return rows, problems


def collect_images(root):
    """Return a list of dicts: image path, annotation path, original folder, group key."""
    items = []
    for folder in FOLDERS:
        img_dir, ann_dir = Path(root) / folder / "img", Path(root) / folder / "ann"
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            ann = ann_dir / (img.name + ".json")
            items.append(
                {
                    "image": img,
                    "ann": ann if ann.exists() else None,
                    "folder": folder,
                    "group": source_key(img.name),
                }
            )
    return items


def group_stats(items):
    """Counts used in the report: groups, duplicate groups, groups spanning folders."""
    groups = {}
    for it in items:
        groups.setdefault(it["group"], []).append(it)
    dup = {k: v for k, v in groups.items() if len(v) > 1}
    cross = {k: v for k, v in groups.items() if len({i["folder"] for i in v}) > 1}
    sizes = {}
    for v in groups.values():
        sizes[len(v)] = sizes.get(len(v), 0) + 1
    return {
        "images": len(items),
        "groups": len(groups),
        "groups_with_copies": len(dup),
        "groups_across_folders": len(cross),
        "group_size_histogram": dict(sorted(sizes.items())),
    }


def group_split(group_sizes, test_frac=0.3, val_frac=0.1, seed=config.SEED):
    """Assign whole groups to 'test', 'val' or 'train'.

    group_sizes maps group key to image count. Groups are shuffled with a fixed
    seed. Groups go to test until test holds about test_frac of all images. Of
    the remaining images, about val_frac go to val, and the rest to train. A
    group is never split, so the share is approximate (it overshoots by less
    than one group).
    """
    keys = sorted(group_sizes)
    random.Random(seed).shuffle(keys)
    total = sum(group_sizes.values())
    test_target = test_frac * total
    assign, in_test = {}, 0
    for k in keys:
        if in_test < test_target:
            assign[k] = "test"
            in_test += group_sizes[k]
    rest = [k for k in keys if k not in assign]
    val_target = val_frac * (total - in_test)
    in_val = 0
    for k in rest:
        if in_val < val_target:
            assign[k] = "val"
            in_val += group_sizes[k]
        else:
            assign[k] = "train"
    return assign

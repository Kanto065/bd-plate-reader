"""Build the two YOLO datasets used for training.

Usage:
  python merge_datasets.py

Run organize_data.py first. Output goes to WORK_DIR/datasets:
  plate_detector/   stage 1, one class 'license_plate'
                    (LPDB-A vehicle images plus NightLPD-BD night images)
  char_detector/    stage 2, all character classes
                    train and val: LPDB-A cropped plates plus about 70 percent of DatasetNinja
                    test: the LPDB-A test split (data.yaml)
                    test_ninja: about 30 percent of DatasetNinja held out, split by source
                    photo so augmented copies never sit on both sides (data_test_ninja.yaml)
Each folder has images/<part>, labels/<part> and data.yaml.
The class names are also written to training/classes.txt exactly as in the
dataset.

Label checks (see docs/dataset_notes.md for what the real data needed):
  Stage 1: boxes with a bad format, out of range coordinates or a class other
           than the plate are dropped. An image with no valid box is dropped.
  Stage 2: a plate with ANY invalid box is dropped completely, because a missing
           character would teach the model a wrong plate.
"""
import collections
import json
import os
import random
import re
import shutil
from pathlib import Path

import cv2
import yaml

import config
import ninja_convert as nc

IMG_EXTS = {".jpg", ".jpeg", ".png"}
RARE_LIMIT = 20  # classes with fewer train boxes than this are flagged
EPS = 1e-3


def find_dir(root, name):
    for p in sorted(Path(root).rglob(name)):
        if p.is_dir():
            return p
    raise SystemExit("Could not find a folder named '%s' under %s" % (name, root))


def find_night(root):
    """Locate the NightLPD-BD images folder and its YOLO detection labels folder.

    The real zip has labels in several formats (coco, pascal voc, yolo detection,
    yolo segmentation). Only the plain YOLO detection one is used. If it is not
    there, any folder named labels is used.
    """
    for images in sorted(Path(root).rglob("images")):
        labels = images.parent / "labels"
        if not (images.is_dir() and labels.is_dir()):
            continue
        det = [d for d in sorted(labels.iterdir()) if d.is_dir() and "detection" in d.name.lower()]
        return images, (det[0] if det else labels)
    raise SystemExit("Could not find the NightLPD-BD images and labels folders under %s" % root)


def pair_files(folder):
    """Return {stem: (image_path, label_path_or_None)} for a flat folder."""
    imgs, txts = {}, {}
    for p in sorted(Path(folder).iterdir()):
        ext = p.suffix.lower()
        if ext in IMG_EXTS:
            if p.stem in imgs:
                print("  duplicate stem with another extension, keeping first:", p.name)
                continue
            imgs[p.stem] = p
        elif ext == ".txt" and p.name.lower() != "classes.txt":
            txts[p.stem] = p
    return {s: (imgs[s], txts.get(s)) for s in imgs}


def read_label(path, n_classes):
    """Parse a YOLO label file.

    Returns (valid_boxes, n_invalid). A box is (cls, x, y, w, h) as floats.
    """
    boxes, bad = [], 0
    if path is None:
        return boxes, 0
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            if len(parts) != 5:
                raise ValueError
            c = int(float(parts[0]))
            x, y, w, h = (float(v) for v in parts[1:])
        except ValueError:
            bad += 1
            continue
        in_range = all(-EPS <= v <= 1 + EPS for v in (x, y, w, h))
        if not (0 <= c < n_classes) or not in_range or w <= 0 or h <= 0:
            bad += 1
            continue
        clamp = lambda v: min(1.0, max(0.0, v))
        boxes.append((c, clamp(x), clamp(y), clamp(w), clamp(h)))
    return boxes, bad


def image_ok(path):
    img = cv2.imread(str(path))
    return img is not None and img.shape[0] > 1 and img.shape[1] > 1


def safe_stem(prefix, stem):
    return prefix + "_" + re.sub(r"[^0-9A-Za-z_-]", "_", stem)


def split_items(items, rng):
    """items: list of dicts. Returns {'train': [...], 'val': [...], 'test': [...]}."""
    items = sorted(items, key=lambda i: i["name"])
    rng.shuffle(items)
    n = len(items)
    n_val = max(1, round(n * config.SPLIT[1]))
    n_test = max(1, round(n * config.SPLIT[2]))
    return {
        "test": items[:n_test],
        "val": items[n_test:n_test + n_val],
        "train": items[n_test + n_val:],
    }


def ensure_class_in_train(splits):
    """Move one image into train for any class that train never sees."""
    train_classes = {b[0] for i in splits["train"] for b in i["boxes"]}
    moved = 0
    for part in ("val", "test"):
        for item in list(splits[part]):
            missing = {b[0] for b in item["boxes"]} - train_classes
            if missing:
                splits[part].remove(item)
                splits["train"].append(item)
                train_classes |= missing
                moved += 1
    if moved:
        print("  moved %d image(s) to train so every class appears in train" % moved)


def write_split(out_root, splits, class_names):
    if out_root.exists():
        shutil.rmtree(out_root)
    for part, items in splits.items():
        (out_root / "images" / part).mkdir(parents=True)
        (out_root / "labels" / part).mkdir(parents=True)
        for it in items:
            dst = out_root / "images" / part / (it["name"] + it["src_img"].suffix.lower())
            shutil.copy2(it["src_img"], dst)
            lines = ["%d %.6f %.6f %.6f %.6f" % b for b in it["boxes"]]
            (out_root / "labels" / part / (it["name"] + ".txt")).write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
    data = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(class_names)},
    }
    with open(out_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def collect(pairs, prefix, n_classes, remap_to_zero, drop_whole_image_on_bad, stats):
    items = []
    for stem, (img, lab) in sorted(pairs.items()):
        if lab is None:
            stats["no_label"].append(img.name)
            continue
        boxes, bad = read_label(lab, n_classes)
        if remap_to_zero:
            boxes = [(0,) + b[1:] for b in boxes]
        if bad and drop_whole_image_on_bad:
            stats["dropped_bad"].append("%s (%d bad box)" % (img.name, bad))
            continue
        stats["bad_boxes"] += bad
        if not boxes:
            stats["empty"].append(img.name)
            continue
        if not image_ok(img):
            stats["corrupt"].append(img.name)
            continue
        items.append({"name": safe_stem(prefix, stem), "src_img": img, "boxes": boxes})
    return items


def new_stats():
    return {"no_label": [], "dropped_bad": [], "empty": [], "corrupt": [], "bad_boxes": 0}


def report(title, stats, kept):
    print("  %s: kept %d" % (title, kept))
    for key in ("no_label", "dropped_bad", "empty", "corrupt"):
        if stats[key]:
            print("    %s: %d %s" % (key, len(stats[key]), stats[key][:5]))
    if stats["bad_boxes"]:
        print("    invalid boxes dropped: %d" % stats["bad_boxes"])


def count_classes(items):
    c = collections.Counter()
    for it in items:
        for b in it["boxes"]:
            c[b[0]] += 1
    return c


NINJA_TEST_FRAC = 0.30  # share of Ninja images held out as an independent test set
NINJA_VAL_FRAC = 0.10  # share of the remaining 70 percent used as the training val set


def write_test_yaml(out_root, part, file_name, class_names):
    """A data yaml whose 'test' entry is another held out folder, for evaluate.py."""
    data = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/" + part,
        "names": {i: n for i, n in enumerate(class_names)},
    }
    with open(out_root / file_name, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def build_ninja(ninja_raw, class_names):
    """Convert DatasetNinja to YOLO items and split them by source photo.

    Returns ({'train': [...], 'val': [...], 'test': [...]}, summary_dict).
    Copies of one source photo (same text before '.rf.') always stay together,
    so no augmented copy of a test photo can be in train or val.
    """
    root = nc.find_ninja_root(ninja_raw)
    cmap = nc.load_class_map()
    name_to_id = {n: i for i, n in enumerate(class_names)}
    bad_targets = sorted({n for n in cmap.values() if n not in name_to_id})
    if bad_targets:
        raise SystemExit("ninja_class_map.json points to unknown classes: %s" % bad_targets)
    ids = {title: name_to_id[name] for title, name in cmap.items()}

    items = nc.collect_images(root)
    stats = nc.group_stats(items)
    print("  images: %d, source groups: %d, groups with several copies: %d, groups spanning folders: %d"
          % (stats["images"], stats["groups"], stats["groups_with_copies"], stats["groups_across_folders"]))
    print("  group size histogram (copies: groups):", stats["group_size_histogram"])

    usable, dropped = [], collections.Counter()
    seen_names = set()
    totals = collections.Counter()
    for it in items:
        if it["ann"] is None:
            dropped["no_annotation"] += 1
            continue
        img = cv2.imread(str(it["image"]))
        if img is None:
            dropped["unreadable_image"] += 1
            continue
        rows, prob = nc.convert_annotation(it["ann"], ids, (img.shape[1], img.shape[0]))
        for k, v in prob.items():
            totals[k] += v
        if prob["unmapped"] or prob["bad_geometry"] or not rows:
            dropped["bad_labels"] += 1
            continue
        name = safe_stem("ninja", it["image"].stem)
        while name in seen_names:
            name += "_x"
        seen_names.add(name)
        usable.append({"name": name, "src_img": it["image"], "boxes": rows, "group": it["group"]})
    if dropped:
        print("  dropped images:", dict(dropped))
    if totals["clipped"] or totals["size_mismatch"]:
        print("  boxes clipped to the image: %d, annotations whose size differed from the image: %d"
              % (totals["clipped"], totals["size_mismatch"]))

    sizes = collections.Counter(i["group"] for i in usable)
    assign = nc.group_split(sizes, NINJA_TEST_FRAC, NINJA_VAL_FRAC, config.SEED)
    out = {"train": [], "val": [], "test": []}
    for it in usable:
        out[assign[it["group"]]].append(it)

    if os.environ.get("BD_NINJA_TEST_ONE_PER_GROUP") == "1":
        first, kept = set(), []
        for it in sorted(out["test"], key=lambda i: i["name"]):
            if it["group"] not in first:
                first.add(it["group"])
                kept.append(it)
        print("  test set reduced to one image per source group: %d -> %d" % (len(out["test"]), len(kept)))
        out["test"] = kept

    groups_by_part = {p: {i["group"] for i in v} for p, v in out.items()}
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        overlap = groups_by_part[a] & groups_by_part[b]
        if overlap:
            raise SystemExit("Leak: %d source groups appear in both %s and %s" % (len(overlap), a, b))
    total_imgs = sum(len(v) for v in out.values())
    print("  split by source group (no group on both sides):")
    for p in ("train", "val", "test"):
        print("    %-5s %5d images from %4d groups (%.1f%% of images)"
              % (p, len(out[p]), len(groups_by_part[p]), 100.0 * len(out[p]) / max(1, total_imgs)))
    test_copies = sum(1 for g in groups_by_part["test"] if sizes[g] > 1)
    print("    test groups that have several copies: %d of %d" % (test_copies, len(groups_by_part["test"])))

    (config.DATASETS_DIR).mkdir(parents=True, exist_ok=True)
    (config.DATASETS_DIR / "ninja_split.json").write_text(
        json.dumps({p: sorted(g) for p, g in groups_by_part.items()}, indent=1), encoding="utf-8")
    summary = {
        "images": stats["images"], "groups": stats["groups"],
        "groups_with_copies": stats["groups_with_copies"],
        "groups_across_folders": stats["groups_across_folders"],
        "group_size_histogram": stats["group_size_histogram"],
        "dropped": dict(dropped),
        "split_images": {p: len(out[p]) for p in out},
        "split_groups": {p: len(groups_by_part[p]) for p in out},
    }
    return out, summary


def main():
    raw = config.RAW_DIR
    if not (raw / "lpdb").is_dir():
        raise SystemExit("Run organize_data.py first (no extracted data in %s)" % raw)
    rng = random.Random(config.SEED)

    vehicles = find_dir(raw / "lpdb", "Bangladeshi_Vehicle_Image_with_visible_license_plate")
    crops = find_dir(raw / "lpdb", "Bangla_License_Plate")
    night_images, night_labels = find_night(raw / "night")

    class_names = [l.rstrip("\r\n") for l in (crops / "classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    print("Character classes in classes.txt:", len(class_names))
    (config.ROOT / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")

    # Stage 1: plate detector.
    print("\nStage 1: plate detector")
    plate_items = []
    for label, folder, labels_folder, prefix in (
        ("LPDB-A vehicles", vehicles, None, "veh"),
        ("NightLPD-BD", night_images, night_labels, "night"),
    ):
        if labels_folder is None:
            pairs = pair_files(folder)
        else:
            # NightLPD-BD keeps its own train/val/test subfolders. They are
            # merged here and split again with the rest of the data.
            pairs = {}
            for sub in sorted([folder] + [d for d in folder.rglob("*") if d.is_dir()]):
                for s, (img, _) in pair_files(sub).items():
                    pairs.setdefault(s, (img, None))
            lab_map = {p.stem: p for p in labels_folder.rglob("*.txt")}
            pairs = {s: (img, lab_map.get(s)) for s, (img, _) in pairs.items()}
        stats = new_stats()
        items = collect(pairs, prefix, 1, False, False, stats)
        report(label, stats, len(items))
        if not items:
            raise SystemExit("No usable images found for %s in %s" % (label, folder))
        for it in items:
            it["source"] = prefix
        plate_items.append(items)
    plate_splits = {"train": [], "val": [], "test": []}
    for items in plate_items:  # split each source separately so night images reach every split
        sp = split_items(items, rng)
        for part in plate_splits:
            plate_splits[part].extend(sp[part])
    write_split(config.DATASETS_DIR / "plate_detector", plate_splits, ["license_plate"])
    print("  split sizes:", {k: len(v) for k, v in plate_splits.items()})

    # Stage 2: character detector.
    print("\nStage 2: character detector")
    stats = new_stats()
    char_items = collect(pair_files(crops), "chr", len(class_names), False, True, stats)
    report("LPDB-A crops", stats, len(char_items))
    char_splits = split_items(char_items, rng)
    ensure_class_in_train(char_splits)
    lpdb_train_counts = count_classes(char_splits["train"])

    # DatasetNinja adds training and val data, plus a separate held out test set
    # (images/test_ninja) so LPDB-A and Ninja results are reported separately.
    summary = {}
    ninja_raw = raw / "ninja"
    if ninja_raw.is_dir():
        print("\nStage 2 extra data: DatasetNinja")
        ninja_splits, summary["ninja"] = build_ninja(ninja_raw, class_names)
        char_splits["train"].extend(ninja_splits["train"])
        char_splits["val"].extend(ninja_splits["val"])
        char_splits["test_ninja"] = ninja_splits["test"]
    else:
        print("\nNo DatasetNinja data found, stage 2 uses LPDB-A only")

    out_root = config.DATASETS_DIR / "char_detector"
    write_split(out_root, char_splits, class_names)
    if "test_ninja" in char_splits:
        write_test_yaml(out_root, "test_ninja", "data_test_ninja.yaml", class_names)
    print("  split sizes:", {k: len(v) for k, v in char_splits.items()})
    summary["char_split_sizes"] = {k: len(v) for k, v in char_splits.items()}

    train_counts = count_classes(char_splits["train"])
    all_counts = count_classes(char_items)
    print("\nCharacter class counts. Classes below %d train boxes are flagged "
          "(train counts include DatasetNinja, LPDB-A alone in brackets):" % RARE_LIMIT)
    rare = []
    for i, name in enumerate(class_names):
        if train_counts[i] < RARE_LIMIT:
            rare.append((name, train_counts[i], lpdb_train_counts[i], all_counts[i]))
    for name, tr, lp, al in sorted(rare, key=lambda r: r[1]):
        print("  RARE %-14s train=%d [LPDB-A %d] total LPDB-A=%d" % (name, tr, lp, al))
    still_rare_before = [n for i, n in enumerate(class_names) if lpdb_train_counts[i] < RARE_LIMIT]
    still_rare_after = [n for i, n in enumerate(class_names) if train_counts[i] < RARE_LIMIT]
    print("  rare classes (under %d train boxes): %d with LPDB-A only, %d with DatasetNinja added"
          % (RARE_LIMIT, len(still_rare_before), len(still_rare_after)))
    helped = [(n, lpdb_train_counts[i], train_counts[i]) for i, n in enumerate(class_names)
              if lpdb_train_counts[i] < RARE_LIMIT and train_counts[i] > lpdb_train_counts[i]]
    print("  rare classes that gained DatasetNinja data:", helped)
    summary["rare_before"] = len(still_rare_before)
    summary["rare_after"] = len(still_rare_after)
    summary["rare_helped"] = helped
    (config.DATASETS_DIR / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("  total LPDB-A boxes: %d, classes with zero LPDB-A boxes: %s" % (
        sum(all_counts.values()), [class_names[i] for i in range(len(class_names)) if all_counts[i] == 0]))
    print("\nDatasets written to", config.DATASETS_DIR)


if __name__ == "__main__":
    main()

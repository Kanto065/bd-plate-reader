import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ninja_convert as nc  # noqa: E402
import postprocess as pp  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = pp.load_class_names(os.path.join(HERE, "classes.txt"))


def write_ann(path, width, height, objects):
    objs = []
    for title, x1, y1, x2, y2 in objects:
        objs.append(
            {
                "classTitle": title,
                "geometryType": "rectangle",
                "points": {"exterior": [[x1, y1], [x2, y2]], "interior": []},
            }
        )
    path.write_text(json.dumps({"size": {"width": width, "height": height}, "objects": objs}), encoding="utf-8")


# source_key

def test_source_key_before_rf():
    assert nc.source_key("105_229_1_jpg.rf.8fc05713ac37be7f06f08ce51bc69a1a.jpg") == "105_229_1_jpg"


def test_source_key_ignores_json_suffix():
    assert nc.source_key("a_b_jpg.rf.abc.jpg.json") == "a_b_jpg"


def test_source_key_without_rf_uses_stem():
    assert nc.source_key("plain_photo.jpg") == "plain_photo"


def test_augmented_copies_share_a_key():
    names = ["x_1_jpg.rf.%s.jpg" % h for h in ("aaa", "bbb", "ccc", "ddd")]
    assert len({nc.source_key(n) for n in names}) == 1


# geometry

def test_to_yolo_box_basic():
    box, clipped = nc.to_yolo_box(10, 20, 30, 60, 100, 200)
    assert not clipped
    assert [round(v, 6) for v in box] == [0.2, 0.2, 0.2, 0.2]


def test_to_yolo_box_swapped_corners():
    a, _ = nc.to_yolo_box(10, 20, 30, 60, 100, 200)
    b, _ = nc.to_yolo_box(30, 60, 10, 20, 100, 200)
    assert a == b


def test_to_yolo_box_clips_to_image():
    box, clipped = nc.to_yolo_box(-10, 0, 50, 100, 100, 100)
    assert clipped
    assert box[0] == 0.25 and box[2] == 0.5  # x from 0 to 50
    assert all(0 <= v <= 1 for v in box)


def test_to_yolo_box_degenerate_is_dropped():
    box, _ = nc.to_yolo_box(10, 10, 10.5, 40, 100, 100)
    assert box is None
    box, _ = nc.to_yolo_box(200, 200, 300, 300, 100, 100)  # fully outside
    assert box is None


# convert_annotation

IDS = {"0": 0, "metro": 10, "dhaka": 48}


def test_convert_annotation_ok(tmp_path):
    p = tmp_path / "a.jpg.json"
    write_ann(p, 200, 100, [("dhaka", 20, 10, 100, 50), ("0", 120, 10, 160, 50)])
    rows, problems = nc.convert_annotation(p, IDS)
    assert [r[0] for r in rows] == [48, 0]
    assert problems == {"unmapped": 0, "bad_geometry": 0, "clipped": 0, "size_mismatch": 0}
    assert [round(v, 4) for v in rows[0][1:]] == [0.3, 0.3, 0.4, 0.4]


def test_convert_annotation_unmapped_class(tmp_path):
    p = tmp_path / "a.jpg.json"
    write_ann(p, 200, 100, [("mystery", 20, 10, 100, 50)])
    rows, problems = nc.convert_annotation(p, IDS)
    assert rows == [] and problems["unmapped"] == 1


def test_convert_annotation_non_rectangle(tmp_path):
    p = tmp_path / "a.jpg.json"
    p.write_text(
        json.dumps({"size": {"width": 10, "height": 10}, "objects": [{"classTitle": "0", "geometryType": "polygon", "points": {"exterior": []}}]}),
        encoding="utf-8",
    )
    rows, problems = nc.convert_annotation(p, IDS)
    assert rows == [] and problems["bad_geometry"] == 1


def test_convert_annotation_uses_real_image_size(tmp_path):
    p = tmp_path / "a.jpg.json"
    write_ann(p, 200, 100, [("0", 0, 0, 100, 50)])
    rows, problems = nc.convert_annotation(p, IDS, image_size=(400, 200))
    assert problems["size_mismatch"] == 1
    assert [round(v, 4) for v in rows[0][1:]] == [0.125, 0.125, 0.25, 0.25]


# class map

def test_class_map_covers_27_classes_and_targets_exist():
    cmap = nc.load_class_map()
    assert len(cmap) == 27
    assert sorted(cmap) == sorted(list("0123456789") + [
        "ba", "ca", "chattogram", "da", "dhaka", "ga", "gha", "ha", "ja", "jashore",
        "ka", "kha", "khulna", "metro", "na", "ta", "va"])
    missing = [t for t in cmap.values() if t not in NAMES]
    assert missing == []


def test_class_map_targets_are_unique():
    cmap = nc.load_class_map()
    assert len(set(cmap.values())) == len(cmap)


def test_class_map_irregular_names():
    cmap = nc.load_class_map()
    assert cmap["chattogram"] == "Chatto"  # the short word, not Chattogram
    assert cmap["va"] == "Bha" and cmap["ca"] == "Cha" and cmap["jashore"] == "Jessore"


# group split

def make_sizes(quads=870, singles=581):
    sizes = {"q%04d_jpg" % i: 4 for i in range(quads)}
    sizes.update({"s%04d_jpg" % i: 1 for i in range(singles)})
    return sizes


def test_group_split_covers_every_group_once():
    sizes = make_sizes()
    assign = nc.group_split(sizes)
    assert set(assign) == set(sizes)
    assert set(assign.values()) == {"train", "val", "test"}


def test_group_split_image_shares():
    sizes = make_sizes()
    assign = nc.group_split(sizes, 0.3, 0.1)
    total = sum(sizes.values())
    share = {p: sum(n for k, n in sizes.items() if assign[k] == p) / total for p in ("train", "val", "test")}
    assert 0.29 <= share["test"] <= 0.32
    assert 0.06 <= share["val"] <= 0.09  # about 10 percent of the remaining 70 percent
    assert 0.60 <= share["train"] <= 0.66


def test_group_split_is_deterministic_and_seed_dependent():
    sizes = make_sizes()
    assert nc.group_split(sizes, seed=42) == nc.group_split(sizes, seed=42)
    assert nc.group_split(sizes, seed=42) != nc.group_split(sizes, seed=7)


def test_augmented_copies_never_straddle_a_split():
    # images carry their group key; every image of a group must get the same part
    images = []
    for g in range(200):
        for k in range(4):
            images.append("g%03d_jpg.rf.copy%d.jpg" % (g, k))
    sizes = {}
    for name in images:
        sizes[nc.source_key(name)] = sizes.get(nc.source_key(name), 0) + 1
    assign = nc.group_split(sizes)
    parts_per_group = {}
    for name in images:
        parts_per_group.setdefault(nc.source_key(name), set()).add(assign[nc.source_key(name)])
    assert all(len(p) == 1 for p in parts_per_group.values())
    test_groups = {g for g, p in parts_per_group.items() if p == {"test"}}
    train_groups = {g for g, p in parts_per_group.items() if p == {"train"}}
    assert test_groups and train_groups and not (test_groups & train_groups)


def test_group_stats(tmp_path):
    items = [
        {"image": "a", "ann": None, "folder": "train", "group": "g1"},
        {"image": "b", "ann": None, "folder": "train", "group": "g1"},
        {"image": "c", "ann": None, "folder": "test", "group": "g2"},
        {"image": "d", "ann": None, "folder": "train", "group": "g3"},
        {"image": "e", "ann": None, "folder": "validation", "group": "g3"},
    ]
    s = nc.group_stats(items)
    assert s["images"] == 5 and s["groups"] == 3
    assert s["groups_with_copies"] == 2 and s["groups_across_folders"] == 1
    assert s["group_size_histogram"] == {1: 1, 2: 2}

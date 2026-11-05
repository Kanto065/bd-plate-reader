import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import postprocess as pp  # noqa: E402
from postprocess import Detection  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = pp.load_class_names(os.path.join(HERE, "classes.txt"))
DISPLAY = pp.load_display(os.path.join(HERE, "class_display_bn.json"))
ID = {n: i for i, n in enumerate(NAMES)}


def det(name, x, y, w=0.06, h=0.2, conf=0.9):
    return Detection(ID[name], x, y, w, h, conf)


def two_line_plate():
    """Dhaka Metro Ga on top, 27-8476 below. Detections are deliberately shuffled."""
    top = [det("Dhaka", 0.25, 0.28, 0.2), det("Metro", 0.55, 0.28, 0.2), det("Ga", 0.8, 0.28)]
    digits = "278476"
    bottom = [det(d, 0.12 + 0.14 * i, 0.72) for i, d in enumerate(digits)]
    items = top + bottom
    return [items[i] for i in (5, 0, 8, 2, 7, 1, 4, 6, 3)]


def single_line_plate():
    """Dhaka Metro Gha 112345 on one line."""
    seq = ["Dhaka", "Metro", "Gha", "1", "1", "2", "3", "4", "5"]
    return [det(n, 0.05 + 0.1 * i, 0.5 + (0.01 if i % 2 else -0.01)) for i, n in enumerate(seq)]


def test_class_names_loaded():
    assert len(NAMES) == 102
    assert NAMES[0] == "0" and NAMES[10] == "Metro"


def test_group_two_lines_top_to_bottom():
    lines = pp.group_into_lines(two_line_plate())
    assert len(lines) == 2
    assert [NAMES[d.cls] for d in lines[0]] == ["Dhaka", "Metro", "Ga"]
    assert [NAMES[d.cls] for d in lines[1]] == list("278476")


def test_single_line_is_one_group_sorted_by_x():
    shuffled = list(reversed(single_line_plate()))
    lines = pp.group_into_lines(shuffled)
    assert len(lines) == 1
    xs = [d.x for d in lines[0]]
    assert xs == sorted(xs)


def test_empty_input():
    assert pp.group_into_lines([]) == []
    assert pp.read_plate([], NAMES) == ""


def test_read_plate_english_two_line():
    assert pp.read_plate(two_line_plate(), NAMES) == "Dhaka Metro-Ga 27-8476"


def test_read_plate_bangla_two_line():
    assert pp.read_plate(two_line_plate(), NAMES, DISPLAY) == "ঢাকা মেট্রো-গ ২৭-৮৪৭৬"


def test_read_plate_bangla_single_line():
    assert pp.read_plate(single_line_plate(), NAMES, DISPLAY) == "ঢাকা মেট্রো-ঘ ১১-২৩৪৫"


def test_low_confidence_boxes_are_ignored():
    dets = two_line_plate() + [det("5", 0.9, 0.72, conf=0.1)]
    assert pp.read_plate(dets, NAMES, min_conf=0.25) == "Dhaka Metro-Ga 27-8476"
    assert pp.read_plate(dets, NAMES, min_conf=0.0).endswith("5")


def test_partial_read_still_gives_text():
    only_digits = [det(d, 0.1 + 0.1 * i, 0.5) for i, d in enumerate("1234")]
    assert pp.read_plate(only_digits, NAMES) == "1234"


def test_unknown_class_id_does_not_crash():
    dets = [Detection(999, 0.5, 0.5, 0.1, 0.2, 0.9)]
    assert pp.plate_tokens(dets, NAMES) == ["?"]


def test_plate_tokens_reading_order():
    assert pp.plate_tokens(two_line_plate(), NAMES) == ["Dhaka", "Metro", "Ga"] + list("278476")


def test_edit_distance():
    assert pp.edit_distance(["a", "b"], ["a", "b"]) == 0
    assert pp.edit_distance(["a", "b", "c"], ["a", "c"]) == 1
    assert pp.edit_distance([], ["x", "y"]) == 2


def test_display_map_covers_every_class_name():
    missing = [n for n in NAMES if n not in DISPLAY]
    assert missing == []


def test_display_json_is_valid_utf8():
    with open(os.path.join(HERE, "class_display_bn.json"), encoding="utf-8") as f:
        assert isinstance(json.load(f), dict)

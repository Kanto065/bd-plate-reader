"""Turn detected plate characters into plate text.

This module is pure Python with no dependencies, so Milestone 3 can port it to
TypeScript one function at a time. Boxes use normalized center format, the same
as YOLO labels: (cls, x, y, w, h, conf).

Steps:
  1. group_into_lines: cluster boxes into text lines by their y center.
  2. each line is sorted left to right by x.
  3. class ids are mapped to names, then to text.

A BRTA plate reads, for example (English class names):
  Dhaka Metro Ga          line 1: district, optional Metro, series letter
  2 7 8 4 7 6             line 2: digits
which becomes "Dhaka Metro-Ga 27-8476". With the Bangla display map it becomes
"ঢাকা মেট্রো-গ ২৭-৮৪৭৬".
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

# Series letters in the dataset class list. Digits are 0 to 9, 'Metro' is the
# metro marker, and everything else is a district or metro city word.
LETTERS = frozenset(
    [
        "A", "Bha", "Cha", "Chha", "DA", "Da", "E", "Ga", "Gha", "Ha", "Ja",
        "Jha", "Ka", "Kha", "La", "Ma", "Na", "Pa", "Sa", "Sha", "THA", "Ta",
        "Tha", "U", "Dha", "Ba",
    ]
)
METRO = "Metro"


@dataclass(frozen=True)
class Detection:
    cls: int
    x: float  # center x, normalized 0..1
    y: float  # center y, normalized 0..1
    w: float
    h: float
    conf: float = 1.0


def load_class_names(path) -> List[str]:
    """Read classes.txt, one name per line, in class id order."""
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\r\n") for line in f if line.strip()]


def load_display(path) -> Dict[str, str]:
    """Read class_display_bn.json: class name to Bangla text."""
    import json

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def median(values: Sequence[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def group_into_lines(dets: Sequence[Detection], gap_ratio: float = 0.5) -> List[List[Detection]]:
    """Group detections into lines by y, each line sorted left to right.

    A new line starts when a box center is further than gap_ratio times the
    median box height away from the mean y of the current line. Lines are
    returned top to bottom.
    """
    if not dets:
        return []
    threshold = gap_ratio * median([d.h for d in dets])
    ordered = sorted(dets, key=lambda d: d.y)
    lines: List[List[Detection]] = [[ordered[0]]]
    for d in ordered[1:]:
        current = lines[-1]
        mean_y = sum(c.y for c in current) / len(current)
        if abs(d.y - mean_y) > threshold:
            lines.append([d])
        else:
            current.append(d)
    return [sorted(line, key=lambda d: d.x) for line in lines]


def lines_to_names(lines: Sequence[Sequence[Detection]], names: Sequence[str]) -> List[List[str]]:
    """Map each detection to its class name. Unknown ids become '?'."""
    out = []
    for line in lines:
        out.append([names[d.cls] if 0 <= d.cls < len(names) else "?" for d in line])
    return out


def flatten(lines_names: Sequence[Sequence[str]]) -> List[str]:
    return [n for line in lines_names for n in line]


def format_plate(lines_names: Sequence[Sequence[str]], display: Optional[Dict[str, str]] = None) -> str:
    """Build the plate string from line by line class names.

    Layout: "<district> <Metro>-<letter> <dd>-<dddd>". Pieces that are not
    detected are skipped, so a partial read still gives readable text.
    display maps a class name to text (for example Bangla). Without it the
    class names themselves are used.
    """
    disp = display or {}

    def show(name: str) -> str:
        return disp.get(name, name)

    words, letters, digits = [], [], []
    for name in flatten(lines_names):
        if name.isdigit():
            digits.append(name)
        elif name in LETTERS:
            letters.append(name)
        else:
            words.append(name)  # district, Metro, or anything unknown

    head_parts = [show(w) for w in words]
    head = " ".join(head_parts)
    if letters:
        letter_text = "".join(show(n) for n in letters)
        head = (head + "-" + letter_text) if head else letter_text

    shown = [show(d) for d in digits]
    if len(shown) > 4:
        # Hyphen before the last four digits: 278476 becomes 27-8476.
        number = "".join(shown[:-4]) + "-" + "".join(shown[-4:])
    else:
        number = "".join(shown)
    return " ".join(p for p in (head, number) if p)


def read_plate(dets: Sequence[Detection], names: Sequence[str], display: Optional[Dict[str, str]] = None, min_conf: float = 0.0) -> str:
    kept = [d for d in dets if d.conf >= min_conf]
    lines = group_into_lines(kept)
    return format_plate(lines_to_names(lines, names), display)


def plate_tokens(dets: Sequence[Detection], names: Sequence[str], min_conf: float = 0.0) -> List[str]:
    """Reading order token list, used for exact match scoring."""
    kept = [d for d in dets if d.conf >= min_conf]
    return flatten(lines_to_names(group_into_lines(kept), names))


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

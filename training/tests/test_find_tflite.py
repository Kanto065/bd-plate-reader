import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import export  # noqa: E402


def test_finds_file_next_to_weights(tmp_path):
    w = tmp_path / "best.pt"
    w.write_bytes(b"x")
    f = tmp_path / "best_int8.tflite"
    f.write_bytes(b"x")
    assert export.find_tflite(w) == f


def test_prefers_returned_path(tmp_path):
    w = tmp_path / "best.pt"
    f = tmp_path / "other_int8.tflite"
    f.write_bytes(b"x")
    assert export.find_tflite(w, str(f)) == f


def test_old_saved_model_folder(tmp_path):
    w = tmp_path / "best.pt"
    d = tmp_path / "best_saved_model"
    d.mkdir()
    f = d / "best_full_integer_quant_int8.tflite"
    f.write_bytes(b"x")
    assert export.find_tflite(w) == f

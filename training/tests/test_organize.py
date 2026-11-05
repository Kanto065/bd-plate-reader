import io
import os
import sys
import tarfile
import zipfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import organize_data as od  # noqa: E402

REAL_NAMES = [
    "Bangla LPDB - A.zip",
    "NightLPD-BD Annotated Bangladeshi Nighttime Vehicl.zip",
    "bangladeshi-license-plate-recognition_-character-recognition-DatasetNinja.tar",
    "Bangla Licence plate detection and recognition dataset.zip",
    "load test v2.zip",
    "notes.pdf",
]


def test_archives_are_found_by_pattern(tmp_path):
    for n in REAL_NAMES:
        (tmp_path / n).write_bytes(b"x")
    found = od.find_archives(tmp_path)
    assert found["lpdb"].name == "Bangla LPDB - A.zip"
    assert found["night"].name.startswith("NightLPD-BD")
    assert found["ninja"].name.endswith("DatasetNinja.tar")
    assert found["crops"].name.startswith("Bangla Licence plate")
    assert set(found) == {"lpdb", "night", "ninja", "crops"}


def test_renamed_files_still_match(tmp_path):
    for n in ("bangla_lpdb_a.zip", "nightlpd_bd.zip", "ninja_chars.tar"):
        (tmp_path / n).write_bytes(b"x")
    assert set(od.find_archives(tmp_path)) == {"lpdb", "night", "ninja"}


def test_zip_with_path_traversal_is_rejected(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with pytest.raises(RuntimeError):
        od.safe_extract(bad, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_tar_with_path_traversal_is_rejected(tmp_path):
    bad = tmp_path / "bad.tar"
    with tarfile.open(bad, "w") as tf:
        data = b"x"
        info = tarfile.TarInfo("../evil.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    with pytest.raises(RuntimeError):
        od.safe_extract(bad, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_tar_with_symlink_is_rejected(tmp_path):
    bad = tmp_path / "link.tar"
    with tarfile.open(bad, "w") as tf:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    with pytest.raises(RuntimeError):
        od.safe_extract(bad, tmp_path / "out")


def test_normal_tar_extracts(tmp_path):
    good = tmp_path / "good.tar"
    with tarfile.open(good, "w") as tf:
        data = b"hello"
        info = tarfile.TarInfo("train/img/a.jpg")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    od.safe_extract(good, tmp_path / "out")
    assert (tmp_path / "out" / "train" / "img" / "a.jpg").read_bytes() == b"hello"

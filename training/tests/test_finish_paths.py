"""Guards for the export and evaluate only workflow (colab_finish.ipynb).

The finish notebook runs export.py and evaluate.py on best.pt files that are
already on Drive, without training. These tests make sure neither script starts
to depend on last.pt or on a training finished marker, and that the notebook is
valid JSON that never calls train.py.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402


def test_best_weights_live_under_out_dir():
    for stage in ("plate", "chars"):
        w = config.best_weights(stage)
        assert w.name == "best.pt"
        assert w.parent.name == "weights"
        assert w.parent.parent.name == config.STAGES[stage]["name"]
        assert config.OUT_DIR in w.parents


def test_export_and_evaluate_do_not_need_last_pt_or_training():
    for script in ("export.py", "evaluate.py"):
        src = (ROOT / script).read_text(encoding="utf-8")
        assert "last.pt" not in src, script
        assert "run_finished" not in src, script
        assert "import train" not in src and "from train" not in src, script


def test_finish_notebook_is_valid_and_does_not_train():
    nb = json.loads((ROOT / "colab_finish.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert not re.search(r"train\.py", code)
    assert "export.py" in code and "evaluate.py" in code
    assert "organize_data.py" in code and "merge_datasets.py" in code
    for c in nb["cells"]:
        assert c["cell_type"] in ("code", "markdown")
        assert isinstance(c["source"], list)

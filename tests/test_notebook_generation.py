"""End-to-end tests for scripts/gen_train_notebook.py."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
GEN_SCRIPT = REPO_ROOT / "scripts" / "gen_train_notebook.py"
HELPERS_PATH = REPO_ROOT / "scripts" / "notebook_helpers.py"


def _generate() -> dict:
    with TemporaryDirectory() as td:
        out_path = Path(td) / "out.ipynb"
        result = subprocess.run(
            [sys.executable, str(GEN_SCRIPT), "--out", str(out_path)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"generator failed:\nstdout:{result.stdout}\nstderr:{result.stderr}"
            )
        return json.loads(out_path.read_text())


def _sources(doc: dict) -> list[str]:
    return ["".join(c["source"]) for c in doc["cells"]]


class GenerationSmokeTests(unittest.TestCase):
    def test_generator_runs_and_produces_valid_notebook(self) -> None:
        doc = _generate()
        self.assertEqual(doc["nbformat"], 4)
        self.assertIn("cells", doc)
        self.assertGreater(len(doc["cells"]), 0)

    def test_every_code_cell_parses_as_python(self) -> None:
        doc = _generate()
        for i, cell in enumerate(doc["cells"]):
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            if src.lstrip().startswith(("%%capture", "%pip", "!pip")):
                continue
            try:
                ast.parse(src)
            except SyntaxError as e:
                self.fail(f"cell {i} failed to parse: {e}\n--- source ---\n{src}")


class ConfigCellTests(unittest.TestCase):
    def test_targets_dict_present_with_three_keys(self) -> None:
        joined = "\n".join(_sources(_generate()))
        self.assertIn("TARGETS = {", joined)
        self.assertIn('"noodle"', joined)
        self.assertIn('"rice"', joined)
        self.assertIn('"soup"', joined)
        # variant + imgsz + num_classes baked in
        self.assertIn("lite2", joined)
        self.assertIn("lite1", joined)
        self.assertIn("lite0", joined)
        self.assertIn("imgsz=260", joined)
        self.assertIn("imgsz=240", joined)
        self.assertIn("imgsz=224", joined)

    def test_target_selector_present(self) -> None:
        joined = "\n".join(_sources(_generate()))
        self.assertIn('TARGET = "noodle"', joined)

    def test_drive_paths_point_to_l2(self) -> None:
        joined = "\n".join(_sources(_generate()))
        self.assertIn("/content/drive/MyDrive/FitUpNutritionL2", joined)


class HelpersEmbedTests(unittest.TestCase):
    def test_helpers_cell_is_byte_equal_to_source(self) -> None:
        doc = _generate()
        helpers_src = HELPERS_PATH.read_text().rstrip()
        embedded = [s for s in _sources(doc) if "def extract_target_zip" in s]
        self.assertEqual(len(embedded), 1, "expected exactly one helpers cell")
        self.assertEqual(embedded[0].rstrip(), helpers_src)


if __name__ == "__main__":
    unittest.main()

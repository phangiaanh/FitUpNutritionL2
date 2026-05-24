"""Unit tests for scripts/notebook_helpers.py.

Helpers are TF-free by design so these tests run fast with only
numpy + pillow installed (TF is required only by the notebook itself).
"""

from __future__ import annotations

import io
import sys
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.notebook_helpers import extract_target_zip  # noqa: E402


def _make_zip(zip_path: Path, files: dict[str, bytes]) -> None:
    """Create a zip archive containing the given arcname→content entries."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for arcname, data in files.items():
            zf.writestr(arcname, data)


def _populate(data_dir: Path, target: str) -> None:
    """Pre-populate the unzipped tree so extract_target_zip can skip."""
    for split in ("train", "val", "test"):
        cls_dir = data_dir / f"l2_{target}" / split / "fake_class"
        cls_dir.mkdir(parents=True, exist_ok=True)
        (cls_dir / "x.jpg").write_bytes(b"")


class ExtractTargetZipTests(unittest.TestCase):
    def test_extracts_zip_into_per_target_tree(self) -> None:
        with TemporaryDirectory() as td:
            hf_cache = Path(td) / "hf"
            data_root = Path(td) / "data"
            _make_zip(hf_cache / "l2_rice" / "data.zip", {
                "train/com_tam/a.jpg": b"img",
                "val/com_tam/b.jpg":   b"img",
                "test/com_tam/c.jpg":  b"img",
            })
            extract_target_zip(hf_cache, data_root, target="rice")
            self.assertTrue((data_root / "l2_rice" / "train" / "com_tam" / "a.jpg").is_file())
            self.assertTrue((data_root / "l2_rice" / "val"   / "com_tam" / "b.jpg").is_file())
            self.assertTrue((data_root / "l2_rice" / "test"  / "com_tam" / "c.jpg").is_file())

    def test_skips_when_already_extracted(self) -> None:
        with TemporaryDirectory() as td:
            hf_cache = Path(td) / "hf"
            data_root = Path(td) / "data"
            _populate(data_root, "rice")
            # Note: no zip in hf_cache; function must not look for it on skip path.
            hf_cache.mkdir()
            extract_target_zip(hf_cache, data_root, target="rice")  # must not raise
            self.assertTrue((data_root / "l2_rice" / "train" / "fake_class" / "x.jpg").is_file())

    def test_force_re_extracts(self) -> None:
        with TemporaryDirectory() as td:
            hf_cache = Path(td) / "hf"
            data_root = Path(td) / "data"
            _populate(data_root, "rice")
            _make_zip(hf_cache / "l2_rice" / "data.zip", {
                "train/com_tam/fresh.jpg": b"new",
                "val/com_tam/fresh.jpg":   b"new",
                "test/com_tam/fresh.jpg":  b"new",
            })
            extract_target_zip(hf_cache, data_root, target="rice", force=True)
            self.assertTrue((data_root / "l2_rice" / "train" / "com_tam" / "fresh.jpg").is_file())

    def test_raises_if_zip_missing(self) -> None:
        with TemporaryDirectory() as td:
            hf_cache = Path(td) / "hf"
            data_root = Path(td) / "data"
            hf_cache.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "data.zip"):
                extract_target_zip(hf_cache, data_root, target="rice")


if __name__ == "__main__":
    unittest.main()

"""Unit tests for scripts/notebook_helpers.py.

Helpers are TF-free by design so these tests run fast with only
numpy + pillow installed (TF is required only by the notebook itself).
"""

from __future__ import annotations

import io
import shutil
import sys
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

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


from scripts.notebook_helpers import validate_imagefolder  # noqa: E402


def _make_imagefolder(
    root: Path,
    classes: list[str],
    splits: Iterable[str] = ("train", "val", "test"),
    n_per_class: int = 2,
) -> None:
    """Create a minimal well-formed ImageFolder tree under root/<split>/<class>/."""
    for split in splits:
        for cls in classes:
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                (d / f"{cls}_{i}.jpg").write_bytes(b"")


RICE_CLASSES = ["chao", "com_chien", "com_chien_ga", "com_tam", "com_trang", "xoi"]


class ValidateImagefolderTests(unittest.TestCase):
    def test_accepts_well_formed_tree(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            _make_imagefolder(root, RICE_CLASSES)
            validate_imagefolder(root, RICE_CLASSES)  # should not raise

    def test_rejects_missing_class_folder(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            _make_imagefolder(root, RICE_CLASSES)
            # remove one class from train
            shutil.rmtree(root / "train" / "xoi")
            with self.assertRaisesRegex(ValueError, "missing class"):
                validate_imagefolder(root, RICE_CLASSES)

    def test_rejects_unexpected_class_folder(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            _make_imagefolder(root, RICE_CLASSES)
            (root / "train" / "WHO_PUT_THIS_HERE").mkdir()
            (root / "train" / "WHO_PUT_THIS_HERE" / "x.jpg").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "unexpected class"):
                validate_imagefolder(root, RICE_CLASSES)

    def test_rejects_empty_class_folder(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            _make_imagefolder(root, RICE_CLASSES)
            for p in (root / "train" / "xoi").iterdir():
                p.unlink()
            with self.assertRaisesRegex(ValueError, "empty"):
                validate_imagefolder(root, RICE_CLASSES)

    def test_rejects_non_image_file(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            _make_imagefolder(root, RICE_CLASSES)
            (root / "train" / "xoi" / "notes.txt").write_text("garbage")
            with self.assertRaisesRegex(ValueError, "non-image"):
                validate_imagefolder(root, RICE_CLASSES)

    def test_raises_on_missing_split_dir(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            _make_imagefolder(root, RICE_CLASSES, splits=("train", "val"))
            with self.assertRaisesRegex(ValueError, "missing split"):
                validate_imagefolder(root, RICE_CLASSES)


from scripts.notebook_helpers import compute_class_weights  # noqa: E402


class ComputeClassWeightsTests(unittest.TestCase):
    def test_balanced_input_yields_unit_weights(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            _make_imagefolder(root, RICE_CLASSES, n_per_class=10)
            weights = compute_class_weights(root, RICE_CLASSES)
            self.assertEqual(set(weights.keys()), set(range(len(RICE_CLASSES))))
            for i in range(len(RICE_CLASSES)):
                self.assertAlmostEqual(weights[i], 1.0, places=6)

    def test_underrepresented_class_gets_higher_weight(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            _make_imagefolder(root, RICE_CLASSES, n_per_class=10)
            # Strip xoi (last class) down to 1 sample in train
            xoi_train = root / "train" / "xoi"
            files = sorted(xoi_train.iterdir())
            for f in files[1:]:
                f.unlink()
            weights = compute_class_weights(root, RICE_CLASSES)
            xoi_idx = RICE_CLASSES.index("xoi")
            other_idx = RICE_CLASSES.index("com_tam")
            self.assertGreater(weights[xoi_idx], weights[other_idx])
            self.assertGreater(weights[xoi_idx], 1.0)
            self.assertLess(weights[other_idx], 1.0)

    def test_only_train_split_is_used(self) -> None:
        # val/test imbalance should not affect weights
        with TemporaryDirectory() as td:
            root = Path(td)
            _make_imagefolder(root, RICE_CLASSES, n_per_class=10)
            # Make val wildly imbalanced; train stays balanced
            for f in sorted((root / "val" / "xoi").iterdir())[1:]:
                f.unlink()
            weights = compute_class_weights(root, RICE_CLASSES)
            for i in range(len(RICE_CLASSES)):
                self.assertAlmostEqual(weights[i], 1.0, places=6)


import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from scripts.notebook_helpers import representative_dataset_gen  # noqa: E402


def _write_real_jpegs(root: Path, classes: list[str], n_per_class: int = 3) -> None:
    """Overwrite the empty .jpg files with real 32x32 random images.

    representative_dataset_gen actually opens images, so they must be valid.
    """
    rng = np.random.default_rng(0)
    for split in ("train", "val", "test"):
        for cls in classes:
            cls_dir = root / split / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                arr = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
                Image.fromarray(arr).save(cls_dir / f"{cls}_{i}.jpg", "JPEG")


class RepresentativeDatasetGenTests(unittest.TestCase):
    def test_yields_n_uint8_batches_of_correct_shape(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            _write_real_jpegs(root, RICE_CLASSES, n_per_class=5)
            gen_fn = representative_dataset_gen(root, imgsz=224, n=7, seed=0)
            batches = list(gen_fn())
            self.assertEqual(len(batches), 7)
            for b in batches:
                self.assertIsInstance(b, list)
                self.assertEqual(len(b), 1)
                arr = b[0]
                self.assertEqual(arr.shape, (1, 224, 224, 3))
                self.assertEqual(arr.dtype, np.uint8)

    def test_deterministic_with_seed(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            _write_real_jpegs(root, RICE_CLASSES, n_per_class=5)
            gen1 = list(representative_dataset_gen(root, imgsz=64, n=4, seed=42)())
            gen2 = list(representative_dataset_gen(root, imgsz=64, n=4, seed=42)())
            for a, b in zip(gen1, gen2):
                np.testing.assert_array_equal(a[0], b[0])

    def test_caps_at_available_train_images(self) -> None:
        # Only 6 classes * 2 train images = 12 total available
        with TemporaryDirectory() as td:
            root = Path(td)
            _write_real_jpegs(root, RICE_CLASSES, n_per_class=2)
            gen_fn = representative_dataset_gen(root, imgsz=64, n=200, seed=0)
            batches = list(gen_fn())
            self.assertEqual(len(batches), 12)


if __name__ == "__main__":
    unittest.main()

"""Pure helpers for the L2 EfficientNet Lite training notebook.

These functions are TF-free by design so they can be unit-tested locally
without a multi-GB TensorFlow install. The notebook generator embeds this
file's source verbatim into one notebook cell so the notebook stays
self-contained when run in Colab.
"""

from __future__ import annotations

import random
import shutil
import zipfile
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def extract_target_zip(
    hf_cache,
    data_root,
    target: str,
    force: bool = False,
) -> None:
    """Unzip hf_cache/l2_<target>/data.zip into data_root/l2_<target>/.

    Expected resulting tree:
        data_root/l2_<target>/
          train/<class_name>/<image>.<ext>
          val/<class_name>/<image>.<ext>
          test/<class_name>/<image>.<ext>

    Skip extraction when `force` is False and data_root/l2_<target>/train/
    already has at least one non-empty class subdirectory.

    Raises FileNotFoundError if the zip is missing and extraction needs to happen.
    """
    hf_cache = Path(hf_cache)
    data_root = Path(data_root)
    target_dir = data_root / f"l2_{target}"

    def already_populated() -> bool:
        train_dir = target_dir / "train"
        if not train_dir.is_dir():
            return False
        for cls_dir in train_dir.iterdir():
            if cls_dir.is_dir() and any(cls_dir.iterdir()):
                return True
        return False

    if not force and already_populated():
        print(f"[extract] {target_dir} already populated, skipping")
        return

    zip_path = hf_cache / f"l2_{target}" / "data.zip"
    if not zip_path.is_file():
        raise FileNotFoundError(f"data.zip not found at {zip_path}")

    if force and target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {zip_path} -> {target_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)


def validate_imagefolder(
    data_dir,
    class_names: list[str],
    splits: Iterable[str] = ("train", "val", "test"),
) -> None:
    """Walk data_dir/<split>/<class>/ for every split and assert well-formedness.

    Raises ValueError on the first violation.

    Per split, asserts:
      - The split dir exists.
      - The set of class subdirs exactly matches class_names.
      - Each class subdir is non-empty.
      - Every file under each class subdir has an allowed image extension.

    On success, prints a per-split, per-class count table and the imbalance
    ratio max(count) / min(count).
    """
    data_dir = Path(data_dir)
    expected = set(class_names)

    for split in splits:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            raise ValueError(f"[{split}] missing split dir: {split_dir}")

        present = {p.name for p in split_dir.iterdir() if p.is_dir()}
        missing = expected - present
        if missing:
            raise ValueError(
                f"[{split}] missing class folder(s): {sorted(missing)}"
            )
        extra = present - expected
        if extra:
            raise ValueError(
                f"[{split}] unexpected class folder(s): {sorted(extra)}"
            )

        counts: dict[str, int] = {}
        for cls in class_names:
            cls_dir = split_dir / cls
            files = [p for p in cls_dir.iterdir() if p.is_file()]
            if not files:
                raise ValueError(f"[{split}] class folder empty: {cls_dir}")
            for f in files:
                if f.suffix.lower() not in IMAGE_EXTS:
                    raise ValueError(
                        f"[{split}] non-image file: {f}  "
                        f"(allowed: {sorted(IMAGE_EXTS)})"
                    )
            counts[cls] = len(files)

        ratio = max(counts.values()) / min(counts.values())
        print(f"[{split}] total={sum(counts.values())}  imbalance(max/min)={ratio:.2f}")
        for cls in class_names:
            print(f"  {cls}: {counts[cls]}")


def compute_class_weights(data_dir, class_names: list[str]) -> dict[int, float]:
    """Return {class_index: weight} computed from the TRAIN split counts.

    weight_i = total / (num_classes * count_i)

    Balanced input yields weight ≈ 1.0 for every class. Under-represented
    classes get weight > 1; over-represented classes get weight < 1.
    Suitable for tf.keras.Model.fit(class_weight=...).
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    counts = []
    for cls in class_names:
        cls_dir = train_dir / cls
        n = sum(1 for p in cls_dir.iterdir() if p.is_file())
        if n == 0:
            raise ValueError(f"class {cls!r} has zero training samples")
        counts.append(n)
    total = sum(counts)
    nc = len(class_names)
    return {i: total / (nc * c) for i, c in enumerate(counts)}


def representative_dataset_gen(
    data_dir,
    imgsz: int,
    n: int = 200,
    seed: int = 0,
) -> Callable[[], Iterator[list[np.ndarray]]]:
    """Build a callable that yields up to `n` calibration batches.

    Each yielded item is `[ndarray of shape (1, imgsz, imgsz, 3), dtype=uint8]`
    drawn from random train images, resized via PIL.Image.LANCZOS.

    This shape matches the contract of tf.lite.TFLiteConverter.representative_dataset:
    a no-arg callable returning an iterable of input-list batches.

    If fewer than `n` train images exist, yields what's available (deterministic
    under the given seed).
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    all_images: list[Path] = []
    for cls_dir in sorted(train_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                all_images.append(p)

    rng = random.Random(seed)
    rng.shuffle(all_images)
    selected = all_images[:n]

    def _gen() -> Iterator[list[np.ndarray]]:
        for p in selected:
            with Image.open(p) as im:
                im = im.convert("RGB").resize((imgsz, imgsz), Image.LANCZOS)
            arr = np.asarray(im, dtype=np.uint8)[None, ...]
            yield [arr]

    return _gen

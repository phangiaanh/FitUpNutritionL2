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
from typing import Iterable, Iterator

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

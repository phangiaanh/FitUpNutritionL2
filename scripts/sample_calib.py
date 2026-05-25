"""Sample ~N calibration images from an ImageFolder train split.

Draws stratified round-robin across classes and copies selected images to
--out-dir, preserving class subdirectories. Run this once on the machine
that holds the dataset, then pass --out-dir to convert_tflite_int8.py.

Usage:
    python scripts/sample_calib.py \
        --target noodle \
        --data-dir ~/sources/outsource/datasets/l2_datasets/l2_noodle/data \
        --out-dir /tmp/calib_noodle \
        [--n 200] [--seed 0] [--split train]
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

TARGETS = {
    "noodle": [
        "banh_canh", "bun_bo_hue", "bun_cha", "bun_cha_ca", "bun_mam", "bun_rieu",
        "cao_lau", "hu_tieu", "mi", "mi_quang", "nui_xao_bo", "pho",
    ],
    "rice": ["chao", "com_chien", "com_chien_ga", "com_tam", "com_trang", "xoi"],
    "soup": ["bo_kho", "canh", "lau", "sup_cua"],
}


def sample_stratified(
    data_dir: Path,
    classes: list[str],
    split: str,
    n: int,
    seed: int,
) -> list[tuple[Path, str]]:
    split_dir = data_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    rng = random.Random(seed)
    per_class: dict[str, list[Path]] = {}
    for cls in classes:
        cls_dir = split_dir / cls
        if not cls_dir.is_dir():
            print(f"[warn] class dir missing: {cls_dir}")
            per_class[cls] = []
            continue
        imgs = [p for p in sorted(cls_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        rng.shuffle(imgs)
        per_class[cls] = imgs

    # Round-robin across classes until n reached or images exhausted
    selected: list[tuple[Path, str]] = []
    iters = {cls: iter(imgs) for cls, imgs in per_class.items()}
    while len(selected) < n:
        added_any = False
        for cls in classes:
            if len(selected) >= n:
                break
            try:
                selected.append((next(iters[cls]), cls))
                added_any = True
            except StopIteration:
                pass
        if not added_any:
            break

    return selected


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sample calibration images for TFLite INT8 conversion."
    )
    ap.add_argument("--target",   required=True, choices=sorted(TARGETS.keys()))
    ap.add_argument("--data-dir", required=True,
                    help="ImageFolder root containing train/val/test splits")
    ap.add_argument("--out-dir",  required=True,
                    help="Destination directory for sampled images")
    ap.add_argument("--n",        type=int, default=200,
                    help="Total images to sample (default: 200)")
    ap.add_argument("--seed",     type=int, default=0)
    ap.add_argument("--split",    default="train", choices=["train", "val"],
                    help="Split to sample from (default: train)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir  = Path(args.out_dir).expanduser().resolve()
    classes  = TARGETS[args.target]

    samples = sample_stratified(data_dir, classes, args.split, args.n, args.seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    for img_path, cls in samples:
        dest_dir = out_dir / cls
        dest_dir.mkdir(exist_ok=True)
        shutil.copy2(img_path, dest_dir / img_path.name)

    counts = Counter(cls for _, cls in samples)
    print(f"\nSampled {len(samples)} images  →  {out_dir}")
    for cls in classes:
        print(f"  {cls}: {counts.get(cls, 0)}")


if __name__ == "__main__":
    main()

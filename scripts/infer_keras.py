"""Local sanity test for a downloaded best.keras file.

Usage:
    python scripts/infer_keras.py --target rice --model /path/to/best.keras --image food.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tf_keras
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load_model_compat(path: str):
    # The model config contains 'optional' in InputLayer, added in a newer TF
    # save but not accepted by tf_keras's from_config. Strip it on the way in.
    _orig = tf_keras.layers.InputLayer.from_config.__func__

    @classmethod  # type: ignore[misc]
    def _patched(cls, config):
        config.pop("optional", None)
        return _orig(cls, config)

    tf_keras.layers.InputLayer.from_config = _patched
    try:
        return tf_keras.models.load_model(path)
    finally:
        tf_keras.layers.InputLayer.from_config = classmethod(_orig)


TARGETS = {
    "noodle": dict(imgsz=260, classes=[
        "banh_canh", "bun_bo_hue", "bun_cha", "bun_cha_ca", "bun_mam", "bun_rieu",
        "cao_lau", "hu_tieu", "mi", "mi_quang", "nui_xao_bo", "pho",
    ]),
    "rice":   dict(imgsz=240, classes=[
        "chao", "com_chien", "com_chien_ga", "com_tam", "com_trang", "xoi",
    ]),
    "soup":   dict(imgsz=224, classes=[
        "bo_kho", "canh", "lau", "sup_cua",
    ]),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=sorted(TARGETS.keys()))
    ap.add_argument("--model",  required=True, help="Path to best.keras")
    ap.add_argument("--image",  required=True, help="Path to input image")
    ap.add_argument("--topk",   type=int, default=3)
    args = ap.parse_args()

    cfg   = TARGETS[args.target]
    model = _load_model_compat(args.model)

    img = Image.open(args.image).convert("RGB").resize(
        (cfg["imgsz"], cfg["imgsz"]), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.uint8)[None, ...]
    probs = model.predict(arr, verbose=0)[0]

    print(f"Image  : {args.image}")
    print(f"Model  : {args.model}  (target={args.target}, imgsz={cfg['imgsz']})")
    top = probs.argsort()[-args.topk:][::-1]
    print(f"\nTop-{args.topk}:")
    for rank, i in enumerate(top, 1):
        print(f"  {rank}. {cfg['classes'][i]:20s}  conf={probs[i]:.4f}")


if __name__ == "__main__":
    main()

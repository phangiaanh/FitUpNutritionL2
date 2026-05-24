"""Local sanity test for a deployed l2_<target>_int8.tflite.

Reads class labels and preprocessing parameters embedded in the .tflite via
tflite_support; asserts the uint8 I/O contract; prints top-k predictions.

Usage:
    python scripts/infer_tflite.py --tflite l2_rice_int8.tflite --image food.jpg
"""

from __future__ import annotations

import argparse

import numpy as np
import tensorflow as tf
from PIL import Image


def _read_metadata(tflite_path: str) -> list[str]:
    """Pull class labels from the .tflite's embedded metadata."""
    from tflite_support import metadata as _md

    displayer = _md.MetadataDisplayer.with_model_file(tflite_path)
    label_file = displayer.get_packed_associated_file_list()[0]
    labels = displayer.get_associated_file_buffer(label_file).decode("utf-8").splitlines()
    return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True, help="Path to l2_<target>_int8.tflite")
    ap.add_argument("--image",  required=True, help="Path to input image")
    ap.add_argument("--topk",   type=int, default=3)
    args = ap.parse_args()

    labels = _read_metadata(args.tflite)
    print(f"Embedded labels ({len(labels)}): {labels}")

    interp = tf.lite.Interpreter(model_path=args.tflite)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    assert inp["dtype"] == np.uint8, f"expected uint8 input, got {inp['dtype']}"
    assert out["dtype"] == np.uint8, f"expected uint8 output, got {out['dtype']}"

    _, h, w, _ = inp["shape"]
    img = Image.open(args.image).convert("RGB").resize((w, h), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.uint8)[None, ...]
    interp.set_tensor(inp["index"], arr)
    interp.invoke()
    probs_q = interp.get_tensor(out["index"])[0]
    scale, zp = out["quantization"]
    probs = (probs_q.astype(np.float32) - zp) * scale

    print(f"\nImage  : {args.image}  ({w}x{h})")
    print(f"Model  : {args.tflite}")
    top = probs.argsort()[-args.topk:][::-1]
    print(f"\nTop-{args.topk}:")
    for rank, i in enumerate(top, 1):
        print(f"  {rank}. {labels[i]:20s}  conf={probs[i]:.4f}")


if __name__ == "__main__":
    main()

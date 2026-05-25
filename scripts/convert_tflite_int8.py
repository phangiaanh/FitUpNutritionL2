"""Convert best.keras to a full-integer INT8 TFLite model (uint8 I/O).

The model is loaded via tf_keras (Keras 2 compat layer) because best.keras
was saved with that backend. Labels and preprocessing metadata are embedded
into the .tflite via tflite-support so the mobile app can read them without
a sidecar file.

Version requirements (see VERSIONS note at bottom):
    python          3.10 – 3.12   (tflite-support has no 3.13 wheel)
    tensorflow      2.21.0
    tf_keras        2.21.0
    tflite-support  0.4.4
    pillow         >= 10.0.0
    numpy          >= 1.26.0

Typical workflow:
    # 1. Sample calibration images (run on the machine with the dataset)
    python scripts/sample_calib.py \\
        --target noodle \\
        --data-dir ~/sources/outsource/datasets/l2_datasets/l2_noodle/data \\
        --out-dir /tmp/calib_noodle

    # 2. Convert (run where best.keras lives)
    python scripts/convert_tflite_int8.py \\
        --target noodle \\
        --model  /path/to/best.keras \\
        --calib-dir /tmp/calib_noodle \\
        --out    l2_noodle_int8.tflite
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf_keras
from PIL import Image


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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

TARGETS = {
    "noodle": dict(imgsz=260, classes=[
        "banh_canh", "bun_bo_hue", "bun_cha", "bun_cha_ca", "bun_mam", "bun_rieu",
        "cao_lau", "hu_tieu", "mi", "mi_quang", "nui_xao_bo", "pho",
    ]),
    "rice": dict(imgsz=240, classes=[
        "chao", "com_chien", "com_chien_ga", "com_tam", "com_trang", "xoi",
    ]),
    "soup": dict(imgsz=224, classes=[
        "bo_kho", "canh", "lau", "sup_cua",
    ]),
}


def build_representative_dataset(calib_dir: Path, imgsz: int):
    """Return a no-arg callable that yields (1, H, W, 3) uint8 calibration batches."""
    images = sorted(
        p for p in calib_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        raise FileNotFoundError(f"No images found under {calib_dir}")
    print(f"[calib]   {len(images)} images from {calib_dir}")

    def _gen():
        for p in images:
            with Image.open(p) as im:
                arr = np.asarray(
                    im.convert("RGB").resize((imgsz, imgsz), Image.LANCZOS),
                    dtype=np.uint8,
                )[None, ...]
            yield [arr]

    return _gen


def write_labels_sidecar(tflite_path: str, classes: list[str]) -> str:
    """Always write a <stem>_labels.txt next to the .tflite as a fallback."""
    sidecar = str(Path(tflite_path).with_suffix("")) + "_labels.txt"
    Path(sidecar).write_text("\n".join(classes) + "\n", encoding="utf-8")
    return sidecar


def embed_metadata(tflite_path: str, classes: list[str]) -> bool:
    """Embed labels into the .tflite via tflite-support. Returns True on success."""
    try:
        from tflite_support.metadata_writers import writer_utils
        from tflite_support.metadata_writers.image_classifier import (
            MetadataWriter as ImageClassifierWriter,
        )
    except ModuleNotFoundError:
        return False

    with tempfile.NamedTemporaryFile(
        suffix=".txt", delete=False, mode="w", encoding="utf-8"
    ) as f:
        f.write("\n".join(classes) + "\n")
        labels_tmp = f.name

    try:
        writer = ImageClassifierWriter.create_for_inference(
            writer_utils.load_file(tflite_path),
            input_norm_mean=[127.5],
            input_norm_std=[127.5],
            label_file_paths=[labels_tmp],
        )
        writer_utils.save_file(writer.populate(), tflite_path)
    finally:
        Path(labels_tmp).unlink(missing_ok=True)

    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert best.keras → INT8 TFLite with embedded metadata."
    )
    ap.add_argument("--target",    required=True, choices=sorted(TARGETS.keys()))
    ap.add_argument("--model",     required=True, help="Path to best.keras")
    ap.add_argument("--calib-dir", required=True,
                    help="Directory of calibration images (output of sample_calib.py)")
    ap.add_argument("--out",       default=None,
                    help="Output path (default: ./l2_<target>_int8.tflite)")
    args = ap.parse_args()

    cfg       = TARGETS[args.target]
    out_path  = args.out or f"l2_{args.target}_int8.tflite"
    calib_dir = Path(args.calib_dir).expanduser().resolve()

    # --- load ----------------------------------------------------------------
    print(f"[load]    {args.model}")
    model = _load_model_compat(args.model)
    print(f"[load]    OK — {model.name}  "
          f"input={model.input_shape}  output={model.output_shape}")

    # --- convert -------------------------------------------------------------
    rep_dataset = build_representative_dataset(calib_dir, cfg["imgsz"])

    print("[convert] running full-integer PTQ …")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations             = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset    = rep_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type      = tf.uint8
    converter.inference_output_type     = tf.uint8
    tflite_bytes = converter.convert()
    Path(out_path).write_bytes(tflite_bytes)
    print(f"[convert] {out_path}  ({len(tflite_bytes) / 1024:.1f} KB)")

    # --- metadata ------------------------------------------------------------
    sidecar = write_labels_sidecar(out_path, cfg["classes"])
    print(f"[meta]    labels sidecar → {sidecar}")
    print("[meta]    embedding labels into .tflite …")
    if embed_metadata(out_path, cfg["classes"]):
        print("[meta]    embedded OK (tflite-support)")
    else:
        print("[meta]    tflite-support not available — skipped embedded metadata")
        print(f"[meta]    use sidecar {sidecar} for class labels")

    # --- smoke test ----------------------------------------------------------
    interp = tf.lite.Interpreter(model_path=out_path)
    interp.allocate_tensors()
    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    assert inp_det["dtype"] == np.uint8, \
        f"expected uint8 input, got {inp_det['dtype']}"
    assert out_det["dtype"] == np.uint8, \
        f"expected uint8 output, got {out_det['dtype']}"
    print(f"[verify]  dtype OK — input={inp_det['dtype']}  output={out_det['dtype']}")
    print(f"\nDone → {out_path}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# VERSIONS
# ---------------------------------------------------------------------------
# tflite-support does NOT publish wheels for Python 3.13. Use Python 3.10–3.12.
#
# Pinned stack (match training env to avoid .keras format mismatches):
#   python          3.10 – 3.12
#   tensorflow      2.21.0
#   tf_keras        2.21.0        # loads best.keras (saved with Keras 2 API)
#   tflite-support  0.4.4         # metadata embedding
#   pillow         >= 10.0.0
#   numpy          >= 1.26.0
#
# Install:
#   pip install tensorflow==2.21.0 tf_keras==2.21.0 "tflite-support==0.4.4" \
#               "pillow>=10.0.0" "numpy>=1.26.0"

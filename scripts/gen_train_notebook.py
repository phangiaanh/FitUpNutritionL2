#!/usr/bin/env python3
"""Generate notebooks/train_l2_efficientnet_lite.ipynb.

The notebook is built up from a list of cells defined inline below. Helper
functions used inside the notebook live in scripts/notebook_helpers.py and
are embedded verbatim into one cell so the notebook stays self-contained
when run in Colab.

The user sets a TARGET ∈ {"noodle", "rice", "soup"} in an early cell and
runs the notebook once per target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "notebooks" / "train_l2_efficientnet_lite.ipynb"
HELPERS_PATH = REPO_ROOT / "scripts" / "notebook_helpers.py"


def jl(text: str) -> list[str]:
    lines = text.split("\n")
    return [ln + "\n" for ln in lines]


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": jl(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": jl(text),
        "outputs": [],
        "execution_count": None,
    }


def build_cells() -> list[dict]:
    helpers_source = HELPERS_PATH.read_text()
    cells: list[dict] = []

    # Cell 1: Markdown header
    cells.append(markdown(
        r"""# L2 Vietnamese-dish fine-grained classifier - EfficientNet Lite

Trains one **EfficientNet Lite** classifier per L1 food category. Run the
notebook once for each `TARGET` ∈ `{"noodle", "rice", "soup"}`.

### The stack

```
L1 (YOLO11s)      detects coarse category and crops the box
   │
   ▼
L2 (this notebook) classifies the specific Vietnamese dish inside the crop
```

### Dataset on Hugging Face
`WatermelonAnh/FoodClassifierL2` — one `data.zip` per L2 target
([HF dataset page](https://huggingface.co/datasets/WatermelonAnh/FoodClassifierL2)).

Each zip unpacks to standard ImageFolder layout:
`train|val|test / <class_name> / <image>.<ext>`.

### Deployment target
Final artifact is an INT8 `.tflite` with **uint8 input + uint8 output** and
class labels embedded via `tflite-support`. An FP16 `.tflite` is also produced
as a fallback. Checkpoints and exports persist to Google Drive so a Colab
disconnect never destroys progress."""
    ))

    # Cell 2: pip install
    cells.append(code(
        "%%capture\n"
        "%pip install -q --upgrade tensorflow-hub tflite-support huggingface_hub pillow"
    ))

    # Cell 3: Imports + GPU check
    cells.append(code(
        r"""from __future__ import annotations

import os
import random
import shutil
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from huggingface_hub import login, snapshot_download

gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    raise SystemExit(
        "GPU required - Runtime -> Change runtime type -> GPU (A100 recommended)."
    )

print("TF:", tf.__version__)
print("GPU:", gpus[0].name)"""
    ))

    # Cell 4: Drive mount + HF login
    cells.append(code(
        r"""try:
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")
except ImportError:
    print("Outside Colab - ensure DRIVE_ROOT exists on the host filesystem.")

TOKEN = os.environ.get("HF_TOKEN", "").strip()
if TOKEN:
    login(token=TOKEN)
else:
    login()
print("Hugging Face login OK.")"""
    ))

    # Cell 5: Config
    cells.append(code(
        r"""DRIVE_ROOT  = "/content/drive/MyDrive/FitUpNutritionL2"
RUNS_DIR    = os.path.join(DRIVE_ROOT, "runs")
EXPORTS_DIR = os.path.join(DRIVE_ROOT, "exports")

HF_DATASET_REPO = "WatermelonAnh/FoodClassifierL2"
HF_CACHE        = "/content/hf_cache"
DATA_ROOT       = "/content/l2_data"

# === The one knob to change between runs ===========================
TARGET = "noodle"           # "noodle" | "rice" | "soup"
# ===================================================================

TARGETS = {
    "noodle": dict(
        variant="lite2", imgsz=260, num_classes=12,
        tfhub_url="https://tfhub.dev/tensorflow/efficientnet/lite2/feature-vector/2",
        classes=[
            "banh_canh", "bun_bo_hue", "bun_cha", "bun_cha_ca", "bun_mam", "bun_rieu",
            "cao_lau", "hu_tieu", "mi", "mi_quang", "nui_xao_bo", "pho",
        ],
    ),
    "rice": dict(
        variant="lite1", imgsz=240, num_classes=6,
        tfhub_url="https://tfhub.dev/tensorflow/efficientnet/lite1/feature-vector/2",
        classes=[
            "chao", "com_chien", "com_chien_ga", "com_tam", "com_trang", "xoi",
        ],
    ),
    "soup": dict(
        variant="lite0", imgsz=224, num_classes=4,
        tfhub_url="https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2",
        classes=[
            "bo_kho", "canh", "lau", "sup_cua",
        ],
    ),
}

cfg          = TARGETS[TARGET]
DATA_DIR     = f"{DATA_ROOT}/l2_{TARGET}"
RUN_DIR      = f"{RUNS_DIR}/l2_{TARGET}"
BACKUP_DIR   = f"{RUN_DIR}/backup"
BEST_KERAS   = f"{RUN_DIR}/best.keras"
LABELS_TXT   = f"{RUN_DIR}/labels.txt"
TRAINING_LOG = f"{RUN_DIR}/training_log.csv"
FORCE_REDOWNLOAD = False

# Training hyperparameters
BATCH_SIZE       = 64
STAGE1_EPOCHS    = 8
STAGE1_LR        = 1e-3
STAGE1_PATIENCE  = 4
STAGE2_EPOCHS    = 50
STAGE2_LR        = 1e-4
STAGE2_LR_FLOOR  = 1e-6
STAGE2_PATIENCE  = 10
DROPOUT          = 0.2
CALIB_SAMPLES    = 200

os.makedirs(RUN_DIR,     exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)
print(f"Config OK. TARGET={TARGET}  variant={cfg['variant']}  "
      f"imgsz={cfg['imgsz']}  num_classes={cfg['num_classes']}")"""
    ))

    # Cell 6: Helpers (embedded notebook_helpers.py)
    cells.append(markdown(
        "### Helpers\n\n"
        "The next cell is the verbatim contents of `scripts/notebook_helpers.py` "
        "from the repo. Re-run this cell after pulling helper updates."
    ))
    cells.append(code(helpers_source.rstrip()))

    # Cell 7: Markdown + dataset download/extract
    cells.append(markdown(
        "### Dataset: download from HF, extract zip, validate, build tf.data"
    ))
    cells.append(code(
        r"""if FORCE_REDOWNLOAD or not (Path(HF_CACHE) / f"l2_{TARGET}" / "data.zip").is_file():
    snapshot_download(
        repo_id="WatermelonAnh/FoodClassifierL2",
        repo_type="dataset",
        local_dir=HF_CACHE,
        allow_patterns=[f"l2_{TARGET}/*"],
    )
else:
    print(f"[download] HF cache for l2_{TARGET} already populated, skipping.")

extract_target_zip(HF_CACHE, DATA_ROOT, target=TARGET, force=FORCE_REDOWNLOAD)
print("data dir:", DATA_DIR)"""
    ))

    # Cell 8: Pre-flight validation
    cells.append(markdown(
        "### Label validation\n\n"
        "Fails loudly if any class folder is missing, empty, or contains "
        "non-image files. Run this **before** training."
    ))
    cells.append(code(
        'validate_imagefolder(DATA_DIR, cfg["classes"])'
    ))

    # Cell 9: tf.data datasets
    cells.append(markdown("### tf.data pipelines (train / val / test)"))
    cells.append(code(
        r"""def _make_ds(split: str, *, shuffle: bool, cache: bool):
    ds = tf.keras.utils.image_dataset_from_directory(
        f"{DATA_DIR}/{split}",
        labels="inferred",
        label_mode="int",
        class_names=cfg["classes"],          # locks label order to TARGETS
        image_size=(cfg["imgsz"], cfg["imgsz"]),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=42,
    )
    if cache:
        ds = ds.cache()
    return ds.prefetch(tf.data.AUTOTUNE)

train_ds = _make_ds("train", shuffle=True,  cache=True)
val_ds   = _make_ds("val",   shuffle=False, cache=True)
test_ds  = _make_ds("test",  shuffle=False, cache=False)

# image_dataset_from_directory yields float32 in [0,255]; cast to uint8 for the
# model's uint8 Input contract.
def _to_uint8(x, y):
    return tf.cast(x, tf.uint8), y

train_ds = train_ds.map(_to_uint8, num_parallel_calls=tf.data.AUTOTUNE)
val_ds   = val_ds.map(_to_uint8,   num_parallel_calls=tf.data.AUTOTUNE)
test_ds  = test_ds.map(_to_uint8,  num_parallel_calls=tf.data.AUTOTUNE)
print("Datasets ready.")"""
    ))

    # Cell 10: Build model
    cells.append(markdown(
        "### Build model\n\n"
        "EfficientNet Lite backbone from TF Hub (frozen for stage 1), with the "
        "augmentation block baked in as the first sub-block (train-time only, "
        "stripped on TFLite export). Input dtype is `uint8`; normalization to "
        "`[-1, 1]` happens inside the model."
    ))
    cells.append(code(
        r"""augmentation_block = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.RandomBrightness(0.1),
], name="augmentation")

tfhub_url = cfg["tfhub_url"]
imgsz     = cfg["imgsz"]
nc        = cfg["num_classes"]

inputs = tf.keras.Input(shape=(imgsz, imgsz, 3), dtype=tf.uint8)
x = tf.cast(inputs, tf.float32)
x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)
x = augmentation_block(x)
hub_layer = hub.KerasLayer(tfhub_url, trainable=False, name="efficientnet_lite_backbone")
features = hub_layer(x)
x = tf.keras.layers.Dropout(DROPOUT)(features)
outputs = tf.keras.layers.Dense(nc, activation="softmax", name="classifier_head")(x)
model = tf.keras.Model(inputs, outputs, name=f"l2_{TARGET}_efficientnet_{cfg['variant']}")
model.summary()"""
    ))

    return cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
            },
            "colab": {"provenance": []},
            "accelerator": "GPU",
        },
        "cells": build_cells(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    print("Wrote", args.out)


if __name__ == "__main__":
    main()

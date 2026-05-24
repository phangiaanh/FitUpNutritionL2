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
    if not lines:
        return []
    result = [ln + "\n" for ln in lines[:-1]]
    result.append(lines[-1])
    return result


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

    # Cell 11: Class weights
    cells.append(markdown(
        "### Class weights\n\n"
        "Inverse-frequency weights computed from the **train** split, passed to "
        "both `fit` calls."
    ))
    cells.append(code(
        r"""class_weight = compute_class_weights(DATA_DIR, cfg["classes"])
print("class_weight:")
for i, cls in enumerate(cfg["classes"]):
    print(f"  {i:2d} {cls:20s}  {class_weight[i]:.3f}")"""
    ))

    # Cell 12: Stage 1 - head only
    cells.append(markdown(
        "### Stage 1 - train head only (backbone frozen)\n\n"
        "Lets the random head settle without disturbing the pretrained backbone "
        "features. Resume via `BackupAndRestore` if a previous Colab session "
        "died mid-training."
    ))
    cells.append(code(
        r"""def _common_callbacks(patience: int):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            BEST_KERAS,
            monitor="val_sparse_categorical_accuracy",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.BackupAndRestore(backup_dir=BACKUP_DIR),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_sparse_categorical_accuracy",
            mode="max",
            patience=patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(TRAINING_LOG, append=True),
    ]


# Stage 1: head-only
hub_layer.trainable = False
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=STAGE1_LR),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(name="sparse_categorical_accuracy"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy"),
    ],
)

history_s1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=STAGE1_EPOCHS,
    class_weight=class_weight,
    callbacks=_common_callbacks(STAGE1_PATIENCE),
)"""
    ))

    # Cell 13: Stage 2 - full fine-tune with cosine LR
    cells.append(markdown(
        "### Stage 2 - full fine-tune (backbone unfrozen, cosine LR)\n\n"
        "**Re-compiling the model is mandatory** - flipping "
        "`trainable` is only picked up at compile time."
    ))
    cells.append(code(
        r"""# Stage 2: full fine-tune
hub_layer.trainable = True

# Compute steps_per_epoch for the cosine schedule
steps_per_epoch = int(np.ceil(sum(1 for _ in train_ds.unbatch())  # cached, cheap
                              / BATCH_SIZE))
cosine_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=STAGE2_LR,
    decay_steps=steps_per_epoch * STAGE2_EPOCHS,
    alpha=STAGE2_LR_FLOOR / STAGE2_LR,
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=cosine_schedule),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(name="sparse_categorical_accuracy"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy"),
    ],
)

history_s2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=STAGE1_EPOCHS + STAGE2_EPOCHS,
    initial_epoch=STAGE1_EPOCHS,
    class_weight=class_weight,
    callbacks=_common_callbacks(STAGE2_PATIENCE),
)"""
    ))

    # Cell 14: Evaluate on test split
    cells.append(markdown(
        "### Evaluate on test split\n\n"
        "Reports overall top-1 / top-3 accuracy, per-class precision/recall/F1, "
        "and a confusion matrix (raw counts + row-normalized)."
    ))
    cells.append(code(
        r"""import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

best_model = tf.keras.models.load_model(BEST_KERAS)
test_loss, test_top1, test_top3 = best_model.evaluate(test_ds, verbose=1)
print(f"\n[{TARGET}] test top-1={test_top1:.4f}  top-3={test_top3:.4f}")

y_true, y_pred = [], []
for x, y in test_ds:
    probs = best_model.predict(x, verbose=0)
    y_true.append(y.numpy())
    y_pred.append(np.argmax(probs, axis=1))
y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

# Per-class report
report_txt = classification_report(
    y_true, y_pred,
    labels=list(range(cfg["num_classes"])),
    target_names=cfg["classes"],
    digits=4,
    zero_division=0,
)
print("\n" + report_txt)
Path(f"{RUN_DIR}/per_class_report.txt").write_text(report_txt)

# Confusion matrix: raw counts + row-normalized
cm = confusion_matrix(y_true, y_pred, labels=list(range(cfg["num_classes"])))
cm_norm = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, mat, title, fmt in [
    (axes[0], cm,      "raw counts",   "d"),
    (axes[1], cm_norm, "row-normalized", ".2f"),
]:
    im = ax.imshow(mat, cmap="Blues")
    ax.set_title(f"{TARGET}: confusion matrix - {title}")
    ax.set_xticks(range(cfg["num_classes"]))
    ax.set_yticks(range(cfg["num_classes"]))
    ax.set_xticklabels(cfg["classes"], rotation=45, ha="right")
    ax.set_yticklabels(cfg["classes"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(cfg["num_classes"]):
        for j in range(cfg["num_classes"]):
            ax.text(j, i, format(mat[i, j], fmt),
                    ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() * 0.5 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(f"{RUN_DIR}/confusion_matrix.png", dpi=120)
plt.show()

# Training curves
if Path(TRAINING_LOG).is_file():
    import csv
    epochs, loss, val_loss, acc, val_acc = [], [], [], [], []
    with open(TRAINING_LOG) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            loss.append(float(row["loss"]))
            val_loss.append(float(row["val_loss"]))
            acc.append(float(row["sparse_categorical_accuracy"]))
            val_acc.append(float(row["val_sparse_categorical_accuracy"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, loss, label="train"); axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_title("loss");     axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(epochs, acc,  label="train"); axes[1].plot(epochs, val_acc,  label="val")
    axes[1].set_title("accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{RUN_DIR}/training_curves.png", dpi=120)
    plt.show()

if test_top1 < 0.70:
    print(f"\nWARNING: test top-1={test_top1:.4f} < 0.70 - this run is weak; "
          f"export still proceeds but the artifact may not be ship-quality.")"""
    ))

    # Cell 15: INT8 TFLite export + metadata
    cells.append(markdown(
        "### TFLite export - INT8 with uint8 I/O (primary)\n\n"
        "Full integer quantization. Mobile app passes raw uint8 pixels and "
        "reads quantized output via the metadata embedded into the .tflite."
    ))
    cells.append(code(
        r"""from tflite_support.metadata_writers import writer_utils
from tflite_support.metadata_writers.image_classifier import MetadataWriter as ImageClassifierWriter

# Write labels.txt (consumed by the metadata writer)
labels_path = write_labels_txt(LABELS_TXT, cfg["classes"])

# Convert with INT8 PTQ + uint8 I/O
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter.optimizations          = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen(
    DATA_DIR, imgsz=cfg["imgsz"], n=CALIB_SAMPLES, seed=0,
)
converter.target_spec.supported_ops  = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type   = tf.uint8
converter.inference_output_type  = tf.uint8
tflite_int8 = converter.convert()
int8_path = f"{EXPORTS_DIR}/l2_{TARGET}_int8.tflite"
Path(int8_path).write_bytes(tflite_int8)
print(f"INT8 TFLite -> {int8_path}  ({len(tflite_int8) / 1024:.1f} KB)")

# Embed labels + preprocessing metadata
writer = ImageClassifierWriter.create_for_inference(
    writer_utils.load_file(int8_path),
    input_norm_mean=[127.5],
    input_norm_std=[127.5],
    label_file_paths=[str(labels_path)],
)
writer_utils.save_file(writer.populate(), int8_path)
print(f"Embedded labels + preprocessing metadata into {int8_path}")"""
    ))

    # Cell 16: FP16 fallback export
    cells.append(markdown(
        "### TFLite export - FP16 (fallback)\n\n"
        "Half-precision fallback for accuracy debugging if INT8 ever shows "
        "per-class regressions on-device. No embedded metadata - this is not "
        "the deploy artifact."
    ))
    cells.append(code(
        r"""converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter.optimizations               = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_fp16 = converter.convert()
fp16_path = f"{EXPORTS_DIR}/l2_{TARGET}_fp16.tflite"
Path(fp16_path).write_bytes(tflite_fp16)
print(f"FP16 TFLite -> {fp16_path}  ({len(tflite_fp16) / 1024:.1f} KB)")"""
    ))

    # Cell 17: Smoke test on INT8 TFLite
    cells.append(markdown(
        "### Inference smoke test (INT8 TFLite)\n\n"
        "Validates the deployed artifact end-to-end: dtype contract, "
        "dequantization, top-3 prediction on a random test image."
    ))
    cells.append(code(
        r"""interp = tf.lite.Interpreter(model_path=int8_path)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
assert inp["dtype"] == np.uint8, f"expected uint8 input, got {inp['dtype']}"
assert out["dtype"] == np.uint8, f"expected uint8 output, got {out['dtype']}"

# Pick one random test image (recursively, so any class is fair game)
test_files = sorted(Path(f"{DATA_DIR}/test").rglob("*"))
test_files = [p for p in test_files if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}]
sample = random.choice(test_files)
true_class = sample.parent.name

img = np.array(Image.open(sample).convert("RGB").resize(
    (cfg["imgsz"], cfg["imgsz"]), Image.LANCZOS), dtype=np.uint8)[None, ...]
interp.set_tensor(inp["index"], img)
interp.invoke()
probs_q = interp.get_tensor(out["index"])[0]
scale, zp = out["quantization"]
probs = (probs_q.astype(np.float32) - zp) * scale

top3 = probs.argsort()[-3:][::-1]
print(f"Sample : {sample}")
print(f"True   : {true_class}")
print("Top-3  :", [(cfg["classes"][i], float(probs[i])) for i in top3])

# Save annotated preview
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(Image.open(sample).convert("RGB"))
ax.set_title(
    f"true={true_class}\n"
    + " | ".join(f"{cfg['classes'][i]}:{probs[i]:.2f}" for i in top3),
    fontsize=10,
)
ax.axis("off")
smoke_png = f"{EXPORTS_DIR}/l2_{TARGET}_smoke_test.png"
fig.savefig(smoke_png, dpi=120, bbox_inches="tight")
plt.show()
print(f"Saved annotated preview to {smoke_png}")"""
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

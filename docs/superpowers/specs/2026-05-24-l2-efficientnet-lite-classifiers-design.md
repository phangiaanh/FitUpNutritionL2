# FitUpNutritionL2 — EfficientNet Lite fine-grained classifiers (noodle / rice / soup)

Date: 2026-05-24

## Goal

Train one **EfficientNet Lite** fine-grained classifier per L1 food category, starting with three categories (`noodle`, `rice`, `soup`). Each model consumes a crop produced by the L1 YOLO11s detector and names the specific Vietnamese dish. Final artifact per target is a quantized `.tflite` with embedded class labels, ready for on-device mobile inference.

L2 is an isolated project from L1 (different framework, different repo). L1 is referenced only for the Colab + Google Drive mount idiom, the Hugging Face `snapshot_download` pattern, and the `gen_train_notebook.py` / `notebook_helpers.py` repo layout.

## Constraints and priorities

- **Framework:** TensorFlow / Keras. EfficientNet Lite's native lineage is TF Hub; native `tf.lite.TFLiteConverter` is the cleanest path to mobile.
- **Compute:** Training on Google Colab with an A100 GPU. Notebook is the primary execution surface.
- **Deployment target:** INT8 `.tflite` per classifier, with **uint8 input + uint8 output** — the mobile app passes raw `[0,255]` pixel bytes and reads a quantized output vector. FP16 `.tflite` is also produced as a fallback for accuracy debugging.
- **Persistence:** Checkpoints, plots, exports, and `BackupAndRestore` state all land on Google Drive so a Colab disconnect never destroys progress.
- **Scope discipline:** Three targets only in the first cut (`noodle`, `rice`, `soup`). The notebook is wired so adding a fourth target later is a one-row change to the `TARGETS` table once its `data.zip` exists on Hugging Face.

## Targets

One classifier per L1 category. The variant is sized to the per-target class count (more classes ⇒ slightly bigger backbone):

| TARGET     | variant | imgsz | num_classes | classes                                                                                              |
| ---------- | ------- | ----- | ----------- | ---------------------------------------------------------------------------------------------------- |
| `noodle`   | Lite2   | 260   | 12          | banh_canh, bun_bo_hue, bun_cha, bun_cha_ca, bun_mam, bun_rieu, cao_lau, hu_tieu, mi, mi_quang, nui_xao_bo, pho |
| `rice`     | Lite1   | 240   | 6           | chao, com_chien, com_chien_ga, com_tam, com_trang, xoi                                              |
| `soup`     | Lite0   | 224   | 4           | bo_kho, canh, lau, sup_cua                                                                          |
| `beverage` | (Lite0) | (224) | (TBD)       | reserved — row filled in once `l2_beverage/data.zip` lands on Hugging Face                          |

**Class order is fixed in code.** It is the single source of truth for label indices; the filesystem ordering of class directories is not trusted at any point. The order baked into `TARGETS[...]["classes"]` is the order used to compile the model, build class weights, write `labels.txt`, embed labels into the TFLite, and produce the confusion matrix.

## Paths and config

Defined once in an early notebook cell:

```python
DRIVE_ROOT  = "/content/drive/MyDrive/FitUpNutritionL2"
RUNS_DIR    = os.path.join(DRIVE_ROOT, "runs")        # Keras checkpoints + plots, per target
EXPORTS_DIR = os.path.join(DRIVE_ROOT, "exports")     # .tflite files, per target

HF_DATASET_REPO = "WatermelonAnh/FoodClassifierL2"
HF_CACHE        = "/content/hf_cache"                  # whole-repo snapshot (one per session)
DATA_ROOT       = "/content/l2_data"                   # per-target unzipped trees go here

TARGET = "noodle"                                      # the one knob: "noodle" | "rice" | "soup"

TARGETS = {
    "noodle": dict(variant="lite2", imgsz=260, num_classes=12, classes=[
        "banh_canh", "bun_bo_hue", "bun_cha", "bun_cha_ca", "bun_mam", "bun_rieu",
        "cao_lau", "hu_tieu", "mi", "mi_quang", "nui_xao_bo", "pho",
    ]),
    "rice":   dict(variant="lite1", imgsz=240, num_classes=6,  classes=[
        "chao", "com_chien", "com_chien_ga", "com_tam", "com_trang", "xoi",
    ]),
    "soup":   dict(variant="lite0", imgsz=224, num_classes=4,  classes=[
        "bo_kho", "canh", "lau", "sup_cua",
    ]),
}

cfg              = TARGETS[TARGET]
DATA_DIR         = f"{DATA_ROOT}/l2_{TARGET}"          # train/val/test/<class>/<img>
RUN_DIR          = f"{RUNS_DIR}/l2_{TARGET}"
FORCE_REDOWNLOAD = False
```

One notebook is run three times (once per `TARGET`). Each run writes to a separate subfolder on Drive and does not collide with the others.

## Dataset

### Source

Hugging Face dataset repo `WatermelonAnh/FoodClassifierL2`. Layout on the hub:

```
WatermelonAnh/FoodClassifierL2/
├── l2_noodle/    { data.zip, dataset.yaml, build_stats.txt }
├── l2_rice/      { data.zip, dataset.yaml, build_stats.txt }
├── l2_soup/      { data.zip, dataset.yaml, build_stats.txt }
└── l2_beverage/  { dataset.yaml, build_stats.txt }   ← no data.zip yet; out of scope
```

Each `data.zip` unzips to:

```
l2_<target>/
├── train/<class_name>/<image_filename>.<ext>
├── val/<class_name>/<image_filename>.<ext>
└── test/<class_name>/<image_filename>.<ext>
```

Standard Keras `ImageFolder` layout — labels are inferred from directory names. `<ext>` is one of `{.jpg, .jpeg, .png, .webp, .bmp}`. `dataset.yaml` and `build_stats.txt` are present at the hub layer (provenance / sample counts) but are not consumed by the notebook — class names come from `TARGETS[...]["classes"]`, not the filesystem.

### Fetch flow

1. `snapshot_download(repo_id="WatermelonAnh/FoodClassifierL2", repo_type="dataset", local_dir=HF_CACHE, allow_patterns=[f"l2_{TARGET}/*"])` — only the current target's subfolder is pulled, to keep Colab disk usage small.
2. `extract_target_zip(HF_CACHE, DATA_ROOT, TARGET, force=FORCE_REDOWNLOAD)` (helper) unzips `HF_CACHE/l2_<TARGET>/data.zip` into `DATA_DIR`. Skips re-extraction if `DATA_DIR/train/<first_class>/` already contains files.
3. Re-running the cell when `DATA_DIR` is already populated is a no-op.

### Pre-flight validation (mandatory)

`validate_imagefolder(DATA_DIR, cfg["classes"], splits=("train","val","test"))` (helper, embedded into the notebook from `scripts/notebook_helpers.py`) walks every split:

- Each `<split>/<class>/` directory exists and contains ≥ 1 file. Empty class folders fail.
- The set of class subdirs under each split **exactly matches** `cfg["classes"]` — no extras, no missing. Catches dataset rename drift before training burns an hour.
- Every file under each class folder has an extension in `{.jpg, .jpeg, .png, .webp, .bmp}`. Other files are warned and skipped.
- Prints a per-split per-class count table and the imbalance ratio `max(count) / min(count)` per split, so it is visible up-front whether class weighting will matter.

Raises `ValueError` on the first violation. **Fail fast** — never start training on a malformed dataset.

## Model architecture

### Backbone source

EfficientNet Lite is not in `tf.keras.applications`. Pulled from **TensorFlow Hub**:

| variant | TF Hub URL                                                         | input |
| ------- | ------------------------------------------------------------------ | ----- |
| Lite0   | `https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2` | 224   |
| Lite1   | `https://tfhub.dev/tensorflow/efficientnet/lite1/feature-vector/2` | 240   |
| Lite2   | `https://tfhub.dev/tensorflow/efficientnet/lite2/feature-vector/2` | 260   |

These are headless feature-vector checkpoints (1280-d output) trained on ImageNet. Loaded via `hub.KerasLayer(url, trainable=False)` initially and flipped to `trainable=True` for stage 2.

### Graph

```python
inputs = tf.keras.Input(shape=(imgsz, imgsz, 3), dtype=tf.uint8)
x = tf.cast(inputs, tf.float32)
x = tf.keras.layers.Rescaling(1./127.5, offset=-1.0)(x)        # EN-Lite expects [-1, 1]
x = augmentation_block(x)                                       # train-time only
features = hub.KerasLayer(tfhub_url, trainable=False)(x)        # stage 1: frozen
x = tf.keras.layers.Dropout(0.2)(features)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)
```

The `uint8` input dtype here is training-time convenience: the `Rescaling` layer normalizes inside the model. At TFLite-export time, `inference_input_type=tf.uint8` aligns the exported model's I/O with this contract — the mobile app passes raw bytes without any preprocessing math.

### Augmentation block

A `tf.keras.Sequential` of:

```python
tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.RandomBrightness(0.1),
])
```

Baked in as a sub-block of the Keras model. These layers are no-ops at `model.predict` time and are stripped on TFLite export automatically — no manual mode switching needed.

`RandomFlip` is horizontal-only (vertical flip is unrealistic for food photographs). All other parameters are intentionally moderate; aggressive augmentation (mixup, RandAugment) can blur fine-grained signals between visually-similar dishes (e.g. `com_chien` vs `com_chien_ga`) and is deliberately omitted from the first cut.

## Training

### Two-stage schedule

| stage | epochs (max) | backbone trainable? | LR                        | optimizer            | early-stop patience |
| ----- | ------------ | ------------------- | ------------------------- | -------------------- | ------------------- |
| 1     | 8            | No (frozen)         | `1e-3` (flat)             | Adam                 | 4 (val accuracy)    |
| 2     | 50           | Yes (fully unfrozen)| `1e-4` → cosine to `1e-6` | Adam (re-compiled)   | 10 (val accuracy)   |

Stage 1 trains only the random head — keeps the pretrained backbone features stable while the head's large initial gradients settle. Stage 2 unfreezes the backbone for end-to-end fine-tuning at a low LR; cosine decay tapers the learning rate over the remaining epoch budget.

**Re-compile is mandatory between stages.** Toggling `hub_layer.trainable = True` does nothing unless `model.compile(...)` is called again afterwards (Keras snapshots the trainable-variable list at compile time). The notebook does this explicitly with a comment so it is not silently regressed by future edits.

### Loss, metrics, batch size

- Loss: `SparseCategoricalCrossentropy`.
- Metrics: `SparseCategoricalAccuracy` + `SparseTopKCategoricalAccuracy(k=3)`. Top-3 is informative for fine-grained dish recognition where the correct label often lands in 2nd or 3rd.
- Batch size: 64 (fits Lite2 @ 260 on A100 80 GB with headroom).

### Class imbalance

Computed from the train split before stage 1:

```python
class_weight = {i: total / (num_classes * count_i)
                for i, count_i in enumerate(train_counts_per_class)}
```

Passed to both `model.fit(...)` calls via `class_weight=class_weight`. The pre-flight validation cell prints the per-class counts and imbalance ratio so the actual weights are visible before training starts.

### Callbacks (both stages, same list, pointed at the same files)

- `ModelCheckpoint(RUN_DIR/best.keras, monitor="val_sparse_categorical_accuracy", save_best_only=True)` — the single source for evaluation and export.
- `BackupAndRestore(RUN_DIR/backup/)` — Keras's official resume mechanism. If Colab disconnects, re-running the notebook with the same `TARGET` resumes training from the last completed epoch. Both stages share the same backup directory; the callback handles stage boundaries transparently.
- `EarlyStopping(monitor="val_sparse_categorical_accuracy", patience=..., restore_best_weights=True)`.
- `CSVLogger(RUN_DIR/training_log.csv, append=True)` — continuous CSV across both stages, used for the training-curves plot.

### Per-target Drive output

```
runs/l2_<target>/
├── best.keras                ← val-acc-best checkpoint, the export source
├── backup/                   ← BackupAndRestore state (auto-managed)
├── training_log.csv          ← per-epoch metrics, both stages concatenated
├── training_curves.png       ← loss + accuracy curves, both stages
├── confusion_matrix.png      ← raw counts + row-normalized, side-by-side
└── per_class_report.txt      ← sklearn classification_report output
```

## Evaluation

Runs after stage 2 finishes, **before** any export. Loads `best.keras` (the val-best checkpoint, not the final-epoch weights).

```python
best_model = tf.keras.models.load_model(f"{RUN_DIR}/best.keras")

# Overall
test_loss, test_top1, test_top3 = best_model.evaluate(test_ds, verbose=1)

# Per-class
y_true, y_pred = [], []
for x, y in test_ds:
    y_true.append(y.numpy())
    y_pred.append(np.argmax(best_model.predict(x, verbose=0), axis=1))
y_true = np.concatenate(y_true);  y_pred = np.concatenate(y_pred)
# → sklearn.metrics.classification_report (saved to per_class_report.txt)
# → sklearn.metrics.confusion_matrix     (plotted to confusion_matrix.png)
```

Reported and persisted:

- Overall top-1 accuracy, top-3 accuracy (printed in the notebook).
- Per-class precision / recall / F1 / support (printed table + `per_class_report.txt`).
- Confusion matrix in two views (raw counts + row-normalized percentages), side-by-side in `confusion_matrix.png`.
- Training curves (loss + accuracy across both stages), sourced from `training_log.csv`, in `training_curves.png`.
- A single grep-able summary line: `[noodle] test top-1=0.913  top-3=0.984`.

**Soft floor:** if test top-1 < `0.70`, the cell prints a `WARNING:` line. Export proceeds anyway (so the artifact exists to inspect), but the warning surfaces "this run is bad" before anyone ships it.

Dependencies for this cell: `sklearn.metrics` + `matplotlib` only. Both ship in Colab by default; pinned in `requirements.txt` for local dev.

## TFLite export

Two artifacts per run, both produced from `best.keras`.

### Primary: full integer INT8, uint8 I/O

```python
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter.optimizations          = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen(DATA_DIR, cfg["imgsz"], n=200)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type   = tf.uint8
converter.inference_output_type  = tf.uint8
tflite_int8 = converter.convert()
```

`representative_dataset_gen(data_dir, imgsz, n=200, seed=0)` (helper) samples 200 random images from `data_dir/train/**/`, resizes each to `(imgsz, imgsz)` as `uint8`, and yields them one-at-a-time as `[tf.constant(img[None], dtype=tf.uint8)]`. 200 samples is the TF guidance sweet-spot — enough to fit activation ranges, fast enough that calibration runs in ~30 s on A100.

### Embedded label metadata

After conversion, `tflite-support`'s `ImageClassifierWriter` attaches the class labels and preprocessing parameters into the `.tflite` itself:

```python
from tflite_support.metadata_writers import image_classifier
writer = image_classifier.MetadataWriter.create_for_inference(
    tflite_int8,
    input_norm_mean=[127.5], input_norm_std=[127.5],   # encodes the [-1, 1] preprocess
    label_file_paths=[labels_txt_path],                # one class name per line, cfg["classes"] order
)
Path(f"{EXPORTS_DIR}/l2_{TARGET}_int8.tflite").write_bytes(writer.populate())
```

The mobile app can read class names and preprocessing directly from the `.tflite` (via TFLite Task Library or `tflite_support.metadata`) instead of shipping a separate `labels.txt`. `tflite-support` is added to `requirements.txt`; Colab ships it pre-installed.

### Fallback: FP16

```python
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter.optimizations              = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
# I/O stays float32 here — FP16 is a debugging fallback, not the hot path
tflite_fp16 = converter.convert()
Path(f"{EXPORTS_DIR}/l2_{TARGET}_fp16.tflite").write_bytes(tflite_fp16)
```

No label metadata is embedded in the FP16 artifact — it is for accuracy-regression debugging only, not for deployment. If INT8 ever shows per-class regressions on-device, the mobile app can swap to the FP16 file as a temporary measure while we investigate.

### Inference smoke test

Final cell — validates the deployed artifact, not the Keras model:

```python
interp = tf.lite.Interpreter(model_path=f"{EXPORTS_DIR}/l2_{TARGET}_int8.tflite")
interp.allocate_tensors()
inp = interp.get_input_details()[0];  out = interp.get_output_details()[0]
assert inp["dtype"] == np.uint8                                  # contract check
assert out["dtype"] == np.uint8

sample = random.choice(list(Path(f"{DATA_DIR}/test").rglob("*.jpg")))
true_class = sample.parent.name
img = np.array(Image.open(sample).convert("RGB").resize(
    (cfg["imgsz"], cfg["imgsz"])), dtype=np.uint8)[None]

interp.set_tensor(inp["index"], img)
interp.invoke()
probs_q = interp.get_tensor(out["index"])[0]
scale, zp = out["quantization"]
probs = (probs_q.astype(np.float32) - zp) * scale                # dequantize for display
top3  = probs.argsort()[-3:][::-1]
# Print "true=<class>   pred=[c1:p1, c2:p2, c3:p3]"
# Save annotated preview to EXPORTS_DIR/l2_<target>_smoke_test.png
```

The smoke test is intentionally **not** a hard pass/fail. Its job is to catch "the `.tflite` is structurally broken" or "uint8 I/O contract is wrong," not to re-measure accuracy.

### Per-target export output

```
exports/
├── l2_<target>_int8.tflite       ← mobile deploy artifact, with embedded labels + preprocessing
├── l2_<target>_fp16.tflite       ← debugging fallback
└── l2_<target>_smoke_test.png    ← annotated inference preview
```

## Repo changes

### New files

- `notebooks/train_l2_efficientnet_lite.ipynb` — generated by `scripts/gen_train_notebook.py`. Never hand-edited.
- `scripts/gen_train_notebook.py` — single source of truth. Same builder pattern as L1: a list of `markdown(...)` / `code(...)` cells, dumped to `.ipynb` JSON.
- `scripts/notebook_helpers.py` — pure functions, unit-tested locally. Embedded verbatim into one notebook cell so the notebook is self-contained in Colab.
- `scripts/infer_keras.py` — local sanity test on a downloaded `best.keras` for a given target.
- `scripts/infer_tflite.py` — local sanity test on a downloaded `.tflite`. Reads embedded label metadata; asserts the uint8 I/O contract.
- `tests/test_notebook_helpers.py` — unit tests for the helper functions.
- `tests/test_notebook_generation.py` — generator parses, expected sections present, embedded helpers cell is byte-equal to the source file.
- `tests/__init__.py` — empty.
- `requirements.txt` — see below.

### Rewritten

- `README.md` — replaces the stub with a real intro (the L1→L2 stack picture, HF dataset layout, the TARGETS table, "set TARGET, run notebook" usage, regen + test instructions, local dev). Same shape as L1's README.

### `requirements.txt`

```
tensorflow>=2.15.0
tensorflow-hub>=0.16.0
tflite-support>=0.4.4
huggingface_hub>=0.21.0
pillow>=10.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
```

For local notebook-generator + tests in a venv. Colab ships TF + sklearn + matplotlib already.

## `scripts/notebook_helpers.py` API

Pure, unit-testable, no GPU required:

| Function                                                    | Purpose                                                                                          |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `extract_target_zip(hf_cache, data_root, target, force)`    | Unzip `hf_cache/l2_<target>/data.zip` into `data_root/l2_<target>/{train,val,test}/<class>/`. Skip if `train/<first_class>/` already non-empty unless `force=True`. Raises `FileNotFoundError` if the zip is missing. |
| `validate_imagefolder(data_dir, class_names, splits)`       | Assert class folders exactly match `class_names` per split; assert files have image extensions; print per-class counts + imbalance ratio. Raises `ValueError` on first violation. |
| `compute_class_weights(data_dir, class_names)`              | Returns `{class_index: weight}` from train counts using `total / (num_classes * count_i)`.       |
| `representative_dataset_gen(data_dir, imgsz, n=200, seed=0)`| Returns a generator yielding `[uint8 tensor of shape (1, imgsz, imgsz, 3)]` from train images. Used as `tf.lite.TFLiteConverter.representative_dataset`. |
| `write_labels_txt(out_path, class_names)`                   | One class name per line in fixed order. Consumed by the `tflite-support` metadata writer.        |

## Notebook cell outline (final)

`train_l2_efficientnet_lite.ipynb` is built by `scripts/gen_train_notebook.py` in this order:

1. **Markdown header** — purpose, the L1→L2 stack picture, the `TARGET` concept, dataset link.
2. **`pip install`** — `tensorflow-hub`, `tflite-support`, `huggingface_hub`, `pillow` (the rest are already in Colab).
3. **Imports + GPU check** — fail loudly if no GPU runtime selected.
4. **Drive mount + HF login** — same idiom as L1; only the `DRIVE_ROOT` and dataset repo differ.
5. **Config + `TARGET` selector** — paths, `TARGETS` lookup, the `TARGET = "noodle"` line that the user edits before running.
6. **Helpers** — embedded `scripts/notebook_helpers.py` source.
7. **Dataset fetch + unzip** — `snapshot_download(allow_patterns=[f"l2_{TARGET}/*"])` then `extract_target_zip`.
8. **Folder pre-flight** — `validate_imagefolder(...)`.
9. **`tf.data` datasets** — `image_dataset_from_directory` for train / val / test, with `class_names=cfg["classes"]` to lock label order.
10. **Build model** — TF Hub feature vector + Rescaling + augmentation block + dropout + dense.
11. **Compute class weights**.
12. **Stage 1: head-only training** — backbone frozen, compile, fit with callbacks.
13. **Stage 2: full fine-tune** — unfreeze, re-compile at `1e-4` with cosine decay, fit.
14. **Evaluate on test split** — top-1 / top-3 / classification report / confusion matrix / curves.
15. **TFLite INT8 export + metadata embedding**.
16. **TFLite FP16 fallback export**.
17. **Inference smoke test on the INT8 `.tflite`**.

## Tests

Run locally with `python -m unittest discover tests -v`. No GPU, no Hugging Face access.

**`test_notebook_helpers.py`:**
- `extract_target_zip` round-trip on a tiny synthetic zip in `tempfile.TemporaryDirectory()`.
- `validate_imagefolder` accepts a well-formed tree; rejects a missing class folder; rejects an unexpected class folder; rejects a non-image file.
- `compute_class_weights` returns weights summing to `num_classes` for balanced input; returns weight `> 1` for under-represented classes.
- `representative_dataset_gen` yields exactly `n` arrays, each with shape `(1, imgsz, imgsz, 3)` and dtype `uint8`.

**`test_notebook_generation.py`:**
- Every code cell in the generated notebook parses as valid Python (`ast.parse`).
- Every expected markdown section header is present.
- The embedded `notebook_helpers.py` cell is byte-equal to the source file's contents.
- The `TARGETS` dict in the config cell has the three current keys with the correct `num_classes` for each.

## Out of scope

- **Other L1 targets** — `l2_beverage`, `l2_grilled_fried`, `l2_banh_bread`, `l2_fruit`, `l2_dessert_snack`. The `TARGETS` table is structured so adding one is a one-row change once its `data.zip` lands on Hugging Face. Beverage has a reserved row but its row is not wired up because no data exists yet.
- **Inference orchestration between L1 and L2** — that is a mobile-app concern, not training.
- **Hyperparameter search / sweeps** — fixed recipe per variant.
- **On-device benchmarking** — separate concern in the mobile app project.
- **Publishing the `.keras` or `.tflite` to a Hugging Face model repo** — Drive-only for now, same as L1.
- **A `TARGET = "all"` mode** that trains all three classifiers in one notebook session — explicitly considered and declined in favor of clean per-target resume and parallelizability across Colab sessions.

## Success criteria

1. The regenerated notebook runs end-to-end on a fresh Colab A100 with `HF_TOKEN` set and `TARGET ∈ {noodle, rice, soup}`, producing `best.keras`, `l2_<target>_int8.tflite` (with embedded labels), and `l2_<target>_fp16.tflite` under `DRIVE_ROOT/`.
2. Re-running the notebook with a different `TARGET` writes to a separate Drive subfolder and does not collide with prior runs' outputs.
3. After an intentional runtime reset mid-training, re-running the notebook with the same `TARGET` resumes from the last completed epoch via `BackupAndRestore` rather than restarting from epoch 0.
4. Test-set top-1 + top-3 + per-class classification report + confusion matrix + training curves are produced and persisted for each `TARGET`. A soft warning is printed if top-1 < 0.70 (export still proceeds).
5. The INT8 `.tflite` accepts `uint8` input and produces `uint8` output, with embedded class labels readable via `tflite_support.metadata`. The smoke-test cell exercises this contract end-to-end on a random test-split image.
6. `python -m unittest discover tests -v` passes locally without GPU or Hugging Face access (all tests use synthetic in-memory data).

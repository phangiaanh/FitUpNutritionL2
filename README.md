# FitUpNutrition L2 - Fine-grained Vietnamese dish classifiers

Train one **EfficientNet Lite** classifier per L1 food category. Run the Colab
notebook once for each `TARGET` (`"noodle"`, `"rice"`, `"soup"`); each run
fine-tunes its own model, evaluates on the test split, and exports a quantized
`.tflite` to Google Drive so a Colab disconnect never destroys progress.

## The stack

```
L1 (YOLO11s)      detects coarse category and crops the box
   │
   ▼
L2 (this repo)    classifies the specific Vietnamese dish inside the crop
```

L1 lives in a separate project. The two are deployed as independent TFLite
files on the mobile app.

## Targets (current)

| TARGET   | variant   | imgsz | num_classes | classes                                                                                              |
| -------- | --------- | ----- | ----------- | ---------------------------------------------------------------------------------------------------- |
| `noodle` | EN-Lite2  | 260   | 12          | banh_canh, bun_bo_hue, bun_cha, bun_cha_ca, bun_mam, bun_rieu, cao_lau, hu_tieu, mi, mi_quang, nui_xao_bo, pho |
| `rice`   | EN-Lite1  | 240   | 6           | chao, com_chien, com_chien_ga, com_tam, com_trang, xoi                                              |
| `soup`   | EN-Lite0  | 224   | 4           | bo_kho, canh, lau, sup_cua                                                                          |

`beverage`, `grilled_fried`, `banh_bread`, `fruit`, `dessert_snack` are
deliberately deferred until their datasets land on Hugging Face.

## Dataset on Hugging Face

Repo `WatermelonAnh/FoodClassifierL2`
([HF dataset page](https://huggingface.co/datasets/WatermelonAnh/FoodClassifierL2)) —
one subfolder per L2 target:

```
WatermelonAnh/FoodClassifierL2/
├── l2_noodle/    { data.zip, dataset.yaml, build_stats.txt }
├── l2_rice/      { data.zip, dataset.yaml, build_stats.txt }
├── l2_soup/      { data.zip, dataset.yaml, build_stats.txt }
└── l2_beverage/  { dataset.yaml, build_stats.txt }   ← no data yet
```

Each `data.zip` unpacks to:

```
l2_<target>/
├── train/<class_name>/<image>.<ext>
├── val/<class_name>/<image>.<ext>
└── test/<class_name>/<image>.<ext>
```

## Colab notebook

Open or upload [`notebooks/train_l2_efficientnet_lite.ipynb`](notebooks/train_l2_efficientnet_lite.ipynb):

1. **Runtime → GPU** (A100 recommended).
2. Run the **drive mount + Hugging Face login** cells (`HF_TOKEN` with read access to the dataset repo).
3. Set `TARGET = "noodle"` (or `"rice"` or `"soup"`) in the config cell.
4. Run all cells.

Artifacts land under `/content/drive/MyDrive/FitUpNutritionL2/`:

```
runs/l2_<target>/
├── best.keras                ← val-acc-best checkpoint (export source)
├── backup/                   ← BackupAndRestore state (resume on disconnect)
├── labels.txt                ← class names in fixed order
├── training_log.csv          ← per-epoch metrics, both stages concatenated
├── training_curves.png
├── confusion_matrix.png
└── per_class_report.txt
exports/
├── l2_<target>_int8.tflite       ← mobile deploy artifact, uint8 I/O + embedded labels
├── l2_<target>_fp16.tflite       ← debugging fallback
└── l2_<target>_smoke_test.png    ← annotated inference preview
```

If a Colab session disconnects mid-run, just re-run the notebook with the same
`TARGET`: `BackupAndRestore` picks up from the last completed epoch.

## Regenerating the notebook

The notebook source-of-truth is
[`scripts/gen_train_notebook.py`](scripts/gen_train_notebook.py). Pure helpers
(`extract_target_zip`, `validate_imagefolder`, `compute_class_weights`,
`representative_dataset_gen`, `write_labels_txt`) live in
[`scripts/notebook_helpers.py`](scripts/notebook_helpers.py) and are embedded
verbatim into one notebook cell so the notebook stays self-contained in Colab.

Regenerate the `.ipynb` after editing either file:

```bash
python3 scripts/gen_train_notebook.py
```

## Tests

```bash
python3 -m unittest discover tests -v
```

Tests cover the helper functions (zip extract, ImageFolder validation, class
weights, representative-dataset generator, labels writer) and the notebook
generator (every code cell parses as Python, expected sections are present,
embedded helpers cell is byte-equal to the source file).

## Local dev (optional)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Training itself runs on Colab; locally you can run the generator + tests.
```

Local sanity inference is available via:

```bash
python3 scripts/infer_keras.py  --target rice --model best.keras   --image food.jpg
python3 scripts/infer_tflite.py --tflite l2_rice_int8.tflite       --image food.jpg
```

# MS-COCO Multi-Label Classification

PyTorch project for 80-class multi-label image classification on an MS-COCO-style dataset.

The code uses torchvision pretrained backbones, replaces the final classifier with `Linear -> BatchNorm1d`, trains with `BCEWithLogitsLoss` (with computed `pos_weight`), tunes threshold on validation, and exports test predictions to JSON.

## Repository Contents

- `training.py`: train/validation pipeline; writes run folder artifacts (`best_model.pt`, `run_config.json`, `confusion_matrix.png`).
- `testing.py`: loads checkpoint and exports test predictions JSON.
- `generate_confusion_matrix.py`: confusion matrix utilities (`generate`, `ensure-if-missing`, and batch generation for run folders).
- `model_performance_table.py`: aggregates run metadata (prefers `run_config.json`) and writes CSV.
- `config.py`: dataset paths, class list, base defaults.
- `models_factory.py`: model registry and classifier-head replacement.
- `dataset_readers.py`: train/test dataset loaders.
- `utils.py`: train/validation loops and progress bar.
- `tensorboard_logging.py`: TensorBoard helper.
- `sync_pretrained_model_cache.py`: downloads all registered model weights and prunes stale cached files.
- `get_common_classes.py`, `find_head_path.py`: helper scripts.

## Supported Backbones

From `models_factory.py` (`MODEL_SPECS`):

- `resnet18`
- `resnet50`
- `densenet121`
- `mobilenet_v2`
- `mobilenet_v3_large`
- `mobilenet_v3_small`
- `efficientnet_b0`
- `efficientnet_v2_s`
- `efficientnet_b4`
- `convnext_tiny`
- `convnext_small`
- `convnext_base`
- `convnext_large`
- `regnet_y_800mf`
- `swin_t`
- `swin_v2_t`
- `swin_v2_s`
- `swin_v2_b`

## Default Paths and Dataset Layout

Defaults are defined in `config.py`:

- Local data root: `~/ms-coco`
- Dataset root: `~/ms-coco/ms-coco-dataset`
- Pretrained cache root: `~/ms-coco/pre-trained_models`
- Trained models root: `<repo>/trained_models`

Expected dataset structure:

```text
~/ms-coco/
`-- ms-coco-dataset/
    |-- images/
    |   |-- train-resized/   # train images (.jpg)
    |   `-- test-resized/    # test images (.jpg)
    `-- labels/
        `-- train/           # one .cls file per train image

<repo>/
`-- trained_models/
    |-- best_model.pt                        # active/latest checkpoint for testing.py
    |-- <model_name>_<YYYYMMDD-HHMMSS>/
    |   |-- best_model.pt
    |   |-- run_config.json
    |   `-- confusion_matrix.png
    `-- ...
```

Each `*.cls` file must contain one class index per line (0-79), matching `config.py::CLASSES`.

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Configuration

### `config.py` (global defaults)

- `MODEL_NAME`: selected model name.
- `BEST_MODEL_PATH`: active checkpoint path (`trained_models/best_model.pt`).
- `FREEZE_BACKBONE`: starting freeze state used by `training.py`.
- `LOCAL_FOLDER`, `DATASET_FOLDER`, `PRETRAINED_MODELS_FOLDER`, `TRAINED_MODELS_FOLDER`.
- `NUM_CLASSES` and `CLASSES`.

### `training.py` (runtime knobs)

Key defaults include:

- Epochs/batching: `NUM_EPOCHS=25`, `TRAIN_BATCH_SIZE_FROZEN=256`, `TRAIN_BATCH_SIZE_UNFROZEN=16`, `VAL_BATCH_SIZE=256`.
- Gradient accumulation: `GRAD_ACCUM_STEPS_FROZEN=1`, `GRAD_ACCUM_STEPS_UNFROZEN=4`.
- AMP: `USE_AMP=True`, `AMP_DTYPE=torch.float16` (CUDA only).
- Freeze schedule: start from `FREEZE_BACKBONE`, unfreeze at `UNFREEZE_BACKBONE_EPOCH=1`.
- Partial unfreeze option: `UNFREEZE_LAST_N_BACKBONE_LAYERS=None` (full unfreeze).
- LR schedule: differential LR enabled by default (`BACKBONE_BASE_LR=1e-5`, `HEAD_BASE_LR=1e-4`), milestones `(9,)`, decay `1e-2`.
- Data split: `VAL_SPLIT=0.05`, `SEED=42`, `NUM_WORKERS=4`.
- Threshold tuning candidates: `0.05 ... 0.95` step `0.05`.
- Early stopping config present but disabled by default.
- TensorBoard disabled by default (`USE_TENSORBOARD=False`).

### `testing.py` (runtime knobs)

- `BATCH_SIZE=32`
- `NUM_WORKERS=0`
- `TH_MULTI_LABEL=0.5` (used only if checkpoint does not provide thresholds)
- `MODEL_PATH=BEST_MODEL_PATH`
- `OUTPUT_PATH=Path("predictions.json")` (base name; final file includes metadata tokens)

## Training

```bash
python training.py
```

What happens:

- Loads selected pretrained backbone and replaces classifier head.
- Splits train data into train/val subsets (`random_split`, seeded).
- Computes class-balanced `pos_weight` from the train subset.
- Trains with optional freeze/unfreeze schedule and gradient accumulation.
- Tunes threshold on validation using weighted multi-label F1.
- Tracks run-best checkpoint by validation F1 for this run.
- Creates a run folder: `trained_models/<model_name>_<timestamp>/`.
- Saves:
  - `best_model.pt`
  - `run_config.json` (full run configuration + results + artifact paths)
  - `confusion_matrix.png` (generated if missing)
- Updates `trained_models/best_model.pt` as active checkpoint for `testing.py`.

TensorBoard:

1. Set `USE_TENSORBOARD = True` in `training.py`.
2. Run `tensorboard --logdir runs`.

## Inference / Test Prediction

```bash
python testing.py
```

What happens:

- Loads checkpoint from `MODEL_PATH`.
- Rebuilds model using `checkpoint["model_name"]` (fallback: `config.MODEL_NAME`).
- Uses inference threshold in this order: `best_threshold` -> `th_multi_label` -> `testing.py::TH_MULTI_LABEL`.
- Writes predictions as `{image_id: [class_indices...]}` JSON.
- Output filename is expanded with metadata (model/F1/epoch/train settings/test settings), e.g. `predictions_<...>.json`.

## Confusion Matrix

```bash
python generate_confusion_matrix.py
```

`generate_confusion_matrix.py` now supports two workflows:

1. Batch mode (default): scan run folders in `trained_models/` and generate missing `confusion_matrix.png` per run.
2. Single-checkpoint mode: ensure/generate matrix for `MODEL_PATH` (fallback when no run folders are found).

Key script-level constants:

- `SPLIT`: `"val"`, `"train"`, or `"all"`
- `VAL_SPLIT`, `SEED`, `BATCH_SIZE`, `NUM_WORKERS`
- `TH_MULTI_LABEL` (`None` => checkpoint threshold fallback)
- `NORMALIZE`: `"none"`, `"rows"`, `"all"`
- `TOP_K_CLASSES` (`0` renders all classes)
- `GENERATE_FOR_ALL_RUNS`: batch mode switch
- `RUN_CHECKPOINT_FILENAME`, `RUN_CONFUSION_MATRIX_FILENAME`
- `OVERWRITE_EXISTING`: if `False`, existing matrices are kept

Developer API in `generate_confusion_matrix.py`:

- `generate_confusion_matrix_for_checkpoint(...)`: always generate.
- `ensure_confusion_matrix_for_checkpoint(...)`: generate only if missing.
- `generate_missing_confusion_matrices_for_runs(...)`: batch fill missing matrices.

## Checkpoint Report Table

```bash
python model_performance_table.py
```

Behavior:

- Recursively loads `run_config.json` from `trained_models` (preferred source).
- Falls back to checkpoint metadata if no run configs are found.
- Drops heavy `state_dict` payload and non-F1 metric columns.
- Optional grouping/sorting via top-of-file constants.
- Writes CSV to `<trained_models>/model_performance_table.csv`.

## Pretrained Cache Sync

```bash
python sync_pretrained_model_cache.py
```

Behavior:

- Iterates all models in `MODEL_SPECS` and downloads/loads their default weights.
- Uses `PRETRAINED_MODELS_FOLDER` as torch hub cache root.
- Removes stale `.pt/.pth` files not referenced by current registry URLs.

## Utility Scripts

- `python get_common_classes.py`: prints overlap between `config.CLASSES` and weight metadata categories.
- `python find_head_path.py`: quick helper for discovering the last linear head path in a torchvision model.

## Prediction JSON Format

```json
{
    "000000000139": [0, 56, 57, 60, 62],
    "000000000285": [21]
}
```

Keys are image filenames without `.jpg`; values are predicted class indices.

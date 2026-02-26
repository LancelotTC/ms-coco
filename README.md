# MS-COCO Multi-Label Classification

PyTorch project for 80-class multi-label image classification on an MS-COCO-style dataset.

The code uses torchvision pretrained backbones, replaces the final classifier with `Linear -> BatchNorm1d`, trains with `BCEWithLogitsLoss` (with computed `pos_weight`), tunes the decision threshold on validation, and exports test predictions to JSON.

## Repository Contents

- `training.py`: end-to-end train/validation pipeline, threshold tuning, checkpoint selection/saving.
- `testing.py`: checkpoint loading and test-set prediction JSON export.
- `config.py`: paths, class list, base defaults (including `MODEL_NAME`).
- `models_factory.py`: model registry and classifier-head replacement.
- `dataset_readers.py`: datasets for train labels (`*.cls`) and test images.
- `utils.py`: train/validation loops, threshold tuning utility, terminal progress bar.
- `tensorboard_logging.py`: TensorBoard helper used by `training.py` when enabled.
- `generate_confusion_matrix.py`: pairwise confusion-matrix generation from a checkpoint.
- `model_performance_table.py`: checkpoint metadata aggregation to CSV.
- `sync_pretrained_model_cache.py`: downloads all registered model weights and prunes stale cached files.
- `get_common_classes.py`: quick check of class-name overlap between COCO classes and model weight metadata.
- `find_head_path.py`: helper to inspect the last linear layer path for a torchvision model.
- `The MS COCO classification challenge.ipynb`: assignment/skeleton notebook.

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

- Project root: folder containing this repository
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
    `-- best_model.pt        # active checkpoint path used by testing/confusion scripts
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

- `MODEL_NAME`: selected model name (currently ends as `convnext_large` due last assignment in file).
- `BEST_MODEL_PATH`: active checkpoint path.
- `FREEZE_BACKBONE`: starting freeze state used by `training.py`.
- `LOCAL_FOLDER`, `DATASET_FOLDER`, `PRETRAINED_MODELS_FOLDER`, `TRAINED_MODELS_FOLDER`.
- `NUM_CLASSES` and `CLASSES`.

Note: `TRAIN_METRICS_EVERY_N_EPOCHS` and `VAL_EVERY_N_EPOCHS` also exist in `config.py`, but `training.py` uses its own local constants for those values.

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
- `OUTPUT_PATH=Path("predictions.json")` (base name; actual output filename includes metadata tokens)

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
- Tracks run-best checkpoint by validation F1.
- Saves per-configuration checkpoint in `trained_models/<model_name>/...pt`.
- Compares against existing config checkpoint and overwrites only if new F1 is better.
- Writes selected checkpoint to `BEST_MODEL_PATH` for downstream scripts.

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

Script-level config includes:

- `SPLIT`: `"val"`, `"train"`, or `"all"`
- `VAL_SPLIT`, `SEED`, `BATCH_SIZE`, `NUM_WORKERS`
- `TH_MULTI_LABEL` (`None` => checkpoint threshold fallback)
- `NORMALIZE`: `"none"`, `"rows"`, `"all"`
- `TOP_K_CLASSES` (`0` renders all classes)
- `OUTPUT_PATH` (default `trained_models/confusion_matrix.png`)

## Checkpoint Report Table

```bash
python model_performance_table.py
```

Behavior:

- Recursively loads checkpoint metadata from `trained_models`.
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

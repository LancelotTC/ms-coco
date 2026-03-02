# MS-COCO Multi-Label Classification

PyTorch project for 80-class multi-label image classification on an MS-COCO-style dataset.

The training pipeline uses torchvision pretrained backbones, replaces the classifier head with `Linear -> BatchNorm1d`, trains with `BCEWithLogitsLoss` using class-balanced `pos_weight`, tunes threshold on validation, and writes run artifacts under `trained_models/`.

## Repository Contents

- `training.py`: train/validation pipeline, checkpointing, TensorBoard logging, run metadata export, and post-training artifact generation.
- `testing.py`: prediction generation (`predictions.json`) for all run folders by default.
- `generate_confusion_matrix.py`: confusion matrix generation for all run folders by default.
- `model_performance_table.py`: aggregates run metadata into `trained_models/model_performance_table.csv`.
- `config.py`: dataset paths, class names, active model name, and global defaults.
- `models_factory.py`: backbone registry and classifier-head replacement.
- `dataset_readers.py`: train/test dataset loaders.
- `tensorboard_logging.py`: TensorBoard scalar layout and hparams logging helpers.
- `REPORT.md`: project report with evaluation and analysis.
- `report_images/`: plots used in the report.
- `trained_models/`: run artifacts and TensorBoard event files (details below).

## Included Runs and Artifacts

Some trained model runs are already included in this repository (the same ones discussed in `REPORT.md`).

Checkpoint `.pt` files are too large for normal repository hosting, so they are git-ignored (`*.pt` in `.gitignore`) and may be missing when you clone.

For the included runs, useful artifacts are available in `trained_models/`, including:

- `run_config.json` (configuration + results metadata)
- `confusion_matrix.png`
- `predictions.json`
- TensorBoard event files under `trained_models/tensorboard_runs/`
- aggregated CSV table `trained_models/model_performance_table.csv`

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
- `vit_b_16`

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Dataset Layout

Defaults are defined in `config.py`:

- `LOCAL_FOLDER = ~/ms-coco`
- `DATASET_FOLDER = ~/ms-coco/ms-coco-dataset`
- `PRETRAINED_MODELS_FOLDER = ~/ms-coco/pre-trained_models`
- `TRAINED_MODELS_FOLDER = <repo>/trained_models`

Expected structure:

```text
~/ms-coco/
`-- ms-coco-dataset/
    |-- images/
    |   |-- train-resized/   # train images (.jpg)
    |   `-- test-resized/    # test images (.jpg)
    `-- labels/
        `-- train/           # one .cls file per train image
```

Each `.cls` file must contain one class index per line (`0..79`) matching `config.py::CLASSES`.

## Current Runtime Defaults

### `config.py`

- `MODEL_NAME = "convnext_tiny"` (active assignment)
- `FREEZE_BACKBONE = True`
- `BEST_MODEL_PATH = trained_models/best_model.pt`

### `training.py`

- `NUM_EPOCHS = 14`
- `TRAIN_BATCH_SIZE_FROZEN = 32`
- `TRAIN_BATCH_SIZE_UNFROZEN = 16`
- `VAL_BATCH_SIZE = 32`
- `GRAD_ACCUM_STEPS_FROZEN = 1`
- `GRAD_ACCUM_STEPS_UNFROZEN = 1`
- `USE_AMP = True`
- `FREEZE_BACKBONE_AT_START = True`
- `UNFREEZE_BACKBONE_EPOCH = 1`
- `UNFREEZE_LAST_N_BACKBONE_LAYERS = None` (full unfreeze)
- differential LR enabled:
  - `BACKBONE_BASE_LR = 1e-5`
  - `HEAD_BASE_LR = 1e-4`
  - `LR_MILESTONES = (5, 10)`
  - `LR_DECAY_FACTOR = 1e-2`
- `VAL_SPLIT = 0.05`
- `SEED = 42`
- `NUM_WORKERS = 10`
- early stopping enabled:
  - `EARLY_STOPPING_ENABLED = True`
  - `EARLY_STOPPING_PATIENCE = 4`
  - `EARLY_STOPPING_MIN_DELTA = 0.0`
- TensorBoard enabled: `USE_TENSORBOARD = True`

### `testing.py`

- `GENERATE_FOR_ALL_RUNS = True` (default behavior)
- `BATCH_SIZE = 32`
- `NUM_WORKERS = 0`
- `TH_MULTI_LABEL = 0.5`
- `OVERWRITE_EXISTING = False`

### `generate_confusion_matrix.py`

- `GENERATE_FOR_ALL_RUNS = True` (default behavior)
- `SPLIT = "val"`
- `VAL_SPLIT = 0.05`
- `BATCH_SIZE = 64`
- `NUM_WORKERS = 0`
- `NORMALIZE = "rows"`
- `TOP_K_CLASSES = 40`
- `OVERWRITE_EXISTING = False`

## Training

```bash
python training.py
```

What training does:

- creates a run folder: `trained_models/<model_name>_<timestamp>/`
- saves `best_model.pt` in the run folder
- updates active checkpoint `trained_models/best_model.pt`
- writes `run_config.json` with configuration/results/artifact metadata
- ensures missing confusion matrices for run folders
- ensures missing predictions for run folders
- logs TensorBoard scalars and hparams to:
  - `trained_models/tensorboard_runs/<model_name>/<run_name>/`

Launch TensorBoard:

```bash
tensorboard --logdir trained_models/tensorboard_runs
```

## Prediction Generation

```bash
python testing.py
```

Default behavior:

- scans `trained_models/**/best_model.pt`
- writes missing `predictions.json` per run folder
- skips existing prediction files unless `OVERWRITE_EXISTING=True`

Fallback behavior:

- if no run folders are found, it runs single-checkpoint inference using `MODEL_PATH`.

## Confusion Matrix Generation

```bash
python generate_confusion_matrix.py
```

Default behavior:

- scans `trained_models/**/best_model.pt`
- writes missing `confusion_matrix.png` per run folder
- skips existing confusion matrices unless `OVERWRITE_EXISTING=True`

Fallback behavior:

- if no run folders are found, it generates one matrix for `MODEL_PATH`.

## Model Performance Table

```bash
python model_performance_table.py
```

Behavior:

- prefers `trained_models/**/run_config.json` as source
- falls back to checkpoints if run configs are unavailable
- writes `trained_models/model_performance_table.csv`
- includes model identity/path and key metrics/runtime columns

## Prediction JSON Format

```json
{
    "000000000139": [0, 56, 57, 60, 62],
    "000000000285": [21]
}
```

Keys are image filenames without `.jpg`, values are predicted class indices.

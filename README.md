# MS-COCO Multi-Label Classification

This project trains and evaluates an image multi-label classifier on an MS-COCO style dataset using transfer learning in PyTorch.

It supports multiple torchvision backbones, replaces the final classification head with an 80-class `Linear -> BatchNorm1d` logits head, trains on one-hot multi-label targets, and exports test predictions to JSON.

## What This Repository Contains

- `training.py`: train/validation pipeline, checkpoint saving, optional TensorBoard logging.
- `testing.py`: loads the best checkpoint and generates predictions for test images.
- `models_factory.py`: model registry and classifier-head replacement logic.
- `dataset_readers.py`: train/test dataset loaders.
- `utils.py`: training loop, validation metrics, and terminal progress bar.
- `tensorboard_logging.py`: TensorBoard scalar logging helper.
- `config.py`: dataset paths, class list, model defaults.
- `predictions.json`: sample output from inference.

## Supported Backbones

Defined in `models_factory.py`:

- `resnet18`
- `resnet50`
- `densenet121`
- `mobilenet_v2`
- `efficientnet_b0`
- `vgg16`

## Dataset Layout (Expected by Code)

Paths are derived from `config.py`:

- Local root: `~/ms-coco`
- Dataset root: `~/ms-coco/ms-coco-dataset`

Expected structure:

```text
~/ms-coco/
|-- ms-coco-dataset/
|   |-- images/
|   |   |-- train-resized/      # training images (.jpg)
|   |   `-- test-resized/       # test images (.jpg)
|   `-- labels/
|       `-- train/              # one .cls file per training image
`-- trained_models/
    `-- best_model.pt           # created after training
```

Label files (`*.cls`) should contain one class index per line (0-79), matching COCO class order from `config.py`.

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install torch torchvision pillow
# Optional (only if using TensorBoard)
pip install tensorboard
```

## Configuration

Edit `config.py` to set:

- `MODEL_NAME` (default: `resnet18`)
- directory constants if you do not use the default `~/ms-coco` layout
- `BEST_MODEL_PATH` (checkpoint output path)
- `FREEZE_BACKBONE` (used as the initial freeze state in `training.py`)
- `TRAIN_METRICS_EVERY_N_EPOCHS` (`0` disables extra train-set eval; higher values run it less often)
- `VAL_EVERY_N_EPOCHS` (run validation every N epochs; final epoch always validates)
- `EARLY_STOPPING_ENABLED`, `EARLY_STOPPING_PATIENCE`, `EARLY_STOPPING_MIN_DELTA` (stop when validation F1 plateaus)
- `PRETRAINED_MODELS_FOLDER` (torchvision pretrained-weights cache directory)

Edit `training.py` / `testing.py` for runtime hyperparameters:

- batch sizes
- number of epochs
- freeze milestone settings (`FREEZE_BACKBONE_AT_START`, `UNFREEZE_BACKBONE_EPOCH`)
- learning-rate settings (`USE_DIFFERENTIAL_LR`, `LEARNING_RATE` or `BACKBONE_BASE_LR`/`HEAD_BASE_LR`, `LR_MILESTONES`, `LR_DECAY_FACTOR`)
- threshold for multi-label prediction (`TH_MULTI_LABEL`)

## Training

```bash
python training.py
```

Behavior:

- Loads pretrained backbone weights from torchvision.
- Supports freeze-then-unfreeze with a single milestone:
- starts with backbone frozen when `FREEZE_BACKBONE_AT_START=True`
- unfreezes backbone at epoch `UNFREEZE_BACKBONE_EPOCH` (if within total epochs)
- Learning-rate schedule is independent from freeze/unfreeze and applied through one scheduler over the whole run.
- Supports either one LR for all params (`LEARNING_RATE`) or differential LR (`BACKBONE_BASE_LR`/`HEAD_BASE_LR`).
- Uses `BCEWithLogitsLoss` for optimization.
- Computes class-balanced `pos_weight` from the train split and passes it to `BCEWithLogitsLoss`.
- Tunes the multi-label threshold on validation (best weighted F1) and saves it in checkpoint metadata as `best_threshold`.
- Uses `MultiStepLR` milestones (`LR_MILESTONES`) with decay factor `LR_DECAY_FACTOR`.
- Splits training data into train/validation (`VAL_SPLIT`).
- Supports speed/quality tradeoff with periodic metrics:
- train-set evaluation cadence via `TRAIN_METRICS_EVERY_N_EPOCHS`
- validation cadence via `VAL_EVERY_N_EPOCHS` (with forced final-epoch validation)
- Optional early stopping by validation F1 plateau, controlled by `EARLY_STOPPING_*` constants.
- Tracks and selects checkpoints strictly by validation `F1` score.
- Writes a per-model, config-stable checkpoint path under `trained_models/<model_name>/` where the filename contains config tokens (not metrics).
- Overwrites that config checkpoint only when a new run achieves a better `F1` than the existing file for the same config.
- Also updates `BEST_MODEL_PATH` with the selected checkpoint for easy testing.
- Prints a readable config summary before training and an F1-focused summary after training.

Enable TensorBoard in `training.py`:

```python
USE_TENSORBOARD = True
```

Then run:

```bash
tensorboard --logdir runs
```

## Inference / Test Prediction

```bash
python testing.py
```

Behavior:

- Loads checkpoint from `MODEL_PATH`.
- Recreates the same model architecture using saved `model_name`.
- Runs inference on `TEST_IMAGES_DIR`.
- Prints a readable testing config summary before inference and a summary after inference.
- Applies `sigmoid` to model logits, then thresholds with checkpoint `best_threshold` (fallback to configured threshold if missing).
- Writes predictions to a detailed filename that includes model name and available checkpoint metadata (best F1, epoch, batch size, learning rate, threshold), plus testing batch size and threshold.

## Results Table (Pandas)

```bash
python model_performance_table.py
```

This prints a command-line table of saved checkpoints with model name, F1 score, and configuration fields read from checkpoint metadata.

You can programmatically control sorting and grouping by editing constants at the top of `model_performance_table.py`:

- `SORT_BY`
- `SORT_ASCENDING`
- `GROUP_BY_COLUMNS`
- `GROUP_MODE` (`best` or `mean`)
- `CHECKPOINTS_ROOT`, `CHECKPOINT_GLOB`

Output format:

```json
{
    "000000000139": [0, 56, 57, 60, 62],
    "000000000285": [21]
}
```

Each key is an image filename (without `.jpg`), and each value is a list of predicted class indices.

## Notes

- `NUM_CLASSES` is fixed to 80.
- Class names are available in `config.py` as `CLASSES`.
- `NUM_WORKERS` defaults to `0` in both training and testing for compatibility.

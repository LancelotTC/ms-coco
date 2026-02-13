# MS-COCO Multi-Label Classification

This project trains and evaluates an image multi-label classifier on an MS-COCO style dataset using transfer learning in PyTorch.

It supports multiple torchvision backbones, replaces the final classification head with an 80-class sigmoid head, trains on one-hot multi-label targets, and exports test predictions to JSON.

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

Edit `training.py` / `testing.py` for runtime hyperparameters:

- batch sizes
- number of epochs
- learning rate
- threshold for multi-label prediction (`TH_MULTI_LABEL`)

## Training

```bash
python training.py
```

Behavior:

- Loads pretrained backbone weights from torchvision.
- Freezes all backbone parameters.
- Trains only the replaced classifier head (`Linear -> Sigmoid`) with `BCELoss`.
- Splits training data into train/validation (`VAL_SPLIT`).
- Saves best model by validation F1 score to `BEST_MODEL_PATH`.

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
- Writes predictions to `predictions.json`.

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

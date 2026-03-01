import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import BEST_MODEL_PATH, MODEL_NAME, NUM_CLASSES, TEST_IMAGES_DIR
from dataset_readers import COCOTestImageDataset
from metadata_utils import (
    checkpoint_epoch_token,
    checkpoint_inference_threshold,
    checkpoint_model_name,
    checkpoint_total_epochs,
    epoch_token,
)
from models_factory import AVAILABLE_MODELS, create_model
from references import (
    CKPT_BATCH_SIZE,
    CKPT_BEST_EPOCH,
    CKPT_BEST_THRESHOLD,
    CKPT_BEST_VAL_F1,
    CKPT_LEARNING_RATE,
    CKPT_LEARNING_RATES,
    CKPT_STATE_DICT,
    CKPT_THRESHOLD,
)
from utils import ProgressBar, print_section, tokenize_float


# Test data loading and inference threshold.
BATCH_SIZE: int = 32
NUM_WORKERS: int = 0
TH_MULTI_LABEL: float = 0.5

MODEL_PATH: Path = BEST_MODEL_PATH

OUTPUT_PATH: Path = Path("predictions.json")


def build_predictions_path(
    base_path: Path,
    model_name: str,
    estimated_f1: float | None,
    best_epoch: int | None,
    total_epochs: int | None,
    train_batch_size: int | None,
    train_learning_rate: float | None,
    train_th_multi_label: float | None,
    train_best_threshold: float | None,
    test_batch_size: int,
    test_th_multi_label: float,
) -> Path:
    suffix = base_path.suffix or ".json"
    stem = base_path.stem
    f1_token = tokenize_float(estimated_f1) if estimated_f1 is not None else "na"
    epoch_value = epoch_token(best_epoch, total_epochs)
    train_bs_token = str(train_batch_size) if train_batch_size is not None else "na"
    train_lr_token = tokenize_float(train_learning_rate, precision=6) if train_learning_rate is not None else "na"
    train_th_token = tokenize_float(train_th_multi_label, precision=3) if train_th_multi_label is not None else "na"
    best_th_token = tokenize_float(train_best_threshold, precision=3) if train_best_threshold is not None else "na"
    file_name = (
        f"{stem}_{model_name}"
        f"_f1-{f1_token}"
        f"_ep-{epoch_value}"
        f"_bs-{train_bs_token}"
        f"_lr-{train_lr_token}"
        f"_th-{train_th_token}"
        f"_bth-{best_th_token}"
        f"_testbs-{test_batch_size}"
        f"_testth-{tokenize_float(test_th_multi_label, precision=3)}"
        f"{suffix}"
    )
    return base_path.with_name(file_name)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model_name = checkpoint_model_name(checkpoint, MODEL_NAME)
    estimated_f1 = checkpoint.get(CKPT_BEST_VAL_F1)
    best_epoch = checkpoint.get(CKPT_BEST_EPOCH)
    total_epochs = checkpoint_total_epochs(checkpoint)
    train_batch_size = checkpoint.get(CKPT_BATCH_SIZE)
    train_learning_rate = checkpoint.get(CKPT_LEARNING_RATE)
    train_learning_rates = checkpoint.get(CKPT_LEARNING_RATES)
    train_th_multi_label = checkpoint.get(CKPT_THRESHOLD)
    train_best_threshold = checkpoint.get(CKPT_BEST_THRESHOLD)
    inference_threshold = checkpoint_inference_threshold(checkpoint, TH_MULTI_LABEL)
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_name}' not supported. Available: {', '.join(AVAILABLE_MODELS)}")

    test_config = {
        "model_name": model_name,
        "device": device.type,
        "checkpoint_path": MODEL_PATH,
        "estimated_best_val_f1": f"{float(estimated_f1):.4f}" if estimated_f1 is not None else "n/a",
        "estimated_best_epoch": checkpoint_epoch_token(checkpoint),
        "total_epochs(from_ckpt)": total_epochs if total_epochs is not None else "n/a",
        "train_batch_size(from_ckpt)": train_batch_size if train_batch_size is not None else "n/a",
        "train_learning_rate(from_ckpt)": train_learning_rate if train_learning_rate is not None else "n/a",
        "train_learning_rates(from_ckpt)": train_learning_rates if train_learning_rates is not None else "n/a",
        "train_th_multi_label(from_ckpt)": train_th_multi_label if train_th_multi_label is not None else "n/a",
        "train_best_threshold(from_ckpt)": train_best_threshold if train_best_threshold is not None else "n/a",
        "test_batch_size": BATCH_SIZE,
        "test_th_multi_label": inference_threshold,
        "test_num_workers": NUM_WORKERS,
    }
    print_section("TESTING START CONFIG", test_config)

    net, transform, _ = create_model(model_name, NUM_CLASSES, pretrained=True)
    if transform is None:
        raise RuntimeError("No transform available. Use a pretrained model or provide a custom transform.")

    test_dataset = COCOTestImageDataset(TEST_IMAGES_DIR, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    net.load_state_dict(checkpoint[CKPT_STATE_DICT])
    net = net.to(device)
    net.eval()

    output = {}
    progress_bar = None
    total_batches = len(test_loader)
    if total_batches > 0:
        progress_bar = ProgressBar(total=total_batches, start_at=0, label="    Testing")
    with torch.no_grad():
        for images, names in test_loader:
            images = images.to(device)
            outputs = net(images)
            probabilities = torch.sigmoid(outputs)
            predictions = probabilities > inference_threshold
            for i, name in enumerate(names):
                indices = predictions[i].nonzero(as_tuple=False).squeeze(1).tolist()
                output[name] = indices
            if progress_bar:
                progress_bar.increment()
    if progress_bar:
        progress_bar.finish()

    output_path = build_predictions_path(
        OUTPUT_PATH,
        model_name=model_name,
        estimated_f1=float(estimated_f1) if estimated_f1 is not None else None,
        best_epoch=int(best_epoch) if best_epoch is not None else None,
        total_epochs=int(total_epochs) if total_epochs is not None else None,
        train_batch_size=int(train_batch_size) if train_batch_size is not None else None,
        train_learning_rate=float(train_learning_rate) if train_learning_rate is not None else None,
        train_th_multi_label=float(train_th_multi_label) if train_th_multi_label is not None else None,
        train_best_threshold=float(train_best_threshold) if train_best_threshold is not None else None,
        test_batch_size=BATCH_SIZE,
        test_th_multi_label=inference_threshold,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    summary = {
        "model_name": model_name,
        "estimated_best_val_f1": f"{float(estimated_f1):.4f}" if estimated_f1 is not None else "n/a",
        "estimated_best_epoch": checkpoint_epoch_token(checkpoint),
        "total_epochs": total_epochs if total_epochs is not None else "n/a",
        "num_test_images": len(test_dataset),
        "predictions_path": output_path,
    }
    print_section("TESTING SUMMARY", summary)


if __name__ == "__main__":
    if MODEL_NAME not in AVAILABLE_MODELS:
        raise ValueError(f"MODEL_NAME must be one of: {', '.join(AVAILABLE_MODELS)}")
    main()

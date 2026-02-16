import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import BEST_MODEL_PATH, MODEL_NAME, NUM_CLASSES, TEST_IMAGES_DIR
from dataset_readers import COCOTestImageDataset
from models_factory import AVAILABLE_MODELS, create_model
from utils import ProgressBar, print_section, tokenize_float


BATCH_SIZE = 32
NUM_WORKERS = 0
TH_MULTI_LABEL = 0.5

MODEL_PATH = BEST_MODEL_PATH
OUTPUT_PATH = Path("predictions.json")


def _build_predictions_path(
    base_path: Path,
    model_name: str,
    estimated_f1: float | None,
    best_epoch: int | None,
    total_epochs: int | None,
    train_batch_size: int | None,
    train_learning_rate: float | None,
    train_th_multi_label: float | None,
    test_batch_size: int,
    test_th_multi_label: float,
) -> Path:
    suffix = base_path.suffix or ".json"
    stem = base_path.stem
    f1_token = tokenize_float(estimated_f1) if estimated_f1 is not None else "na"
    if best_epoch is not None and total_epochs is not None:
        epoch_token = f"{best_epoch}of{total_epochs}"
    elif best_epoch is not None:
        epoch_token = str(best_epoch)
    else:
        epoch_token = "na"
    train_bs_token = str(train_batch_size) if train_batch_size is not None else "na"
    train_lr_token = tokenize_float(train_learning_rate, precision=6) if train_learning_rate is not None else "na"
    train_th_token = tokenize_float(train_th_multi_label, precision=3) if train_th_multi_label is not None else "na"
    file_name = (
        f"{stem}_{model_name}"
        f"_f1-{f1_token}"
        f"_ep-{epoch_token}"
        f"_bs-{train_bs_token}"
        f"_lr-{train_lr_token}"
        f"_th-{train_th_token}"
        f"_testbs-{test_batch_size}"
        f"_testth-{tokenize_float(test_th_multi_label, precision=3)}"
        f"{suffix}"
    )
    return base_path.with_name(file_name)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model_name = checkpoint.get("model_name", MODEL_NAME)
    estimated_f1 = checkpoint.get("best_val_f1")
    best_epoch = checkpoint.get("best_epoch")
    total_epochs = checkpoint.get("total_epochs", checkpoint.get("num_epochs"))
    train_batch_size = checkpoint.get("batch_size")
    train_learning_rate = checkpoint.get("learning_rate")
    train_th_multi_label = checkpoint.get("th_multi_label")
    inference_threshold = float(checkpoint.get("best_threshold", train_th_multi_label if train_th_multi_label is not None else TH_MULTI_LABEL))
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_name}' not supported. Available: {', '.join(AVAILABLE_MODELS)}")

    test_config = {
        "model_name": model_name,
        "device": device.type,
        "checkpoint_path": MODEL_PATH,
        "estimated_best_val_f1": f"{float(estimated_f1):.4f}" if estimated_f1 is not None else "n/a",
        "estimated_best_epoch": (
            f"{best_epoch}of{total_epochs}" if best_epoch is not None and total_epochs is not None else best_epoch
        )
        if best_epoch is not None
        else "n/a",
        "total_epochs(from_ckpt)": total_epochs if total_epochs is not None else "n/a",
        "train_batch_size(from_ckpt)": train_batch_size if train_batch_size is not None else "n/a",
        "train_learning_rate(from_ckpt)": train_learning_rate if train_learning_rate is not None else "n/a",
        "train_th_multi_label(from_ckpt)": train_th_multi_label if train_th_multi_label is not None else "n/a",
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

    net.load_state_dict(checkpoint["state_dict"])
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

    output_path = _build_predictions_path(
        OUTPUT_PATH,
        model_name=model_name,
        estimated_f1=float(estimated_f1) if estimated_f1 is not None else None,
        best_epoch=int(best_epoch) if best_epoch is not None else None,
        total_epochs=int(total_epochs) if total_epochs is not None else None,
        train_batch_size=int(train_batch_size) if train_batch_size is not None else None,
        train_learning_rate=float(train_learning_rate) if train_learning_rate is not None else None,
        train_th_multi_label=float(train_th_multi_label) if train_th_multi_label is not None else None,
        test_batch_size=BATCH_SIZE,
        test_th_multi_label=inference_threshold,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    summary = {
        "model_name": model_name,
        "estimated_best_val_f1": f"{float(estimated_f1):.4f}" if estimated_f1 is not None else "n/a",
        "estimated_best_epoch": (
            f"{best_epoch}of{total_epochs}" if best_epoch is not None and total_epochs is not None else best_epoch
        )
        if best_epoch is not None
        else "n/a",
        "total_epochs": total_epochs if total_epochs is not None else "n/a",
        "num_test_images": len(test_dataset),
        "predictions_path": output_path,
    }
    print_section("TESTING SUMMARY", summary)


if __name__ == "__main__":
    if MODEL_NAME not in AVAILABLE_MODELS:
        raise ValueError(f"MODEL_NAME must be one of: {', '.join(AVAILABLE_MODELS)}")
    main()

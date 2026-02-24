import matplotlib.pyplot as plt
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from config import BEST_MODEL_PATH, CLASSES, MODEL_NAME, NUM_CLASSES, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR
from dataset_readers import COCOTrainImageDataset
from models_factory import AVAILABLE_MODELS, MODEL_SPECS, create_model
from utils import ProgressBar, print_section


MODEL_PATH = BEST_MODEL_PATH

# Dataset split options: "val", "train", "all"
SPLIT = "val"
VAL_SPLIT = 0.05
SEED = 42

BATCH_SIZE = 64
NUM_WORKERS = 0

# None => checkpoint best_threshold/th_multi_label/0.5 fallback.
TH_MULTI_LABEL = None

# Plot options: "none", "rows", "all"
NORMALIZE = "rows"
TOP_K_CLASSES = 40  # 0 => render all classes
OUTPUT_PATH = Path("trained_models") / "confusion_matrix.png"


def _resolve_transform(model_name: str):
    spec = MODEL_SPECS.get(model_name)
    if spec is None:
        raise ValueError(f"Unknown model '{model_name}'. Available: {', '.join(AVAILABLE_MODELS)}")
    return spec.weights.transforms()


def _validate_config() -> None:
    if SPLIT not in {"val", "train", "all"}:
        raise ValueError("SPLIT must be one of: 'val', 'train', 'all'.")
    if VAL_SPLIT <= 0 or VAL_SPLIT >= 1:
        raise ValueError("VAL_SPLIT must be between 0 and 1.")
    if BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE must be >= 1.")
    if TOP_K_CLASSES < 0:
        raise ValueError("TOP_K_CLASSES must be >= 0.")
    if NORMALIZE not in {"none", "rows", "all"}:
        raise ValueError("NORMALIZE must be one of: 'none', 'rows', 'all'.")
    if TH_MULTI_LABEL is not None and (TH_MULTI_LABEL < 0 or TH_MULTI_LABEL > 1):
        raise ValueError("TH_MULTI_LABEL must be between 0 and 1 when set.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")


def _select_dataset(
    full_dataset: torch.utils.data.Dataset,
    *,
    split: str,
    val_split: float,
    seed: int,
) -> torch.utils.data.Dataset:
    if split == "all":
        return full_dataset

    val_size = max(1, int(len(full_dataset) * val_split))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return val_set if split == "val" else train_set


def _update_pairwise_confusion_matrix(
    matrix: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
) -> None:
    predictions = torch.where(scores > threshold, 1.0, 0.0)
    top1_predictions = torch.argmax(scores, dim=1)

    for sample_idx in range(labels.shape[0]):
        true_indices = labels[sample_idx].nonzero(as_tuple=False).squeeze(1)
        if true_indices.numel() == 0:
            continue

        pred_indices = predictions[sample_idx].nonzero(as_tuple=False).squeeze(1)
        if pred_indices.numel() == 0:
            pred_indices = torch.tensor([int(top1_predictions[sample_idx])], dtype=torch.long)

        pred_list = pred_indices.tolist()
        for true_idx in true_indices.tolist():
            for pred_idx in pred_list:
                matrix[true_idx, pred_idx] += 1


def _build_plot_data(
    confusion_matrix: torch.Tensor,
    class_names: tuple[str, ...],
    normalize: str,
    top_k_classes: int,
) -> tuple[torch.Tensor, list[str]]:
    support = confusion_matrix.sum(dim=1)
    selected_indices = torch.arange(confusion_matrix.size(0))
    if top_k_classes > 0 and top_k_classes < confusion_matrix.size(0):
        selected_indices = torch.topk(support, top_k_classes).indices
        selected_indices, _ = torch.sort(selected_indices)

    matrix = confusion_matrix.index_select(0, selected_indices).index_select(1, selected_indices).float()
    if normalize == "rows":
        row_sums = matrix.sum(dim=1, keepdim=True)
        matrix = torch.where(row_sums > 0, matrix / row_sums, torch.zeros_like(matrix))
    elif normalize == "all":
        total = matrix.sum()
        matrix = matrix / total if total > 0 else torch.zeros_like(matrix)

    labels = [class_names[int(index)] for index in selected_indices]
    return matrix, labels


def _plot_confusion_matrix(
    matrix: torch.Tensor,
    labels: list[str],
    *,
    title: str,
    normalize: str,
    output_path: Path,
) -> None:
    classes_count = len(labels)
    fig_size = max(8.0, min(30.0, classes_count * 0.35))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(matrix.numpy(), cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(classes_count))
    ax.set_yticks(range(classes_count))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    annotate = classes_count <= 25
    if annotate:
        for i in range(classes_count):
            for j in range(classes_count):
                value = matrix[i, j].item()
                text = f"{value:.2f}" if normalize != "none" else str(int(round(value)))
                ax.text(j, i, text, ha="center", va="center", fontsize=6, color="black")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    _validate_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model_name = checkpoint.get("model_name", MODEL_NAME)
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_name}' not supported. Available: {', '.join(AVAILABLE_MODELS)}")

    threshold = TH_MULTI_LABEL
    if threshold is None:
        threshold = float(checkpoint.get("best_threshold", checkpoint.get("th_multi_label", 0.5)))

    config_items = {
        "model_path": MODEL_PATH,
        "model_name": model_name,
        "split": SPLIT,
        "val_split": VAL_SPLIT,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "threshold": threshold,
        "normalize": NORMALIZE,
        "top_k_classes": TOP_K_CLASSES if TOP_K_CLASSES > 0 else "all",
        "device": device.type,
        "output_path": OUTPUT_PATH,
    }
    print_section("CONFUSION MATRIX CONFIG", config_items)

    transform = _resolve_transform(model_name)
    dataset = COCOTrainImageDataset(TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, transform=transform)
    selected_dataset = _select_dataset(
        dataset,
        split=SPLIT,
        val_split=VAL_SPLIT,
        seed=SEED,
    )
    if len(selected_dataset) == 0:
        raise RuntimeError("Selected dataset split is empty; cannot generate a confusion matrix.")
    dataloader = DataLoader(
        selected_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    net, _, _ = create_model(model_name, NUM_CLASSES, pretrained=False)
    net.load_state_dict(checkpoint["state_dict"])
    net = net.to(device)
    net.eval()

    confusion_matrix = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    progress_bar = ProgressBar(total=len(dataloader), start_at=0, label="    Evaluating")
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = net(images)
            scores = torch.sigmoid(logits)
            _update_pairwise_confusion_matrix(confusion_matrix, scores.cpu(), labels.cpu(), float(threshold))
            progress_bar.increment()
    progress_bar.finish()

    plot_matrix, plot_labels = _build_plot_data(
        confusion_matrix,
        CLASSES,
        normalize=NORMALIZE,
        top_k_classes=TOP_K_CLASSES,
    )
    plot_title = f"Pairwise confusion matrix ({model_name}, split={SPLIT}, th={threshold:.2f})"
    _plot_confusion_matrix(
        plot_matrix,
        plot_labels,
        title=plot_title,
        normalize=NORMALIZE,
        output_path=OUTPUT_PATH,
    )

    summary = {
        "num_samples": len(selected_dataset),
        "classes_rendered": len(plot_labels),
        "output_path": OUTPUT_PATH,
    }
    print_section("CONFUSION MATRIX SUMMARY", summary)


if __name__ == "__main__":
    main()

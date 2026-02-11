from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from config import MODEL_NAME, NUM_CLASSES, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR
from dataset_readers import COCOTrainImageDataset
from models_factory import AVAILABLE_MODELS, create_model, freeze_all
from utils import train_loop, validation_loop

try:
    from torch.utils.tensorboard import SummaryWriter

    from tensorboard_logging import update_graphs

    TENSORBOARD_AVAILABLE = True
except ModuleNotFoundError:
    TENSORBOARD_AVAILABLE = False


BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.1
SEED = 42
NUM_WORKERS = 0

TH_MULTI_LABEL = 0.5
MBATCH_LOSS_GROUP = -1

USE_TENSORBOARD = False
MODEL_PATH = Path("models/best_model.pt")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net, transform, head_params = create_model(MODEL_NAME, NUM_CLASSES, pretrained=True)
    if transform is None:
        raise RuntimeError("No transform available. Use a pretrained model or provide a custom transform.")

    full_dataset = COCOTrainImageDataset(TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, transform=transform)

    val_size = max(1, int(len(full_dataset) * VAL_SPLIT))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    freeze_all(net)
    for param in head_params:
        param.requires_grad = True
    net = net.to(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(head_params, lr=LEARNING_RATE)

    best_metric = -1.0
    summary_writer = None
    if USE_TENSORBOARD and TENSORBOARD_AVAILABLE:
        summary_writer = SummaryWriter()

    for epoch in range(NUM_EPOCHS):
        mbatch_losses = train_loop(
            train_loader,
            net,
            criterion,
            optimizer,
            device,
            mbatch_loss_group=MBATCH_LOSS_GROUP,
        )

        train_results = validation_loop(
            train_loader,
            net,
            criterion,
            NUM_CLASSES,
            device,
            multi_label=True,
            th_multi_label=TH_MULTI_LABEL,
            one_hot=True,
        )
        val_results = validation_loop(
            val_loader,
            net,
            criterion,
            NUM_CLASSES,
            device,
            multi_label=True,
            th_multi_label=TH_MULTI_LABEL,
            one_hot=True,
        )

        if summary_writer:
            update_graphs(
                summary_writer,
                epoch,
                train_results,
                val_results,
                mbatch_group=MBATCH_LOSS_GROUP,
                mbatch_count=len(train_loader),
                mbatch_losses=mbatch_losses or [],
            )

        current_metric = float(val_results["f1"])
        if current_metric > best_metric:
            best_metric = current_metric
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_name": MODEL_NAME, "state_dict": net.state_dict()}, MODEL_PATH)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} "
            f"train_f1={float(train_results['f1']):.4f} "
            f"val_f1={float(val_results['f1']):.4f} "
            f"val_loss={float(val_results['loss']):.4f}"
        )

    if summary_writer:
        summary_writer.close()


if __name__ == "__main__":
    if MODEL_NAME not in AVAILABLE_MODELS:
        raise ValueError(f"MODEL_NAME must be one of: {', '.join(AVAILABLE_MODELS)}")
    main()

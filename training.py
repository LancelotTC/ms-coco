from pathlib import Path
import time
from typing import Iterable

import torch
from torch.utils.data import DataLoader, random_split

from config import (
    BEST_MODEL_PATH,
    FREEZE_BACKBONE,
    MODEL_NAME,
    NUM_CLASSES,
    TRAIN_IMAGES_DIR,
    TRAIN_LABELS_DIR,
    TRAIN_METRICS_EVERY_N_EPOCHS,
    VAL_EVERY_N_EPOCHS,
)
from dataset_readers import COCOTrainImageDataset
from models_factory import AVAILABLE_MODELS, create_model, freeze_all
from utils import print_section, tokenize_float, train_loop, tune_threshold_on_validation, validation_loop

try:
    from torch.utils.tensorboard import SummaryWriter

    from tensorboard_logging import update_graphs

    TENSORBOARD_AVAILABLE = True
except ModuleNotFoundError:
    TENSORBOARD_AVAILABLE = False


BATCH_SIZE = 256
NUM_EPOCHS = 20
# Intentionally run train metrics only once, at the final epoch.
TRAIN_METRICS_EVERY_N_EPOCHS = NUM_EPOCHS
LEARNING_RATE = 1e-2
# LR_MILESTONES = (max(1, NUM_EPOCHS // 2), 2 * max(1, NUM_EPOCHS // 4))
LR_MILESTONES = (2, 3, 4, 5, 6, 10)
LR_DECAY_FACTOR = 0.1
VAL_SPLIT = 0.05
SEED = 42
NUM_WORKERS = 4

TH_MULTI_LABEL = 0.5
THRESHOLD_CANDIDATES = tuple(i / 100 for i in range(5, 96, 5))
MBATCH_LOSS_GROUP = -1
EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_PATIENCE = 4
EARLY_STOPPING_MIN_DELTA = 0.0

USE_TENSORBOARD = False
MODEL_PATH = BEST_MODEL_PATH


def should_run_eval(epoch: int, every_n_epochs: int, force_last: bool, total_epochs: int) -> bool:
    if every_n_epochs > 0 and (epoch + 1) % every_n_epochs == 0:
        return True
    return force_last and epoch == total_epochs - 1


def _iter_subset_label_paths(train_subset) -> Iterable[Path]:
    dataset = train_subset.dataset
    annotations_dir = Path(dataset.annotations_dir)
    for idx in train_subset.indices:
        yield annotations_dir / dataset.img_labels[idx]


def _compute_pos_weight(train_subset, num_classes: int) -> torch.Tensor:
    class_positives = torch.zeros(num_classes, dtype=torch.float64)
    num_samples = len(train_subset)
    for label_path in _iter_subset_label_paths(train_subset):
        with label_path.open("r", encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                if not stripped:
                    continue
                class_index = int(stripped)
                if 0 <= class_index < num_classes:
                    class_positives[class_index] += 1.0
    class_negatives = float(num_samples) - class_positives
    pos_weight = torch.where(class_positives > 0, class_negatives / class_positives, torch.ones_like(class_positives))
    return pos_weight.to(dtype=torch.float32)


def _build_config_model_path(
    base_path: Path,
    model_name: str,
    total_epochs: int,
    batch_size: int,
    learning_rate: float,
    th_multi_label: float,
    freeze_backbone: bool,
    val_split: float,
    lr_milestones: tuple[int, ...],
    lr_decay_factor: float,
    train_metrics_every_n_epochs: int,
    val_every_n_epochs: int,
    seed: int,
    num_workers: int,
    num_classes: int,
) -> Path:
    suffix = base_path.suffix or ".pt"
    stem = base_path.stem
    milestones_token = "-".join(str(milestone) for milestone in lr_milestones) if lr_milestones else "none"
    file_name = (
        f"{stem}_{model_name}"
        f"_ep-{total_epochs}"
        f"_bs-{batch_size}"
        f"_lr-{tokenize_float(learning_rate, precision=6)}"
        f"_th-{tokenize_float(th_multi_label, precision=3)}"
        f"_frz-{int(freeze_backbone)}"
        f"_vs-{tokenize_float(val_split, precision=3)}"
        f"_ms-{milestones_token}x{tokenize_float(lr_decay_factor, precision=3)}"
        f"_te-{train_metrics_every_n_epochs}"
        f"_ve-{val_every_n_epochs}"
        f"_sd-{seed}"
        f"_nw-{num_workers}"
        f"_nc-{num_classes}"
        f"{suffix}"
    )
    return base_path.with_name(file_name)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_output_dir = MODEL_PATH.parent / MODEL_NAME
    model_output_dir.mkdir(parents=True, exist_ok=True)
    config_model_path = _build_config_model_path(
        model_output_dir / MODEL_PATH.name,
        MODEL_NAME,
        NUM_EPOCHS,
        BATCH_SIZE,
        LEARNING_RATE,
        TH_MULTI_LABEL,
        FREEZE_BACKBONE,
        VAL_SPLIT,
        LR_MILESTONES,
        LR_DECAY_FACTOR,
        TRAIN_METRICS_EVERY_N_EPOCHS,
        VAL_EVERY_N_EPOCHS,
        SEED,
        NUM_WORKERS,
        NUM_CLASSES,
    )

    existing_checkpoint = None
    existing_best_f1 = -1.0
    if config_model_path.exists():
        try:
            existing_checkpoint = torch.load(config_model_path, map_location="cpu")
            existing_best_f1 = float(existing_checkpoint.get("best_val_f1", -1.0))
        except Exception:
            existing_checkpoint = None
            existing_best_f1 = -1.0

    run_config = {
        "model_name": MODEL_NAME,
        "device": device.type,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "lr_milestones": LR_MILESTONES,
        "lr_decay_factor": LR_DECAY_FACTOR,
        "th_multi_label": TH_MULTI_LABEL,
        "threshold_candidates": THRESHOLD_CANDIDATES,
        "freeze_backbone": FREEZE_BACKBONE,
        "val_split": VAL_SPLIT,
        "num_workers": NUM_WORKERS,
        "train_metrics_every_n_epochs": TRAIN_METRICS_EVERY_N_EPOCHS,
        "val_every_n_epochs": VAL_EVERY_N_EPOCHS,
        "early_stopping_enabled": EARLY_STOPPING_ENABLED,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
        "active_checkpoint_path": MODEL_PATH,
        "config_checkpoint_path": config_model_path,
        "existing_config_best_f1": f"{existing_best_f1:.4f}" if existing_best_f1 >= 0 else "none",
    }
    print_section("TRAINING START CONFIG", run_config)

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

    if FREEZE_BACKBONE:
        freeze_all(net)
        for param in head_params:
            param.requires_grad = True
        trainable_params = list(head_params)
        print("Training mode: frozen backbone (head-only fine-tuning)")
    else:
        for param in net.parameters():
            param.requires_grad = True
        trainable_params = [param for param in net.parameters() if param.requires_grad]
        print("Training mode: unfrozen backbone (full-model fine-tuning)")
    net = net.to(device)

    pos_weight = _compute_pos_weight(train_set, NUM_CLASSES).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=(*LR_MILESTONES,),
        gamma=LR_DECAY_FACTOR,
    )

    run_best_f1 = -1.0
    run_best_epoch = -1
    run_best_threshold = TH_MULTI_LABEL
    run_best_checkpoint = None
    last_train_results = None
    last_val_results = None
    no_improve_eval_count = 0
    completed_epochs = 0
    early_stopped = False
    early_stop_reason = "none"
    summary_writer = None
    if USE_TENSORBOARD and TENSORBOARD_AVAILABLE:
        summary_writer = SummaryWriter()

    for epoch in range(NUM_EPOCHS):
        epoch_index = epoch + 1
        epoch_start = time.perf_counter()
        print(f"\nEpoch {epoch_index}:")
        current_lr = optimizer.param_groups[0]["lr"]
        run_train_metrics = should_run_eval(
            epoch,
            TRAIN_METRICS_EVERY_N_EPOCHS,
            force_last=False,
            total_epochs=NUM_EPOCHS,
        )
        run_val_metrics = should_run_eval(
            epoch,
            VAL_EVERY_N_EPOCHS,
            force_last=True,
            total_epochs=NUM_EPOCHS,
        )

        mbatch_losses = train_loop(
            train_loader,
            net,
            criterion,
            optimizer,
            device,
            mbatch_loss_group=MBATCH_LOSS_GROUP,
            progress_label="    Training",
        )

        train_results = None
        val_results = None
        tuned_threshold = TH_MULTI_LABEL
        if run_val_metrics:
            tuned_threshold, val_results = tune_threshold_on_validation(
                val_loader,
                net,
                criterion,
                NUM_CLASSES,
                device,
                one_hot=True,
                threshold_candidates=THRESHOLD_CANDIDATES,
                progress_label="    Validating",
                apply_sigmoid=True,
            )
            last_val_results = val_results

        if run_train_metrics:
            train_eval_threshold = tuned_threshold if run_val_metrics else run_best_threshold
            train_results = validation_loop(
                train_loader,
                net,
                criterion,
                NUM_CLASSES,
                device,
                multi_label=True,
                th_multi_label=train_eval_threshold,
                one_hot=True,
                progress_label=None,
                apply_sigmoid=True,
            )
            last_train_results = train_results

        if summary_writer and train_results is not None and val_results is not None:
            update_graphs(
                summary_writer,
                epoch,
                train_results,
                val_results,
                mbatch_group=MBATCH_LOSS_GROUP,
                mbatch_count=len(train_loader),
                mbatch_losses=mbatch_losses or [],
            )

        if val_results is not None:
            current_metric = float(val_results["f1"])
            if current_metric > run_best_f1 + EARLY_STOPPING_MIN_DELTA:
                run_best_f1 = current_metric
                run_best_epoch = epoch + 1
                run_best_threshold = tuned_threshold
                no_improve_eval_count = 0
                run_best_checkpoint = {
                    "model_name": MODEL_NAME,
                    "state_dict": net.state_dict(),
                    "best_val_f1": run_best_f1,
                    "best_epoch": run_best_epoch,
                    "total_epochs": NUM_EPOCHS,
                    "best_threshold": run_best_threshold,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "th_multi_label": TH_MULTI_LABEL,
                    "freeze_backbone": FREEZE_BACKBONE,
                    "val_split": VAL_SPLIT,
                    "lr_milestones": LR_MILESTONES,
                    "lr_decay_factor": LR_DECAY_FACTOR,
                    "train_metrics_every_n_epochs": TRAIN_METRICS_EVERY_N_EPOCHS,
                    "val_every_n_epochs": VAL_EVERY_N_EPOCHS,
                    "seed": SEED,
                    "num_workers": NUM_WORKERS,
                    "num_classes": NUM_CLASSES,
                    "early_stopping_enabled": EARLY_STOPPING_ENABLED,
                    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                    "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
                }
                print(
                    f"New run-best F1 at epoch {run_best_epoch}: "
                    f"val_f1={run_best_f1:.4f}, th={run_best_threshold:.2f}"
                )
            else:
                no_improve_eval_count += 1

        train_f1_text = f"{float(train_results['f1']):.4f}" if train_results is not None else "skipped"
        val_f1_text = f"{float(val_results['f1']):.4f}" if val_results is not None else "skipped"
        epoch_seconds = int(time.perf_counter() - epoch_start)
        print(
            f"Done: Epoch {epoch_index}/{NUM_EPOCHS} "
            f"lr={current_lr:.6g} "
            f"train_f1={train_f1_text} "
            f"val_f1={val_f1_text} "
            f"time taken = {epoch_seconds} seconds"
        )
        scheduler.step()
        completed_epochs = epoch_index

        if EARLY_STOPPING_ENABLED and val_results is not None and no_improve_eval_count >= EARLY_STOPPING_PATIENCE:
            early_stopped = True
            early_stop_reason = (
                f"validation F1 did not improve by > {EARLY_STOPPING_MIN_DELTA} for "
                f"{EARLY_STOPPING_PATIENCE} validation checks"
            )
            print(f"Early stopping at epoch {epoch_index}/{NUM_EPOCHS}: " f"{early_stop_reason}")
            break

    if summary_writer:
        summary_writer.close()

    selected_checkpoint = existing_checkpoint
    selected_source = "existing_config_checkpoint"
    did_overwrite_config_checkpoint = False
    if run_best_checkpoint is not None and run_best_f1 > existing_best_f1:
        selected_checkpoint = run_best_checkpoint
        selected_source = "new_run_best_checkpoint"
        did_overwrite_config_checkpoint = True
        torch.save(run_best_checkpoint, config_model_path)

    if selected_checkpoint is not None:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(selected_checkpoint, MODEL_PATH)

        net.load_state_dict(selected_checkpoint["state_dict"])
        net.eval()
        selected_threshold = float(
            selected_checkpoint.get("best_threshold", selected_checkpoint.get("th_multi_label", TH_MULTI_LABEL))
        )
        selected_train_results = validation_loop(
            train_loader,
            net,
            criterion,
            NUM_CLASSES,
            device,
            multi_label=True,
            th_multi_label=selected_threshold,
            one_hot=True,
            progress_label="    Train Eval",
            apply_sigmoid=True,
        )
        selected_train_f1 = float(selected_train_results["f1"])
        selected_val_f1 = float(selected_checkpoint["best_val_f1"])

        summary_items = {
            "model_name": MODEL_NAME,
            "selected_from": selected_source,
            "config_checkpoint_overwritten": did_overwrite_config_checkpoint,
            "best_epoch": f"{selected_checkpoint['best_epoch']}of{selected_checkpoint['total_epochs']}",
            "best_val_f1": f"{selected_val_f1:.4f}",
            "best_threshold": f"{selected_threshold:.2f}",
            "selected_train_f1_eval": f"{selected_train_f1:.4f}",
            "last_train_f1": f"{float(last_train_results['f1']):.4f}" if last_train_results else "skipped",
            "last_val_f1": f"{float(last_val_results['f1']):.4f}" if last_val_results else "skipped",
            "run_best_val_f1": f"{run_best_f1:.4f}" if run_best_checkpoint is not None else "none",
            "existing_config_best_f1": f"{existing_best_f1:.4f}" if existing_best_f1 >= 0 else "none",
            "completed_epochs": completed_epochs,
            "early_stopped": early_stopped,
            "early_stop_reason": early_stop_reason,
            "active_checkpoint_path": MODEL_PATH,
            "config_checkpoint_path": config_model_path,
            "config_summary": (
                f"epochs={NUM_EPOCHS}, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}, "
                f"th_multi_label={TH_MULTI_LABEL}, freeze_backbone={FREEZE_BACKBONE}"
            ),
        }
        print_section("TRAINING SUMMARY", summary_items)
    else:
        print("No validation results were produced and no prior config checkpoint exists; no checkpoint saved.")


if __name__ == "__main__":
    if MODEL_NAME not in AVAILABLE_MODELS:
        raise ValueError(f"MODEL_NAME must be one of: {', '.join(AVAILABLE_MODELS)}")

    # from tools import timespan

    t = time.perf_counter()

    main()

    # print(f"Time taken: {timespan(int(time.perf_counter() - t))}")
    print(f"Time taken: {time.perf_counter() - t: .2f}")

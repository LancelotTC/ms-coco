from pathlib import Path

import torch

from config import TRAINED_MODELS_FOLDER

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit("pandas is required for this script. Install with: pip install pandas") from exc


# Programmatic controls
CHECKPOINTS_ROOT = TRAINED_MODELS_FOLDER
CHECKPOINT_GLOB = "**/*.pt"
SKIP_DEFAULT_BEST_MODEL = True  # Skip generic "best_model.pt" files to reduce duplicates.
SORT_BY = "best_val_f1"
SORT_ASCENDING = False
GROUP_BY_COLUMNS: list[str] = []  # Example: ["model_name"] or ["model_name", "freeze_backbone"]
GROUP_MODE = "best"  # Options: "best", "mean"
OUTPUT_CSV_NAME = "model_performance_table.csv"

# Keep the report F1-centric. Any metric column containing these keywords is dropped.
DROP_METRIC_KEYWORDS = ("precision", "recall", "accuracy", "loss")


def _flatten_dict(data: dict[str, object], prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in data.items():
        key_name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, prefix=f"{key_name}."))
        else:
            flat[key_name] = value
    return flat


def _load_checkpoint_row(path: Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu")
    checkpoint = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    row = {
        "checkpoint_filename": path.name,
        "checkpoint_path": str(path),
    }
    row.update(_flatten_dict(checkpoint))
    return row


def _collect_rows(root: Path, pattern: str, skip_default_best_model: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob(pattern)):
        if skip_default_best_model and path.name == "best_model.pt":
            continue
        try:
            rows.append(_load_checkpoint_row(path))
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
    return rows


def _group_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if not GROUP_BY_COLUMNS:
        return df
    missing = [col for col in GROUP_BY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Cannot group by missing columns: {missing}")

    if GROUP_MODE == "best":
        if SORT_BY in df.columns:
            ordered = df.sort_values(SORT_BY, ascending=SORT_ASCENDING)
            return ordered.groupby(GROUP_BY_COLUMNS, dropna=False, as_index=False).head(1).reset_index(drop=True)
        return df.groupby(GROUP_BY_COLUMNS, dropna=False, as_index=False).head(1).reset_index(drop=True)

    if GROUP_MODE == "mean":
        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        keep_numeric = [col for col in numeric if col not in GROUP_BY_COLUMNS]
        grouped = df.groupby(GROUP_BY_COLUMNS, dropna=False)[keep_numeric].mean().reset_index()
        grouped["group_count"] = df.groupby(GROUP_BY_COLUMNS, dropna=False).size().to_numpy()
        return grouped

    raise ValueError(f"Unsupported GROUP_MODE='{GROUP_MODE}'. Use 'best' or 'mean'.")


def _drop_unused_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop: list[str] = []
    for col in df.columns:
        lowered = col.lower()
        if "f1" in lowered:
            continue
        if any(keyword in lowered for keyword in DROP_METRIC_KEYWORDS):
            cols_to_drop.append(col)
    return df.drop(columns=cols_to_drop, errors="ignore")


def _normalize_epoch_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Backward compatibility: older checkpoints may only have num_epochs.
    if "total_epochs" not in df.columns and "num_epochs" in df.columns:
        df["total_epochs"] = df["num_epochs"]
    if "num_epochs" in df.columns:
        df = df.drop(columns=["num_epochs"])
    return df


def main() -> None:
    root = Path(CHECKPOINTS_ROOT)
    output_csv_path = root / OUTPUT_CSV_NAME
    if not root.exists():
        raise SystemExit(f"Checkpoint root does not exist: {root}")

    print(f"Checkpoint root: {root.resolve()}")
    print(f"Checkpoint glob: {CHECKPOINT_GLOB}")
    print(f"Skip default best_model.pt: {SKIP_DEFAULT_BEST_MODEL}")
    print()

    rows = _collect_rows(root, CHECKPOINT_GLOB, skip_default_best_model=SKIP_DEFAULT_BEST_MODEL)
    if not rows and SKIP_DEFAULT_BEST_MODEL:
        print("No rows after skipping default checkpoints; retrying with best_model.pt included.")
        rows = _collect_rows(root, CHECKPOINT_GLOB, skip_default_best_model=False)
    if not rows:
        raise SystemExit(f"No checkpoints found under {root} matching '{CHECKPOINT_GLOB}'.")

    df = pd.DataFrame(rows)
    df = _drop_unused_metric_columns(df)
    df = _normalize_epoch_columns(df)
    df = _group_dataframe(df)

    if SORT_BY in df.columns:
        df = df.sort_values(SORT_BY, ascending=SORT_ASCENDING)
    else:
        print(f"Sort column '{SORT_BY}' not found. Showing unsorted output.")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Wrote {len(df)} rows to: {output_csv_path.resolve()}")


if __name__ == "__main__":
    main()

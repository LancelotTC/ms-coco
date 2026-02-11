import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import MODEL_NAME, NUM_CLASSES, TEST_IMAGES_DIR
from dataset_readers import COCOTestImageDataset
from models_factory import AVAILABLE_MODELS, create_model


BATCH_SIZE = 32
NUM_WORKERS = 0
TH_MULTI_LABEL = 0.5

MODEL_PATH = Path("models/best_model.pt")
OUTPUT_PATH = Path("predictions.json")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model_name = checkpoint.get("model_name", MODEL_NAME)
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_name}' not supported. Available: {', '.join(AVAILABLE_MODELS)}")

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
    with torch.no_grad():
        for images, names in test_loader:
            images = images.to(device)
            outputs = net(images)
            predictions = outputs > TH_MULTI_LABEL
            for i, name in enumerate(names):
                indices = predictions[i].nonzero(as_tuple=False).squeeze(1).tolist()
                output[name] = indices

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    if MODEL_NAME not in AVAILABLE_MODELS:
        raise ValueError(f"MODEL_NAME must be one of: {', '.join(AVAILABLE_MODELS)}")
    main()

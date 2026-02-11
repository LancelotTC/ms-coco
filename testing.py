import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import models

from config import NUM_CLASSES, TEST_IMAGES_DIR
from dataset_readers import COCOTestImageDataset


BATCH_SIZE = 32
NUM_WORKERS = 0
TH_MULTI_LABEL = 0.5

MODEL_PATH = Path("models/best_model.pt")
OUTPUT_PATH = Path("predictions.json")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = models.ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    test_dataset = COCOTestImageDataset(TEST_IMAGES_DIR, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    net = models.resnet18(weights=weights)
    net.fc = torch.nn.Sequential(
        torch.nn.Linear(net.fc.in_features, NUM_CLASSES),
        torch.nn.Sigmoid(),
    )
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
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
    main()

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

import torch
from torchvision import models


@dataclass(frozen=True)
class ModelSpec:
    builder: Callable[..., torch.nn.Module]
    weights: Optional[object]
    head_path: str


MODEL_SPECS = {
    "resnet18": ModelSpec(models.resnet18, models.ResNet18_Weights.DEFAULT, "fc"),
    "resnet50": ModelSpec(models.resnet50, models.ResNet50_Weights.DEFAULT, "fc"),
    "densenet121": ModelSpec(models.densenet121, models.DenseNet121_Weights.DEFAULT, "classifier"),
    "mobilenet_v2": ModelSpec(models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT, "classifier.1"),
    "efficientnet_b0": ModelSpec(models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, "classifier.1"),
    "vgg16": ModelSpec(models.vgg16, models.VGG16_Weights.DEFAULT, "classifier.6"),
}

AVAILABLE_MODELS = tuple(MODEL_SPECS.keys())


def _get_module_by_path(root: torch.nn.Module, path: str) -> torch.nn.Module:
    current = root
    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def _set_module_by_path(root: torch.nn.Module, path: str, module: torch.nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = module
    else:
        setattr(parent, last, module)


def freeze_all(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def _replace_head(
    model: torch.nn.Module,
    head_path: str,
    num_classes: int,
) -> Iterable[torch.nn.Parameter]:
    head = _get_module_by_path(model, head_path)
    if not hasattr(head, "in_features"):
        raise ValueError(f"Head at '{head_path}' does not expose in_features.")
    new_head = torch.nn.Sequential(
        torch.nn.Linear(head.in_features, num_classes),
        torch.nn.Sigmoid(),
    )
    _set_module_by_path(model, head_path, new_head)
    return new_head.parameters()


def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
) -> Tuple[torch.nn.Module, Optional[object], Iterable[torch.nn.Parameter]]:
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {', '.join(AVAILABLE_MODELS)}")
    spec = MODEL_SPECS[model_name]
    weights = spec.weights if pretrained else None
    model = spec.builder(weights=weights)
    transform = weights.transforms() if weights else None
    head_params = list(_replace_head(model, spec.head_path, num_classes))
    return model, transform, head_params

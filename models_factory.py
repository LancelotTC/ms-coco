from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Iterable

import torch
from torchvision import models

from config import PRETRAINED_MODELS_FOLDER


@dataclass(frozen=True)
class ModelSpec:
    builder: Callable[..., torch.nn.Module]
    weights: models.WeightsEnum
    head_path: str


MODEL_SPECS = {
    "resnet18": ModelSpec(models.resnet18, models.ResNet18_Weights.DEFAULT, "fc"),
    "resnet50": ModelSpec(models.resnet50, models.ResNet50_Weights.DEFAULT, "fc"),
    "densenet121": ModelSpec(models.densenet121, models.DenseNet121_Weights.DEFAULT, "classifier"),
    "mobilenet_v2": ModelSpec(models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT, "classifier.1"),
    "mobilenet_v3_large": ModelSpec(
        models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT, "classifier.3"
    ),
    "mobilenet_v3_small": ModelSpec(
        models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT, "classifier.3"
    ),
    "efficientnet_b0": ModelSpec(models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, "classifier.1"),
    "efficientnet_v2_s": ModelSpec(models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.DEFAULT, "classifier.1"),
    "convnext_tiny": ModelSpec(models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT, "classifier.2"),
    "convnext_small": ModelSpec(models.convnext_small, models.ConvNeXt_Small_Weights.DEFAULT, "classifier.2"),
    "regnet_y_800mf": ModelSpec(models.regnet_y_800mf, models.RegNet_Y_800MF_Weights.DEFAULT, "fc"),
    "swin_t": ModelSpec(models.swin_t, models.Swin_T_Weights.DEFAULT, "head"),
    "swin_v2_t": ModelSpec(models.swin_v2_t, models.Swin_V2_T_Weights.DEFAULT, "head"),
    "vgg16": ModelSpec(models.vgg16, models.VGG16_Weights.DEFAULT, "classifier.6"),
    # "alexnet": ModelSpec(models.alexnet, models.AlexNet_Weights.DEFAULT, "classifier.6"),
}

AVAILABLE_MODELS = tuple(MODEL_SPECS.keys())


def _configure_pretrained_weights_cache(cache_dir: Path = PRETRAINED_MODELS_FOLDER) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(cache_dir))


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
        torch.nn.BatchNorm1d(num_classes),
    )
    _set_module_by_path(model, head_path, new_head)
    return new_head.parameters()


def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
) -> tuple[torch.nn.Module, object | None, Iterable[torch.nn.Parameter]]:
    if pretrained:
        _configure_pretrained_weights_cache()
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {', '.join(AVAILABLE_MODELS)}")
    spec = MODEL_SPECS[model_name]
    weights = spec.weights if pretrained else None
    model = spec.builder(weights=weights)
    transform = weights.transforms() if weights else None
    head_params = list(_replace_head(model, spec.head_path, num_classes))
    return model, transform, head_params

"""
Build models by name. All use ImageNet pretrained weights.
"""

import torch.nn as nn
from torchvision import models


def build_model(name: str, num_classes: int) -> nn.Module:
    """Build a classification model. name: resnet50, efficientnet_b2, vit_b_16, convnext_tiny."""
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if name == "efficientnet_b2":
        m = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m

    if name == "vit_b_16":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
        return m

    if name == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m

    raise ValueError(f"Unknown model: {name}")


def get_param_groups(model: nn.Module, name: str, lr_backbone: float = 1e-4, lr_head: float = 1e-3) -> list[dict]:
    """Split model into backbone (pretrained) and head (new classifier) parameter groups."""
    if name == "resnet50":
        head_params = list(model.fc.parameters())
    elif name == "efficientnet_b2":
        head_params = list(model.classifier.parameters())
    elif name == "vit_b_16":
        head_params = list(model.heads.parameters())
    elif name == "convnext_tiny":
        head_params = list(model.classifier.parameters())
    else:
        raise ValueError(f"Unknown model: {name}")

    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

    return [
        {"params": backbone_params, "lr": lr_backbone, "name": "backbone"},
        {"params": head_params, "lr": lr_head, "name": "head"},
    ]


def get_model_names() -> list[str]:
    return ["resnet50", "efficientnet_b2", "vit_b_16", "convnext_tiny", "svm_resnet_features"]

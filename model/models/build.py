"""
Build models by name. All use ImageNet pretrained weights.
torchvision models + timm (for EVA-02).
"""

import torch.nn as nn
from torchvision import models
import timm


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build a classification model, optionally with pretrained weights."""
    if name == "resnet50":
        w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if name == "efficientnet_b2":
        w = models.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b2(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m

    if name == "vit_b_16":
        w = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.vit_b_16(weights=w)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
        return m

    if name == "convnext_tiny":
        w = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.convnext_tiny(weights=w)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m

    if name == "eva02_small":
        # EVA-02 Small (IJCV 2024) — pick best available pretrained tag
        available = timm.list_pretrained("eva02_small*")
        if available and pretrained:
            tag = available[0]  # use first available pretrained config
        else:
            tag = "eva02_small_patch14_224"
        m = timm.create_model(tag, pretrained=pretrained, num_classes=num_classes)
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
    elif name == "eva02_small":
        head_params = list(model.head.parameters())
    else:
        raise ValueError(f"Unknown model: {name}")

    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

    return [
        {"params": backbone_params, "lr": lr_backbone, "name": "backbone"},
        {"params": head_params, "lr": lr_head, "name": "head"},
    ]


def get_model_names() -> list[str]:
    return ["resnet50", "efficientnet_b2", "vit_b_16", "convnext_tiny", "eva02_small", "svm_resnet_features"]

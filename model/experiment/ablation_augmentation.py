"""
Ablation: Data augmentation — with vs without augmentation using ConvNeXt-Tiny.
"""

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_MODEL_DIR))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import get_dataloaders, Caltech101Dataset
from models import build_model, get_param_groups
from train import train_epoch, evaluate


EPOCHS = 20
PATIENCE = 7
WARMUP = 3
LABEL_SMOOTHING = 0.1
LR_BACKBONE = 1e-4
LR_HEAD = 1e-3
RESIZE = 256
CROP = 224


def _no_aug_loaders(batch_size: int = 32, num_workers: int = 4):
    """Build dataloaders without any data augmentation (train uses same transforms as val)."""
    minimal_tf = transforms.Compose([
        transforms.Resize((RESIZE, RESIZE)),
        transforms.CenterCrop(CROP),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = Caltech101Dataset("train", transform=minimal_tf)
    val_ds = Caltech101Dataset("val", transform=minimal_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def _train_variant(name: str, train_loader, val_loader, device: torch.device) -> dict:
    print(f"\n{'='*50}")
    print(f"  Augmentation: {name}")
    print(f"{'='*50}")

    num_classes = len(train_loader.dataset.classes)
    model = build_model("convnext_tiny", num_classes).to(device)
    param_groups = get_param_groups(model, "convnext_tiny", lr_backbone=LR_BACKBONE, lr_head=LR_HEAD)
    optimizer = AdamW(param_groups, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    warmup = min(WARMUP, EPOCHS)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, total_iters=warmup),
            CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup),
        ],
        milestones=[warmup],
    )

    best_acc = 0.0
    patience_counter = 0
    history = {"train_acc": [], "val_acc": []}

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_acc, _ = evaluate(model, train_loader, device)
        val_acc, val_loss = evaluate(model, val_loader, device)
        scheduler.step()

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  Epoch {epoch+1}/{EPOCHS} | train_loss={train_loss:.4f} | "
              f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    gap = history["train_acc"][-1] - history["val_acc"][-1]
    print(f"  Best val acc ({name}): {best_acc:.4f} | "
          f"Final train-val gap: {gap:.4f}")
    return {"best_val_acc": best_acc, "history": history}


def run(device: torch.device) -> dict[str, dict]:
    results = {}

    # With augmentation (default transforms)
    train_loader, val_loader, _ = get_dataloaders(batch_size=32, num_workers=4)
    results["With"] = _train_variant("With", train_loader, val_loader, device)

    # Without augmentation
    train_loader_no, val_loader_no = _no_aug_loaders(batch_size=32, num_workers=4)
    results["Without"] = _train_variant("Without", train_loader_no, val_loader_no, device)

    return results


def print_table(results: dict[str, dict]):
    print(f"\n{'='*50}")
    print("  Ablation: Augmentation (ConvNeXt-Tiny, 20 epochs)")
    print(f"{'='*50}")
    print(f"  {'Augmentation':<15} {'Val Acc':>10} {'Train-Val Gap':>15}")
    print(f"  {'-'*42}")
    for name, info in results.items():
        h = info["history"]
        gap = h["train_acc"][-1] - h["val_acc"][-1]
        print(f"  {name:<15} {info['best_val_acc']:>10.4f} {gap:>14.4f}")
    print()


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    results = run(device)
    print_table(results)

"""
Ablation: Input image size — 112×112 vs 224×224 using EVA-02-Small.
Sizes are multiples of patch_size=14.
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

from dataset import get_dataloaders
from models import build_model, get_param_groups
from train import train_epoch, evaluate


MODEL_NAME = "eva02_small"
SIZES = [112, 224]
EPOCHS = 10
PATIENCE = 5
WARMUP = 2
LABEL_SMOOTHING = 0.1
LR_BACKBONE = 2e-5
LR_HEAD = 2e-4


def run(device: torch.device) -> dict[int, float]:
    results = {}
    for size in SIZES:
        print(f"\n{'='*50}")
        print(f"  Image size: {size}×{size}")
        print(f"{'='*50}")

        train_loader, val_loader, _ = get_dataloaders(
            batch_size=32, num_workers=4, resize=size, crop=size,
        )
        num_classes = len(train_loader.dataset.classes)

        model = build_model(MODEL_NAME, num_classes).to(device)
        param_groups = get_param_groups(model, MODEL_NAME, lr_backbone=LR_BACKBONE, lr_head=LR_HEAD)
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

        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_acc, val_loss = evaluate(model, val_loader, device)
            scheduler.step()
            print(f"  Epoch {epoch+1}/{EPOCHS} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        results[size] = best_acc
        print(f"  Best val acc for {size}×{size}: {best_acc:.4f}")

    return results


def print_table(results: dict[int, float]):
    print(f"\n{'='*50}")
    print("  Ablation: Image Size (EVA-02-Small, 10 epochs)")
    print(f"{'='*50}")
    print(f"  {'Size':<12} {'Val Acc':>10}")
    print(f"  {'-'*22}")
    for size, acc in results.items():
        print(f"  {size}×{size:<8} {acc:>10.4f}")
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

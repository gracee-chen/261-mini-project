"""
Ablation: Optimizer — SGD vs AdamW using EVA-02-Small.
"""

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_MODEL_DIR))

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from dataset import get_dataloaders
from models import build_model, get_param_groups
from train import train_epoch, evaluate


MODEL_NAME = "eva02_small"
EPOCHS = 10
PATIENCE = 5
WARMUP = 2
LABEL_SMOOTHING = 0.1

# Per-optimizer tuned learning rates (fair comparison)
OPTIMIZER_CONFIG = {
    "AdamW": {"lr_backbone": 2e-5, "lr_head": 2e-4},
    "SGD":   {"lr_backbone": 1e-3, "lr_head": 1e-2},
}


def _train_with_optimizer(name: str, train_loader, val_loader, device: torch.device) -> float:
    cfg = OPTIMIZER_CONFIG[name]
    print(f"\n{'='*50}")
    print(f"  Optimizer: {name} (backbone_lr={cfg['lr_backbone']}, head_lr={cfg['lr_head']})")
    print(f"{'='*50}")

    num_classes = len(train_loader.dataset.classes)
    model = build_model(MODEL_NAME, num_classes).to(device)
    param_groups = get_param_groups(model, MODEL_NAME,
                                    lr_backbone=cfg["lr_backbone"], lr_head=cfg["lr_head"])

    if name == "AdamW":
        optimizer = AdamW(param_groups, weight_decay=1e-4)
    elif name == "SGD":
        optimizer = SGD(param_groups, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

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

    print(f"  Best val acc ({name}): {best_acc:.4f}")
    return best_acc


def run(device: torch.device) -> dict[str, float]:
    train_loader, val_loader, _ = get_dataloaders(batch_size=32, num_workers=4)
    results = {}

    for opt_name in ["AdamW", "SGD"]:
        results[opt_name] = _train_with_optimizer(opt_name, train_loader, val_loader, device)

    return results


def print_table(results: dict[str, float]):
    print(f"\n{'='*50}")
    print("  Ablation: Optimizer (EVA-02-Small, 10 epochs)")
    print(f"{'='*50}")
    print(f"  {'Optimizer':<12} {'Val Acc':>10}")
    print(f"  {'-'*22}")
    for name, acc in results.items():
        print(f"  {name:<12} {acc:>10.4f}")
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

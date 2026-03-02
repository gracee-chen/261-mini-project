"""
Train all models on Caltech-101 (deep learning + SVM).
Usage: python train.py [--model resnet50|efficientnet_b2|vit_b_16|convnext_tiny|svm_resnet_features]
"""

import argparse
from pathlib import Path

import joblib
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from dataset import get_dataloaders, get_data_root
from models import build_model, get_model_names, get_param_groups
from models.svm_classifier import FeatureExtractor, extract_features, train_svm, evaluate_svm


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def train_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()
    acc = correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    return acc, avg_loss


def train_svm_model(args):
    """Train SVM with ResNet-18 features (replaces old train_svm.py)."""
    device = torch.device(args.device)
    data_root = args.data_root or get_data_root()

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=4,
        root=data_root,
    )

    extractor = FeatureExtractor().to(device).eval()
    for p in extractor.parameters():
        p.requires_grad = False

    print("Extracting train features...")
    X_train, y_train = extract_features(train_loader, device, extractor=extractor)
    print("Extracting val features...")
    X_val, y_val = extract_features(val_loader, device, extractor=extractor)
    print("Extracting test features...")
    X_test, y_test = extract_features(test_loader, device, extractor=extractor)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    print("Training SVM...")
    svm, scaler = train_svm(X_train, y_train, X_val, y_val, C=1.0, kernel="rbf")

    val_acc = evaluate_svm(svm, scaler, X_val, y_val)
    test_acc = evaluate_svm(svm, scaler, X_test, y_test)

    print(f"Val accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    ckpt_dir = args.checkpoint_dir / "svm_resnet_features"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"svm": svm, "scaler": scaler}, ckpt_dir / "svm.joblib")
    print(f"Saved to {ckpt_dir / 'svm.joblib'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=get_model_names(), default="convnext_tiny")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3, help="(legacy) single LR, ignored when lr_backbone/lr_head set")
    parser.add_argument("--lr_backbone", type=float, default=1e-4, help="Learning rate for pretrained backbone")
    parser.add_argument("--lr_head", type=float, default=1e-3, help="Learning rate for classifier head")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true", help="Enable mixed-precision training")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--checkpoint_dir", type=Path, default=_PROJECT_ROOT / "checkpoints")
    parser.add_argument("--data_root", type=Path, default=None)
    args = parser.parse_args()

    if args.model == "svm_resnet_features":
        train_svm_model(args)
        return

    device = torch.device(args.device)
    data_root = args.data_root or get_data_root()

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=4,
        root=data_root,
    )
    num_classes = len(train_loader.dataset.classes)

    model = build_model(args.model, num_classes).to(device)

    # Differential learning rates
    param_groups = get_param_groups(model, args.model, lr_backbone=args.lr_backbone, lr_head=args.lr_head)
    optimizer = AdamW(param_groups, weight_decay=1e-4)

    # Label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Warmup + Cosine Annealing scheduler
    warmup_epochs = min(args.warmup_epochs, args.epochs)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    # Mixed precision scaler
    scaler = torch.amp.GradScaler(device=args.device) if args.use_amp else None

    best_acc = 0.0
    patience_counter = 0
    ckpt_dir = args.checkpoint_dir / args.model
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training {args.model} | backbone_lr={args.lr_backbone} head_lr={args.lr_head} "
          f"| label_smoothing={args.label_smoothing} | warmup={warmup_epochs} epochs "
          f"| grad_clip={args.max_grad_norm} | amp={args.use_amp}")

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            max_grad_norm=args.max_grad_norm, scaler=scaler, use_amp=args.use_amp,
        )
        val_acc, val_loss = evaluate(model, val_loader, device)
        scheduler.step()

        # Log per-group learning rates
        lr_info = " | ".join(f"lr_{g.get('name', i)}={g['lr']:.2e}" for i, g in enumerate(optimizer.param_groups))
        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f} | val_loss={val_loss:.4f} | {lr_info}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "acc": val_acc, "num_classes": num_classes},
                ckpt_dir / "best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
                break

    print(f"\nBest val accuracy: {best_acc:.4f}")
    print(f"Checkpoint saved to {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

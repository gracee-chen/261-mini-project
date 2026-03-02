"""
Evaluate trained models on test set.
Metrics: accuracy, per-class accuracy, confusion matrix, precision/recall/F1 (macro & weighted), top-k accuracy.

Usage:
  python evaluate.py --model convnext_tiny --checkpoint ../checkpoints/convnext_tiny/best.pt
  python evaluate.py --model svm_resnet_features
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from dataset import get_dataloaders, get_data_root
from models import build_model, get_model_names
from models.svm_classifier import extract_features
from metrics import compute_metrics, print_metrics, save_confusion_matrix, save_metrics_report
import joblib


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def evaluate_dl(model, loader, device: torch.device, top_k: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return y_true, y_pred, y_prob (for top-k)."""
    model.eval()
    all_preds, all_labels = [], []
    all_probs = []

    for imgs, labels in tqdm(loader, desc="Evaluating"):
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy().tolist())
        all_probs.append(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.vstack(all_probs) if all_probs else None
    return y_true, y_pred, y_prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name or svm_resnet_features")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    data_root = args.data_root or get_data_root()
    output_dir = args.output_dir or _PROJECT_ROOT / "logs" / "eval" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader = get_dataloaders(batch_size=32, num_workers=4, root=data_root)
    class_names = test_loader.dataset.classes
    num_classes = len(class_names)

    if args.model == "svm_resnet_features":
        ckpt_path = args.checkpoint or _PROJECT_ROOT / "checkpoints" / "svm_resnet_features" / "svm.joblib"
        data = joblib.load(ckpt_path)
        svm, scaler = data["svm"], data["scaler"]
        X_test, y_true = extract_features(test_loader, device)
        X_scaled = scaler.transform(X_test)
        y_pred = svm.predict(X_scaled)
        y_prob = svm.predict_proba(X_scaled) if hasattr(svm, "predict_proba") else None
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    else:
        ckpt_path = args.checkpoint or _PROJECT_ROOT / "checkpoints" / args.model / "best.pt"
        model = build_model(args.model, num_classes)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model = model.to(device)
        y_true, y_pred, y_prob = evaluate_dl(model, test_loader, device, args.top_k)

    metrics = compute_metrics(
        y_true, y_pred,
        y_prob=y_prob,
        class_names=class_names,
        top_k=args.top_k,
        num_classes=num_classes,
    )

    print_metrics(metrics, top_k=args.top_k)

    save_metrics_report(metrics, output_dir / "metrics.txt", top_k=args.top_k)
    save_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        output_dir / "confusion_matrix.png",
    )
    print(f"\nSaved to {output_dir}/")


if __name__ == "__main__":
    main()

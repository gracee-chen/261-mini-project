"""
Compute top-5 accuracy and macro/weighted P/R/F1 for all 6 models.
Run after training:
    cd model/ && python evaluate_metrics.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import joblib
from sklearn.metrics import precision_recall_fscore_support

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from dataset import get_dataloaders
from models import build_model
from models.svm_classifier import FeatureExtractor

CHECKPOINT_DIR = _PROJECT_ROOT / "checkpoints"

MODEL_NAMES = ["resnet50", "efficientnet_b2", "vit_b_16", "convnext_tiny", "eva02_small", "svm_resnet_features"]
DISPLAY_NAMES = ["ResNet-50", "EfficientNet-B2", "ViT-B/16", "ConvNeXt-Tiny", "EVA-02-Small", "SVM+ResNet-18"]


def topk_accuracy(probs: np.ndarray, y_true: np.ndarray, k: int = 5) -> float:
    """Compute top-k accuracy from probability/score matrix."""
    topk_preds = np.argsort(probs, axis=1)[:, -k:]  # (N, k)
    correct = np.any(topk_preds == y_true[:, None], axis=1)
    return correct.mean()


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    _, _, test_loader = get_dataloaders(batch_size=32, num_workers=4)
    num_classes = len(test_loader.dataset.classes)

    print(f"{'Model':<18} {'Top-1':>6} {'Top-5':>6} {'P(mac)':>7} {'R(mac)':>7} {'F1(mac)':>8} {'P(wt)':>7} {'R(wt)':>7} {'F1(wt)':>8}")
    print("-" * 90)

    for model_name, display_name in zip(MODEL_NAMES, DISPLAY_NAMES):
        if model_name == "svm_resnet_features":
            ckpt_path = CHECKPOINT_DIR / "svm_resnet_features" / "svm.joblib"
            data = joblib.load(ckpt_path)
            svm, scaler = data["svm"], data["scaler"]
            extractor = FeatureExtractor().to(device).eval()
            feats_list, labels_list = [], []
            with torch.no_grad():
                for imgs, labels in test_loader:
                    feats_list.append(extractor(imgs.to(device)).cpu().numpy())
                    labels_list.append(labels.numpy())
            X_test = np.vstack(feats_list)
            y_true = np.concatenate(labels_list)
            X_scaled = scaler.transform(X_test)
            y_pred = svm.predict(X_scaled)
            probs = svm.predict_proba(X_scaled)
        else:
            ckpt_path = CHECKPOINT_DIR / model_name / "best.pt"
            model = build_model(model_name, num_classes, pretrained=False)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model"])
            model = model.to(device).eval()
            all_logits, all_labels = [], []
            with torch.no_grad():
                for imgs, labels in test_loader:
                    logits = model(imgs.to(device))
                    all_logits.append(logits.cpu().numpy())
                    all_labels.append(labels.numpy())
            logits_all = np.vstack(all_logits)
            y_true = np.concatenate(all_labels)
            y_pred = logits_all.argmax(axis=1)
            # softmax for top-k
            exp_logits = np.exp(logits_all - logits_all.max(axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        top1 = (y_pred == y_true).mean()
        top5 = topk_accuracy(probs, y_true, k=5)

        p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

        print(f"{display_name:<18} {top1:>6.2%} {top5:>6.2%} {p_mac:>7.4f} {r_mac:>7.4f} {f1_mac:>8.4f} {p_wt:>7.4f} {r_wt:>7.4f} {f1_wt:>8.4f}")


if __name__ == "__main__":
    main()

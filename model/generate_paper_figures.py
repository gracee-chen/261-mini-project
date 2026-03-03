"""
Generate publication-quality evaluation figures for the paper.
Run after training all models:
    cd model/
    python generate_paper_figures.py

Outputs:
    ../figures/fig_acc_epochs.pdf      — val accuracy vs epochs (all models)
    ../figures/fig_perclass_all.pdf    — per-class accuracy (5 models in one row)
    ../figures/fig_cm_all.pdf          — confusion matrices (5 models in one row)
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------- paths ----------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from dataset import get_dataloaders
from models import build_model
from models.svm_classifier import extract_features, FeatureExtractor

FIGURE_DIR = _PROJECT_ROOT / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = _PROJECT_ROOT / "checkpoints"

# ---------- matplotlib global style (publication) ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

MODEL_NAMES = ["resnet50", "efficientnet_b2", "vit_b_16", "convnext_tiny", "eva02_small", "svm_resnet_features"]
DISPLAY_NAMES = ["ResNet-50", "EfficientNet-B2", "ViT-B/16", "ConvNeXt-Tiny", "EVA-02-S", "SVM+ResNet-18"]
COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"]
MARKERS = ["o", "s", "^", "D", "P", "v"]


# ================================================================
# Evaluate all models -> collect predictions + per-class metrics
# ================================================================
def evaluate_all():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    _, _, test_loader = get_dataloaders(batch_size=32, num_workers=4)
    class_names = test_loader.dataset.classes
    num_classes = len(class_names)

    results = {}
    for model_name in MODEL_NAMES:
        print(f"  {model_name}...", end=" ")

        if model_name == "svm_resnet_features":
            ckpt_path = CHECKPOINT_DIR / "svm_resnet_features" / "svm.joblib"
            data = joblib.load(ckpt_path)
            svm, scaler = data["svm"], data["scaler"]
            X_test, y_true = extract_features(test_loader, device)
            X_scaled = scaler.transform(X_test)
            y_pred = svm.predict(X_scaled)
            y_true, y_pred = np.array(y_true), np.array(y_pred)
        else:
            ckpt_path = CHECKPOINT_DIR / model_name / "best.pt"
            model = build_model(model_name, num_classes, pretrained=False)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model"])
            model = model.to(device).eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs = imgs.to(device)
                    logits = model(imgs)
                    all_preds.extend(logits.argmax(1).cpu().numpy())
                    all_labels.extend(labels.numpy())
            y_true = np.array(all_labels)
            y_pred = np.array(all_preds)

        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
        per_class_acc = np.where(cm.sum(1) > 0, np.diag(cm) / cm.sum(1), 0.0)

        results[model_name] = {
            "accuracy": acc,
            "per_class_accuracy": per_class_acc,
            "confusion_matrix": cm,
        }
        print(f"{acc:.4f}")

    return results, class_names, num_classes


# ================================================================
# Figure 1: Val accuracy + loss vs epochs (two subplots)
# ================================================================
def fig_acc_loss_epochs():
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 3.8))

    for i, (mname, dname) in enumerate(zip(MODEL_NAMES, DISPLAY_NAMES)):
        hist_path = CHECKPOINT_DIR / mname / "history.json"
        if not hist_path.exists():
            print(f"  WARNING: {hist_path} not found, skipping {dname}")
            continue

        with open(hist_path) as f:
            history = json.load(f)

        val_acc = history["val_acc"]
        train_loss = history["train_loss"]
        val_loss = history["val_loss"]
        epochs = list(range(1, len(val_acc) + 1))

        if mname == "svm_resnet_features":
            ax_acc.axhline(y=val_acc[0], color=COLORS[i], linestyle="--",
                           linewidth=1.2, label=f"{dname} ({val_acc[0]:.1%})")
        else:
            ax_acc.plot(epochs, val_acc, color=COLORS[i], marker=MARKERS[i],
                        markersize=4, linewidth=1.5, label=dname)
            ax_loss.plot(epochs, train_loss, color=COLORS[i], marker=MARKERS[i],
                         markersize=4, linewidth=1.5, linestyle="-", label=f"{dname} (train)")
            ax_loss.plot(epochs, val_loss, color=COLORS[i], marker=MARKERS[i],
                         markersize=3, linewidth=1.2, linestyle="--", alpha=0.7)

    # Accuracy subplot
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Validation Accuracy")
    ax_acc.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax_acc.legend(loc="lower right", frameon=True, framealpha=0.9,
                  edgecolor="gray", fancybox=False, fontsize=8)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.set_title("(a) Validation Accuracy", fontsize=11, pad=8)

    # Loss subplot
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(loc="upper right", frameon=True, framealpha=0.9,
                   edgecolor="gray", fancybox=False, fontsize=7)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.set_title("(b) Training (solid) / Validation (dashed) Loss", fontsize=11, pad=8)

    fig.subplots_adjust(wspace=0.28)

    fig.savefig(FIGURE_DIR / "fig_acc_loss_epochs.pdf")
    fig.savefig(FIGURE_DIR / "fig_acc_loss_epochs.png")
    plt.close(fig)
    print("  fig_acc_loss_epochs")


# ================================================================
# Figure 2: Per-class accuracy — 5 models side by side in one row
# ================================================================
def fig_perclass_all(results, class_names):
    n_models = len(MODEL_NAMES)
    n_classes = len(class_names)

    fig, axes = plt.subplots(1, n_models, figsize=(n_models * 3.2, 14), sharey=True)

    # Use a shared sort order (by average accuracy across all models)
    avg_acc = np.mean([results[m]["per_class_accuracy"] for m in MODEL_NAMES], axis=0)
    order = np.argsort(avg_acc)
    sorted_names = [class_names[j] for j in order]

    for i, (mname, dname) in enumerate(zip(MODEL_NAMES, DISPLAY_NAMES)):
        ax = axes[i]
        per_class = results[mname]["per_class_accuracy"]
        sorted_acc = per_class[order]

        y_pos = np.arange(n_classes)
        ax.barh(y_pos, sorted_acc, height=0.7, color=COLORS[i],
                edgecolor="white", linewidth=0.2)

        # Overall accuracy line
        overall = results[mname]["accuracy"]
        ax.axvline(x=overall, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

        ax.set_xlim(0, 1.05)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_title(f"{dname}\n({overall:.1%})", fontsize=9, pad=6)
        ax.tick_params(axis="x", labelsize=7)

        if i == 0:
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_names, fontsize=3.5)
        else:
            ax.tick_params(axis="y", left=False)

    fig.subplots_adjust(wspace=0.08)

    fig.savefig(FIGURE_DIR / "fig_perclass_all.pdf")
    fig.savefig(FIGURE_DIR / "fig_perclass_all.png")
    plt.close(fig)
    print("  fig_perclass_all")


# ================================================================
# Figure 3: Confusion matrices — 5 models side by side in one row
# ================================================================
def fig_cm_all(results, class_names):
    n_models = len(MODEL_NAMES)
    n_classes = len(class_names)

    fig, axes = plt.subplots(1, n_models, figsize=(n_models * 3.5, 3.8))

    for i, (mname, dname) in enumerate(zip(MODEL_NAMES, DISPLAY_NAMES)):
        ax = axes[i]
        cm = results[mname]["confusion_matrix"]
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm.astype(float) / row_sums, 0.0)

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)

        ax.set_title(f"{dname}", fontsize=9, pad=6)

        # Sparse ticks for readability
        tick_positions = list(range(0, n_classes, 20))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_positions, fontsize=6)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_positions, fontsize=6)

        if i == 0:
            ax.set_ylabel("True Label", fontsize=9)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)

        ax.set_xlabel("Predicted", fontsize=8)

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.015, pad=0.02, shrink=0.85)
    cbar.set_label("Recall", fontsize=9)

    fig.savefig(FIGURE_DIR / "fig_cm_all.pdf")
    fig.savefig(FIGURE_DIR / "fig_cm_all.png")
    plt.close(fig)
    print("  fig_cm_all")


# ================================================================
# main
# ================================================================
if __name__ == "__main__":
    print("Evaluating models...")
    results, class_names, num_classes = evaluate_all()

    fig_acc_loss_epochs()
    fig_perclass_all(results, class_names)
    fig_cm_all(results, class_names)

    print("Done. Figures saved to figures/")

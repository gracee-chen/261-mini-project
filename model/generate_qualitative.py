"""
Generate qualitative comparison figure: 5 rows (models) x N columns (test images).
Shows where models agree/disagree — correct predictions in green, wrong in red.
Picks a mix of "all-correct", "some-wrong", and "all-wrong" images for contrast.

Usage: cd model/ && python generate_qualitative.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from dataset import get_dataloaders, get_data_root, Caltech101Dataset
from models import build_model
from models.svm_classifier import FeatureExtractor

FIGURE_DIR = _PROJECT_ROOT / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = ["resnet50", "efficientnet_b2", "vit_b_16", "convnext_tiny", "svm_resnet_features"]
DISPLAY_NAMES = ["ResNet-50", "EfficientNet-B2", "ViT-B/16", "ConvNeXt-Tiny", "SVM+ResNet-18"]

NUM_COLS = 12  # number of images to show

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def load_raw_images(test_ds, indices):
    """Load original (un-normalized) images for display."""
    display_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    imgs = []
    for idx in indices:
        path, label = test_ds.samples[idx]
        img = Image.open(path).convert("RGB")
        imgs.append(display_tf(img))
    return imgs


def get_all_predictions(device):
    """Get predictions from all 5 models on the test set."""
    _, _, test_loader = get_dataloaders(batch_size=32, num_workers=4)
    test_ds = test_loader.dataset
    class_names = test_ds.classes
    num_classes = len(class_names)

    # Collect ground truth
    y_true = np.array([label for _, label in test_ds.samples])

    all_preds = {}

    for model_name in MODEL_NAMES:
        print(f"Predicting with {model_name}...")

        if model_name == "svm_resnet_features":
            ckpt_path = _PROJECT_ROOT / "checkpoints" / "svm_resnet_features" / "svm.joblib"
            data = joblib.load(ckpt_path)
            svm, scaler = data["svm"], data["scaler"]
            extractor = FeatureExtractor().to(device).eval()
            feats_list, labels_list = [], []
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs = imgs.to(device)
                    feats = extractor(imgs).cpu().numpy()
                    feats_list.append(feats)
                    labels_list.append(labels.numpy())
            X_test = np.vstack(feats_list)
            X_scaled = scaler.transform(X_test)
            y_pred = svm.predict(X_scaled)
        else:
            ckpt_path = _PROJECT_ROOT / "checkpoints" / model_name / "best.pt"
            model = build_model(model_name, num_classes)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model"])
            model = model.to(device).eval()
            preds_list = []
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs = imgs.to(device)
                    logits = model(imgs)
                    preds_list.extend(logits.argmax(1).cpu().numpy())
            y_pred = np.array(preds_list)

        all_preds[model_name] = y_pred

    return y_true, all_preds, test_ds, class_names


def select_images(y_true, all_preds, n=NUM_COLS):
    """Select interesting images: mix of all-correct, some-wrong, all-wrong."""
    n_samples = len(y_true)
    pred_matrix = np.stack([all_preds[m] for m in MODEL_NAMES], axis=0)  # (5, N)
    correct_matrix = (pred_matrix == y_true[None, :])  # (5, N)
    n_correct = correct_matrix.sum(axis=0)  # per image: how many models got it right

    # Categories
    all_correct = np.where(n_correct == 5)[0]
    some_wrong = np.where((n_correct >= 1) & (n_correct <= 4))[0]
    all_wrong = np.where(n_correct == 0)[0]

    rng = np.random.default_rng(42)
    selected = []

    # Pick 3 all-correct, 6 some-wrong (most interesting!), 3 all-wrong
    n_all_correct = min(3, len(all_correct))
    n_some_wrong = min(6, len(some_wrong))
    n_all_wrong = min(3, len(all_wrong))

    if len(all_correct) > 0:
        selected.extend(rng.choice(all_correct, n_all_correct, replace=False).tolist())
    if len(some_wrong) > 0:
        # Prefer images where models disagree most (2-3 correct)
        disagree = some_wrong[np.isin(n_correct[some_wrong], [2, 3, 4])]
        if len(disagree) >= n_some_wrong:
            selected.extend(rng.choice(disagree, n_some_wrong, replace=False).tolist())
        else:
            selected.extend(rng.choice(some_wrong, n_some_wrong, replace=False).tolist())
    if len(all_wrong) > 0:
        selected.extend(rng.choice(all_wrong, n_all_wrong, replace=False).tolist())

    # Fill remaining slots
    remaining = n - len(selected)
    if remaining > 0:
        pool = list(set(range(n_samples)) - set(selected))
        selected.extend(rng.choice(pool, remaining, replace=False).tolist())

    return selected[:n]


def generate_figure(y_true, all_preds, test_ds, class_names, indices):
    """Generate the qualitative comparison figure."""
    raw_imgs = load_raw_images(test_ds, indices)
    n_cols = len(indices)
    n_rows = len(MODEL_NAMES)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.3, n_rows * 1.6))

    for row, (model_name, display_name) in enumerate(zip(MODEL_NAMES, DISPLAY_NAMES)):
        for col, idx in enumerate(indices):
            ax = axes[row, col]
            img = raw_imgs[col].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])

            gt = y_true[idx]
            pred = all_preds[model_name][idx]
            pred_name = class_names[pred]
            # Truncate long names
            if len(pred_name) > 12:
                pred_name = pred_name[:11] + "."

            is_correct = (pred == gt)
            color = "#228833" if is_correct else "#CC3311"
            symbol = "✓" if is_correct else "✗"

            ax.set_title(f"{symbol} {pred_name}", fontsize=6, color=color,
                         fontweight="bold", pad=2)

            # Thick border for wrong predictions
            if not is_correct:
                for spine in ax.spines.values():
                    spine.set_edgecolor("#CC3311")
                    spine.set_linewidth(2)
            else:
                for spine in ax.spines.values():
                    spine.set_edgecolor("#CCCCCC")
                    spine.set_linewidth(0.5)

        # Row label (model name)
        axes[row, 0].set_ylabel(display_name, fontsize=9, fontweight="bold",
                                rotation=90, labelpad=8)

    # Column header: ground truth labels
    for col, idx in enumerate(indices):
        gt_name = class_names[y_true[idx]]
        if len(gt_name) > 14:
            gt_name = gt_name[:13] + "."
        axes[0, col].annotate(
            gt_name, xy=(0.5, 1.35), xycoords="axes fraction",
            ha="center", va="bottom", fontsize=6.5, fontstyle="italic",
            color="#333333",
        )

    fig.suptitle("Qualitative Classification Comparison on Caltech-101 Test Set\n"
                 "(green ✓ = correct, red ✗ = incorrect; column headers = ground truth)",
                 fontsize=10, y=1.04, fontweight="bold")

    fig.subplots_adjust(hspace=0.45, wspace=0.15)

    fig.savefig(FIGURE_DIR / "fig_qualitative_comparison.pdf")
    fig.savefig(FIGURE_DIR / "fig_qualitative_comparison.png")
    plt.close(fig)
    print(f"Saved to {FIGURE_DIR / 'fig_qualitative_comparison.pdf'}")


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    y_true, all_preds, test_ds, class_names = get_all_predictions(device)

    # Print summary
    pred_matrix = np.stack([all_preds[m] for m in MODEL_NAMES], axis=0)
    correct_matrix = (pred_matrix == y_true[None, :])
    n_correct = correct_matrix.sum(axis=0)
    print(f"\nImage agreement stats:")
    print(f"  All 5 correct: {(n_correct==5).sum()}")
    print(f"  4 correct:     {(n_correct==4).sum()}")
    print(f"  3 correct:     {(n_correct==3).sum()}")
    print(f"  2 correct:     {(n_correct==2).sum()}")
    print(f"  1 correct:     {(n_correct==1).sum()}")
    print(f"  All wrong:     {(n_correct==0).sum()}")

    indices = select_images(y_true, all_preds, n=NUM_COLS)
    generate_figure(y_true, all_preds, test_ds, class_names, indices)

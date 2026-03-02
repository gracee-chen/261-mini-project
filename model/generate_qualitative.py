"""
Generate qualitative comparison figure: 5 rows (models) x 8 columns (test images).
Compact layout for paper front page. Green/red borders only, label overlaid on image.

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
from matplotlib.patches import FancyBboxPatch
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
DISPLAY_NAMES = ["ResNet-50", "EffNet-B2", "ViT-B/16", "ConvNeXt", "SVM"]

NUM_COLS = 8

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

IMG_SIZE = 96


def load_raw_images(test_ds, indices):
    display_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    imgs = []
    for idx in indices:
        path, label = test_ds.samples[idx]
        img = Image.open(path).convert("RGB")
        imgs.append(display_tf(img))
    return imgs


def get_all_predictions(device):
    _, _, test_loader = get_dataloaders(batch_size=32, num_workers=4)
    test_ds = test_loader.dataset
    class_names = test_ds.classes
    num_classes = len(class_names)
    y_true = np.array([label for _, label in test_ds.samples])
    all_preds = {}

    for model_name in MODEL_NAMES:
        print(f"Predicting with {model_name}...")
        if model_name == "svm_resnet_features":
            ckpt_path = _PROJECT_ROOT / "checkpoints" / "svm_resnet_features" / "svm.joblib"
            data = joblib.load(ckpt_path)
            svm, scaler = data["svm"], data["scaler"]
            extractor = FeatureExtractor().to(device).eval()
            feats_list = []
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs = imgs.to(device)
                    feats_list.append(extractor(imgs).cpu().numpy())
            X_test = np.vstack(feats_list)
            y_pred = svm.predict(scaler.transform(X_test))
        else:
            ckpt_path = _PROJECT_ROOT / "checkpoints" / model_name / "best.pt"
            model = build_model(model_name, num_classes)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model"])
            model = model.to(device).eval()
            preds_list = []
            with torch.no_grad():
                for imgs, labels in test_loader:
                    logits = model(imgs.to(device))
                    preds_list.extend(logits.argmax(1).cpu().numpy())
            y_pred = np.array(preds_list)
        all_preds[model_name] = y_pred

    return y_true, all_preds, test_ds, class_names


def select_images(y_true, all_preds, n=NUM_COLS):
    pred_matrix = np.stack([all_preds[m] for m in MODEL_NAMES], axis=0)
    correct_matrix = (pred_matrix == y_true[None, :])
    n_correct = correct_matrix.sum(axis=0)

    all_correct = np.where(n_correct == 5)[0]
    some_wrong = np.where((n_correct >= 1) & (n_correct <= 4))[0]
    all_wrong = np.where(n_correct == 0)[0]

    rng = np.random.default_rng(42)
    selected = []

    # 2 all-correct, 4 some-wrong, 2 all-wrong
    if len(all_correct) > 0:
        selected.extend(rng.choice(all_correct, min(2, len(all_correct)), replace=False).tolist())
    if len(some_wrong) > 0:
        disagree = some_wrong[np.isin(n_correct[some_wrong], [2, 3, 4])]
        pool = disagree if len(disagree) >= 4 else some_wrong
        selected.extend(rng.choice(pool, min(4, len(pool)), replace=False).tolist())
    if len(all_wrong) > 0:
        selected.extend(rng.choice(all_wrong, min(2, len(all_wrong)), replace=False).tolist())

    remaining = n - len(selected)
    if remaining > 0:
        leftover = list(set(range(len(y_true))) - set(selected))
        selected.extend(rng.choice(leftover, remaining, replace=False).tolist())

    return selected[:n]


def generate_figure(y_true, all_preds, test_ds, class_names, indices):
    raw_imgs = load_raw_images(test_ds, indices)
    n_cols = len(indices)
    n_rows = len(MODEL_NAMES)

    cell_w, cell_h = 1.0, 1.0
    fig_w = n_cols * cell_w + 1.0   # extra for row labels
    fig_h = n_rows * cell_h + 0.5   # extra for GT header
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))

    GREEN = "#2ca02c"
    RED = "#d62728"

    for row, (model_name, display_name) in enumerate(zip(MODEL_NAMES, DISPLAY_NAMES)):
        for col, idx in enumerate(indices):
            ax = axes[row, col]
            img = raw_imgs[col].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])

            gt = y_true[idx]
            pred = all_preds[model_name][idx]
            is_correct = (pred == gt)

            # Border color
            border_color = GREEN if is_correct else RED
            border_width = 2.5
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(border_width)

            # Overlay label on top-left of image
            pred_name = class_names[pred]
            if len(pred_name) > 10:
                pred_name = pred_name[:9] + "."
            ax.text(
                2, 4, pred_name,
                fontsize=4.5, fontweight="bold", color="white",
                ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=border_color,
                          edgecolor="none", alpha=0.85),
                transform=ax.transData,
            )

        # Row label
        axes[row, 0].set_ylabel(
            display_name, fontsize=7, fontweight="bold",
            rotation=90, labelpad=5,
        )

    # GT header on top row
    for col, idx in enumerate(indices):
        gt_name = class_names[y_true[idx]]
        if len(gt_name) > 12:
            gt_name = gt_name[:11] + "."
        axes[0, col].set_title(gt_name, fontsize=5.5, fontstyle="italic",
                                color="#444444", pad=3)

    fig.subplots_adjust(hspace=0.12, wspace=0.08)

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

    pred_matrix = np.stack([all_preds[m] for m in MODEL_NAMES], axis=0)
    correct_matrix = (pred_matrix == y_true[None, :])
    n_correct = correct_matrix.sum(axis=0)
    print(f"\nImage agreement stats:")
    for k in [5, 4, 3, 2, 1, 0]:
        label = "All 5 correct" if k == 5 else ("All wrong" if k == 0 else f"{k} correct")
        print(f"  {label:>15}: {(n_correct==k).sum()}")

    indices = select_images(y_true, all_preds, n=NUM_COLS)
    generate_figure(y_true, all_preds, test_ds, class_names, indices)

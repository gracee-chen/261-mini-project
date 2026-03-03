"""
Generate qualitative comparison figure: 6 rows (models) x 8 columns (test images).
Compact layout for paper front page. Maximizes visible disagreement between models.

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

MODEL_NAMES = ["resnet50", "efficientnet_b2", "vit_b_16", "convnext_tiny", "eva02_small", "svm_resnet_features"]
DISPLAY_NAMES = ["ResNet-50", "EffNet-B2", "ViT-B/16", "ConvNeXt", "EVA-02-S", "SVM"]
# Index of DL-only models (exclude SVM for disagreement scoring)
DL_INDICES = [0, 1, 2, 3, 4]

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
        print(f"  {model_name}...", end=" ", flush=True)
        if model_name == "svm_resnet_features":
            ckpt_path = _PROJECT_ROOT / "checkpoints" / "svm_resnet_features" / "svm.joblib"
            data = joblib.load(ckpt_path)
            svm, scaler = data["svm"], data["scaler"]
            extractor = FeatureExtractor().to(device).eval()
            feats_list = []
            with torch.no_grad():
                for imgs, labels in test_loader:
                    feats_list.append(extractor(imgs.to(device)).cpu().numpy())
            X_test = np.vstack(feats_list)
            y_pred = svm.predict(scaler.transform(X_test))
        else:
            ckpt_path = _PROJECT_ROOT / "checkpoints" / model_name / "best.pt"
            model = build_model(model_name, num_classes, pretrained=False)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model"])
            model = model.to(device).eval()
            preds_list = []
            with torch.no_grad():
                for imgs, labels in test_loader:
                    preds_list.extend(model(imgs.to(device)).argmax(1).cpu().numpy())
            y_pred = np.array(preds_list)
        all_preds[model_name] = y_pred
        print("ok")

    return y_true, all_preds, test_ds, class_names


def select_images(y_true, all_preds, n=NUM_COLS):
    """Select images that maximize visible disagreement across models."""
    n_models = len(MODEL_NAMES)
    pred_matrix = np.stack([all_preds[m] for m in MODEL_NAMES], axis=0)  # (n_models, N)
    correct_matrix = (pred_matrix == y_true[None, :])  # (n_models, N)
    n_correct_all = correct_matrix.sum(axis=0)  # how many models correct

    # DL-only correctness (exclude SVM)
    dl_correct = correct_matrix[DL_INDICES, :]  # (n_dl, N)
    n_dl_correct = dl_correct.sum(axis=0)
    n_dl = len(DL_INDICES)

    # Count unique predictions per image (more unique = more disagreement)
    n_unique_preds = np.array([len(set(pred_matrix[:, i])) for i in range(len(y_true))])

    rng = np.random.default_rng(42)
    selected = []

    # --- Column 1: all models correct (easy case, for contrast) ---
    easy = np.where(n_correct_all == n_models)[0]
    if len(easy) > 0:
        selected.append(rng.choice(easy, 1)[0])

    # --- Columns 2-4: DL models disagree (1-3 DL correct, not all same) ---
    dl_disagree = np.where((n_dl_correct >= 1) & (n_dl_correct <= 3))[0]
    if len(dl_disagree) > 0:
        scores = n_unique_preds[dl_disagree]
        top_idx = dl_disagree[np.argsort(-scores)]
        top_idx = [i for i in top_idx if i not in selected]
        selected.extend(top_idx[:3])

    # --- Column 5: only SVM wrong, all DL correct ---
    svm_only_wrong = np.where((n_dl_correct == n_dl) & (n_correct_all == n_dl))[0]
    if len(svm_only_wrong) > 0:
        pool = [i for i in svm_only_wrong if i not in selected]
        if pool:
            selected.append(rng.choice(pool, 1)[0])

    # --- Column 6: only ConvNeXt correct (shows its strength) ---
    convnext_idx = MODEL_NAMES.index("convnext_tiny")
    only_convnext = np.where(
        (correct_matrix[convnext_idx, :] == True) & (n_correct_all <= 2)
    )[0]
    if len(only_convnext) > 0:
        pool = [i for i in only_convnext if i not in selected]
        if pool:
            selected.append(rng.choice(pool, 1)[0])

    # --- Column 7: ViT wrong but others correct (ViT weakness) ---
    vit_idx = MODEL_NAMES.index("vit_b_16")
    vit_wrong = np.where(
        (correct_matrix[vit_idx, :] == False) & (n_dl_correct >= 3)
    )[0]
    if len(vit_wrong) > 0:
        pool = [i for i in vit_wrong if i not in selected]
        if pool:
            selected.append(rng.choice(pool, 1)[0])

    # --- Column 8: all wrong (hardest) ---
    all_wrong = np.where(n_correct_all == 0)[0]
    if len(all_wrong) > 0:
        pool = [i for i in all_wrong if i not in selected]
        if pool:
            scores = n_unique_preds[pool]
            selected.append(pool[np.argmax(scores)])

    # Fill remaining slots with max-disagreement images
    remaining = n - len(selected)
    if remaining > 0:
        pool = [i for i in range(len(y_true)) if i not in selected]
        scores = n_unique_preds[pool]
        top = np.argsort(-scores)[:remaining]
        selected.extend([pool[i] for i in top])

    return selected[:n]


def generate_figure(y_true, all_preds, test_ds, class_names, indices):
    raw_imgs = load_raw_images(test_ds, indices)
    n_cols = len(indices)
    n_rows = len(MODEL_NAMES)

    cell_w, cell_h = 1.2, 1.15
    fig_w = n_cols * cell_w + 0.8
    fig_h = n_rows * cell_h + 0.35
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

            border_color = GREEN if is_correct else RED
            # White inner padding between image and colored border
            ax.patch.set_facecolor("white")
            ax.set_xlim(-4, IMG_SIZE + 4)
            ax.set_ylim(IMG_SIZE + 4, -4)
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2.5)

            # Overlay predicted label
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
            display_name, fontsize=7.5, fontweight="bold",
            rotation=90, labelpad=5,
        )

    # GT header
    for col, idx in enumerate(indices):
        gt_name = class_names[y_true[idx]]
        if len(gt_name) > 12:
            gt_name = gt_name[:11] + "."
        axes[0, col].set_title(gt_name, fontsize=6, fontstyle="italic",
                                color="#444444", pad=3)

    fig.subplots_adjust(hspace=0.10, wspace=0.08)

    fig.savefig(FIGURE_DIR / "fig_qualitative_comparison.pdf")
    fig.savefig(FIGURE_DIR / "fig_qualitative_comparison.png")
    plt.close(fig)


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Generating qualitative comparison...")
    y_true, all_preds, test_ds, class_names = get_all_predictions(device)
    indices = select_images(y_true, all_preds, n=NUM_COLS)
    generate_figure(y_true, all_preds, test_ds, class_names, indices)
    print("Done. Saved fig_qualitative_comparison.pdf")

"""
Ablation: HOG features vs CNN features (both fed to SVM) on Caltech-101.
"""

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_MODEL_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.feature import hog

from dataset import get_dataloaders
from models.svm_classifier import FeatureExtractor, extract_features, train_svm, evaluate_svm


HOG_IMG_SIZE = 128


def extract_hog_features(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Extract HOG features from a dataloader.

    Each image is converted to grayscale, resized to HOG_IMG_SIZExHOG_IMG_SIZE,
    and HOG descriptors are computed.
    """
    all_features = []
    all_labels = []

    for imgs, labels in loader:
        # imgs: (B, 3, H, W) tensor, already normalized — undo normalization isn't
        # strictly needed for HOG on grayscale, but we convert to [0,1] range first.
        imgs_np = imgs.numpy()
        for i in range(imgs_np.shape[0]):
            img = imgs_np[i]  # (3, H, W)
            # Convert to grayscale: 0.299*R + 0.587*G + 0.114*B
            # First un-normalize: img = img * std + mean
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            img = img * std + mean
            img = np.clip(img, 0, 1)

            gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]

            # Resize to fixed size using simple interpolation
            from PIL import Image as _PILImage
            gray_pil = _PILImage.fromarray((gray * 255).astype(np.uint8), mode="L")
            gray_pil = gray_pil.resize((HOG_IMG_SIZE, HOG_IMG_SIZE))
            gray_resized = np.array(gray_pil, dtype=np.float64) / 255.0

            feat = hog(
                gray_resized,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
            )
            all_features.append(feat)

        all_labels.append(labels.numpy())

    X = np.array(all_features)
    y = np.concatenate(all_labels)
    return X, y


def run(device: torch.device) -> dict[str, dict]:
    train_loader, val_loader, _ = get_dataloaders(batch_size=32, num_workers=4)
    results = {}

    # --- CNN features (ResNet-18) ---
    print(f"\n{'='*50}")
    print("  Feature type: CNN (ResNet-18)")
    print(f"{'='*50}")

    extractor = FeatureExtractor().to(device).eval()
    for p in extractor.parameters():
        p.requires_grad = False

    print("  Extracting CNN train features...")
    X_train_cnn, y_train = extract_features(train_loader, device, extractor=extractor)
    print("  Extracting CNN val features...")
    X_val_cnn, y_val = extract_features(val_loader, device, extractor=extractor)

    print("  Training SVM on CNN features...")
    svm_cnn, scaler_cnn = train_svm(X_train_cnn, y_train, X_val_cnn, y_val)
    cnn_acc = evaluate_svm(svm_cnn, scaler_cnn, X_val_cnn, y_val)

    results["CNN (ResNet-18)"] = {"dim": X_train_cnn.shape[1], "acc": cnn_acc}
    print(f"  CNN features dim={X_train_cnn.shape[1]}, val_acc={cnn_acc:.4f}")

    # --- HOG features ---
    print(f"\n{'='*50}")
    print("  Feature type: HOG")
    print(f"{'='*50}")

    print("  Extracting HOG train features...")
    X_train_hog, y_train_hog = extract_hog_features(train_loader)
    print("  Extracting HOG val features...")
    X_val_hog, y_val_hog = extract_hog_features(val_loader)

    print("  Training SVM on HOG features...")
    svm_hog, scaler_hog = train_svm(X_train_hog, y_train_hog, X_val_hog, y_val_hog)
    hog_acc = evaluate_svm(svm_hog, scaler_hog, X_val_hog, y_val_hog)

    results["HOG"] = {"dim": X_train_hog.shape[1], "acc": hog_acc}
    print(f"  HOG features dim={X_train_hog.shape[1]}, val_acc={hog_acc:.4f}")

    return results


def print_table(results: dict[str, dict]):
    print(f"\n{'='*50}")
    print("  Ablation: HOG vs CNN Features (SVM)")
    print(f"{'='*50}")
    print(f"  {'Features':<20} {'Dim':>6} {'Val Acc':>10}")
    print(f"  {'-'*36}")
    for name, info in results.items():
        print(f"  {name:<20} {info['dim']:>6} {info['acc']:>10.4f}")
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

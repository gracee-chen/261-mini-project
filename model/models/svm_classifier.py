"""
Classical ML: SVM with CNN features.
Extract features using frozen ResNet-18, then train SVM.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class FeatureExtractor(nn.Module):
    """ResNet-18 without final FC, for feature extraction."""

    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x.flatten(1)


def extract_features(
    loader: DataLoader,
    device: torch.device,
    extractor: FeatureExtractor | None = None,
    max_features: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features from dataset using frozen ResNet-18."""
    if extractor is None:
        extractor = FeatureExtractor().to(device).eval()
        for p in extractor.parameters():
            p.requires_grad = False

    all_features = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extracting features"):
            imgs = imgs.to(device)
            feats = extractor(imgs).cpu().numpy()
            all_features.append(feats)
            all_labels.append(labels.numpy())
            if max_features and len(all_features) * loader.batch_size >= max_features:
                break

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    return X, y


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    C: float = 1.0,
    kernel: str = "rbf",
) -> tuple[SVC, StandardScaler]:
    """Train SVM with optional scaling."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    svm = SVC(C=C, kernel=kernel, gamma="scale", probability=True, verbose=True)
    svm.fit(X_train_scaled, y_train)
    return svm, scaler


def evaluate_svm(
    svm: SVC,
    scaler: StandardScaler,
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    X_scaled = scaler.transform(X)
    preds = svm.predict(X_scaled)
    return (preds == y).mean()

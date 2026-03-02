"""
Caltech-101 Dataset and DataLoader utilities.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_data_root() -> Path:
    return _PROJECT_ROOT / "data" / "caltech101_split"


def load_classes(classes_path: Path) -> list[str]:
    with open(classes_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


class Caltech101Dataset(Dataset):
    """Image dataset from caltech101_split/{split}/{class_name}/*.jpg"""

    def __init__(
        self,
        split: str,
        root: Path | None = None,
        transform: transforms.Compose | None = None,
    ):
        self.root = root or get_data_root()
        self.split = split
        self.transform = transform or transforms.ToTensor()

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for class_name in self.classes:
            class_dir = split_dir / class_name
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".bmp"}:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(
    split: str,
    resize: int = 256,
    crop: int = 224,
    color_jitter: list[float] | None = None,
    horizontal_flip: bool = True,
) -> transforms.Compose:
    if split == "train":
        tfs = [
            transforms.Resize((resize, resize)),
            transforms.RandomCrop(crop),
        ]
        if horizontal_flip:
            tfs.append(transforms.RandomHorizontalFlip())
        if color_jitter:
            tfs.append(transforms.ColorJitter(*color_jitter))
        tfs.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25),
        ])
    else:
        tfs = [
            transforms.Resize((resize, resize)),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    return transforms.Compose(tfs)


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    resize: int = 256,
    crop: int = 224,
    color_jitter: list[float] | None = (0.4, 0.4, 0.4, 0.1),
    root: Path | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    root = root or get_data_root()

    train_tf = get_transforms("train", resize, crop, color_jitter)
    eval_tf = get_transforms("val", resize, crop, color_jitter=None, horizontal_flip=False)

    train_ds = Caltech101Dataset("train", root, train_tf)
    val_ds = Caltech101Dataset("val", root, eval_tf)
    test_ds = Caltech101Dataset("test", root, eval_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader

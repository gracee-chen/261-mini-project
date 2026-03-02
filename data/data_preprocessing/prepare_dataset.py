"""
Step 1: Dataset Preparation for Caltech-101
- Download from Kaggle
- Stratified split: 70% train, 15% validation, 15% test
"""

import shutil
import yaml
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm

# Resolve paths relative to project structure: project/data/data_preprocessing/
_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SCRIPT_DIR.parent


def load_config(config_path: Path | str | None = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = _SCRIPT_DIR / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_from_kaggle(kaggle_dataset: str, data_dir: Path) -> Path:
    """
    Download Caltech-101 from Kaggle.
    Requires: pip install kaggle
    You need Kaggle API credentials: ~/.kaggle/kaggle.json
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError("Install kaggle: pip install kaggle")

    try:
        api = KaggleApi()
        api.authenticate()
        data_dir.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(kaggle_dataset, path=str(data_dir), unzip=True)
    except Exception as e:
        manual_path = data_dir / "caltech-101"
        raise RuntimeError(
            f"Kaggle download failed: {e}\n\n"
            "To fix: 1) Kaggle Account -> API -> Create New Token, save as ~/.kaggle/kaggle.json\n"
            f"   OR 2) Manually download from https://www.kaggle.com/datasets/{kaggle_dataset}, "
            f"extract to {manual_path.absolute()} with 101_ObjectCategories/ inside, then run again."
        ) from e

    obj_cat = data_dir / "caltech-101" / "101_ObjectCategories"
    if obj_cat.exists():
        return obj_cat
    obj_cat = data_dir / "101_ObjectCategories"
    if obj_cat.exists():
        return obj_cat
    for d in data_dir.rglob("101_ObjectCategories"):
        if d.is_dir():
            return d
    return data_dir / "caltech-101"


def collect_images_by_class(raw_dir: Path, exclude_background: bool = False) -> dict:
    """
    Scan raw dataset and collect image paths grouped by class.
    Caltech-101 structure: each category = one folder with images.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    class_to_images = defaultdict(list)

    for category_dir in sorted(raw_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        class_name = category_dir.name

        if exclude_background and "BACKGROUND" in class_name.upper():
            continue

        for img_path in category_dir.glob("*"):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".bmp"}:
                class_to_images[class_name].append(img_path)

    class_to_images = {k: v for k, v in class_to_images.items() if len(v) > 0}
    return dict(class_to_images)


def stratified_split(
    class_to_images: dict,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[list, list, list]:
    """
    Stratified split: for each class, split images into train/val/test.
    Returns: (train_paths, val_paths, test_paths) - each is list of (path, class_name)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    train_list = []
    val_list = []
    test_list = []

    rng = np.random.default_rng(random_state)

    for class_name, paths in class_to_images.items():
        paths = list(paths)
        n = len(paths)
        if n == 0:
            continue

        indices = np.arange(n)
        rng.shuffle(indices)

        n_train = max(1, int(n * train_ratio))
        n_val = max(0, int(n * val_ratio))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = n - n_train

        idx_train = indices[:n_train]
        idx_val = indices[n_train : n_train + n_val]
        idx_test = indices[n_train + n_val :]

        for i in idx_train:
            train_list.append((paths[i], class_name))
        for i in idx_val:
            val_list.append((paths[i], class_name))
        for i in idx_test:
            test_list.append((paths[i], class_name))

    return train_list, val_list, test_list


def copy_to_split_structure(
    split_lists: tuple,
    output_dir: Path,
) -> None:
    """Copy images to train/val/test folders, preserving class subfolders."""
    splits = ["train", "val", "test"]
    for split_name, file_list in zip(splits, split_lists):
        split_dir = output_dir / split_name
        for src_path, class_name in tqdm(file_list, desc=f"Copying {split_name}"):
            dst_dir = split_dir / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = dst_dir / src_path.name
            shutil.copy2(src_path, dst_path)


def prepare_dataset(config_path: Path | str | None = None) -> None:
    """Main pipeline: download (or use existing), split, and organize."""
    config = load_config(config_path)
    ds = config["dataset"]
    split_cfg = config["split"]
    exclude_bg = config.get("exclude_background", False)

    raw_dir = _DATA_DIR / ds["raw_subdir"]
    output_dir = _DATA_DIR / ds["output_subdir"]

    if not raw_dir.exists():
        print("Downloading from Kaggle...")
        raw_dir = download_from_kaggle(ds["kaggle_dataset"], _DATA_DIR)
        if not raw_dir.exists():
            raw_dir = _DATA_DIR / ds["kaggle_dataset"].split("/")[-1]
    else:
        candidates = [
            raw_dir,
            raw_dir / "101_ObjectCategories",
            _DATA_DIR / "caltech-101" / "101_ObjectCategories",
        ]
        for c in candidates:
            if c.exists() and any(c.iterdir()):
                raw_dir = c
                break

    print(f"Using raw data from: {raw_dir}")

    class_to_images = collect_images_by_class(raw_dir, exclude_background=exclude_bg)
    n_classes = len(class_to_images)
    n_images = sum(len(v) for v in class_to_images.values())
    print(f"Found {n_classes} classes, {n_images} images")

    train_list, val_list, test_list = stratified_split(
        class_to_images,
        train_ratio=split_cfg["train"],
        val_ratio=split_cfg["val"],
        test_ratio=split_cfg["test"],
        random_state=split_cfg["random_state"],
    )

    print(f"Split: train={len(train_list)}, val={len(val_list)}, test={len(test_list)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    copy_to_split_structure((train_list, val_list, test_list), output_dir)

    class_names = sorted(class_to_images.keys())
    with open(output_dir / "classes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(class_names))

    print(f"\nDone! Data saved to: {output_dir}")
    print(f"  train/  val/  test/  classes.txt")


if __name__ == "__main__":
    prepare_dataset()

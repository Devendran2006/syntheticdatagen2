
"""
X-Ray Dataset Loader & Preprocessing
====================================

Purpose:
    - Load the chest X-ray dataset safely
    - Use only the intended chest_xray directory
    - Keep train / val / test separated
    - Convert images to grayscale
    - Preserve aspect ratio while resizing
    - Pad images to a fixed square size
    - Normalize pixel values to [-1, 1]
    - Provide PyTorch Dataset and DataLoader utilities

Dataset structure:

data/
└── imaging/
    └── xray/
        └── chest_xray/
            ├── train/
            │   ├── NORMAL/
            │   └── PNEUMONIA/
            ├── val/
            │   ├── NORMAL/
            │   └── PNEUMONIA/
            └── test/
                ├── NORMAL/
                └── PNEUMONIA/

Important:
    The test set is never used for training.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "imaging"
    / "xray"
    / "chest_xray"
)

TRAIN_ROOT = DATA_ROOT / "train"
VAL_ROOT = DATA_ROOT / "val"
TEST_ROOT = DATA_ROOT / "test"


# ============================================================
# DATASET CONFIGURATION
# ============================================================

CLASSES = ["NORMAL", "PNEUMONIA"]

CLASS_TO_INDEX = {
    "NORMAL": 0,
    "PNEUMONIA": 1,
}

INDEX_TO_CLASS = {
    0: "NORMAL",
    1: "PNEUMONIA",
}

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

# CPU-friendly initial training resolution.
IMAGE_SIZE = 128


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset_structure() -> None:
    """
    Verify that the expected X-ray dataset structure exists.
    """

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"X-ray dataset root not found:\n{DATA_ROOT}"
        )

    for split_root in [TRAIN_ROOT, VAL_ROOT, TEST_ROOT]:
        if not split_root.exists():
            raise FileNotFoundError(
                f"Missing dataset split:\n{split_root}"
            )

        for class_name in CLASSES:
            class_path = split_root / class_name

            if not class_path.exists():
                raise FileNotFoundError(
                    f"Missing class directory:\n{class_path}"
                )


# ============================================================
# IMAGE DISCOVERY
# ============================================================

def get_image_files(root: Path) -> List[Path]:
    """
    Return supported image files from a directory recursively.
    """

    if not root.exists():
        return []

    files = [
        path
        for path in root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    ]

    return sorted(files)


def get_split_files(split: str) -> List[Tuple[Path, int]]:
    """
    Return image paths and class labels for a dataset split.

    Parameters
    ----------
    split:
        "train", "val", or "test"
    """

    split_map = {
        "train": TRAIN_ROOT,
        "val": VAL_ROOT,
        "test": TEST_ROOT,
    }

    if split not in split_map:
        raise ValueError(
            "split must be one of: train, val, test"
        )

    root = split_map[split]

    samples = []

    for class_name in CLASSES:
        class_path = root / class_name
        label = CLASS_TO_INDEX[class_name]

        for image_path in get_image_files(class_path):
            samples.append((image_path, label))

    return sorted(samples, key=lambda item: str(item[0]).lower())


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def resize_and_pad(image: Image.Image, size: int) -> Image.Image:
    """
    Convert image to grayscale and resize while preserving
    aspect ratio.

    Empty areas are padded with black pixels.
    """

    image = image.convert("L")

    width, height = image.size

    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions.")

    scale = min(size / width, size / height)

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new(
        "L",
        (size, size),
        color=0,
    )

    left = (size - new_width) // 2
    top = (size - new_height) // 2

    canvas.paste(image, (left, top))

    return canvas


# ============================================================
# TORCH TRANSFORMS
# ============================================================

class XRayTransform:
    """
    Preprocessing pipeline for X-ray generation.

    Output:
        Tensor shape = [1, IMAGE_SIZE, IMAGE_SIZE]
        Pixel range = [-1, 1]
    """

    def __init__(self, size: int = IMAGE_SIZE):
        self.size = size

    def __call__(self, image: Image.Image) -> torch.Tensor:

        image = resize_and_pad(image, self.size)

        array = np.asarray(
            image,
            dtype=np.float32,
        )

        array = array / 127.5 - 1.0

        tensor = torch.from_numpy(array)

        tensor = tensor.unsqueeze(0)

        return tensor


# ============================================================
# PYTORCH DATASET
# ============================================================

class XRayDataset(Dataset):
    """
    PyTorch dataset for chest X-ray images.

    Labels:
        0 = NORMAL
        1 = PNEUMONIA
    """

    def __init__(
        self,
        split: str = "train",
        image_size: int = IMAGE_SIZE,
    ):
        validate_dataset_structure()

        self.split = split
        self.image_size = image_size

        self.samples = get_split_files(split)

        if not self.samples:
            raise RuntimeError(
                f"No X-ray images found for split: {split}"
            )

        self.transform = XRayTransform(
            size=image_size
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, int]:

        image_path, label = self.samples[index]

        try:
            with Image.open(image_path) as image:
                image = image.copy()

            tensor = self.transform(image)

        except Exception as exc:
            raise RuntimeError(
                f"Failed to load X-ray image:\n"
                f"{image_path}\n"
                f"Error: {exc}"
            ) from exc

        return tensor, label


# ============================================================
# DATASET SUMMARY
# ============================================================

def get_dataset_summary() -> Dict[str, Dict[str, int]]:
    """
    Return image counts for every split and class.
    """

    validate_dataset_structure()

    summary = {}

    for split in ["train", "val", "test"]:

        samples = get_split_files(split)

        counts = {
            class_name: 0
            for class_name in CLASSES
        }

        for _, label in samples:
            counts[INDEX_TO_CLASS[label]] += 1

        counts["TOTAL"] = len(samples)

        summary[split] = counts

    return summary


# ============================================================
# DATALOADER
# ============================================================

def create_dataloader(
    split: str = "train",
    batch_size: int = 32,
    image_size: int = IMAGE_SIZE,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a PyTorch DataLoader.

    CPU-friendly defaults are intentionally used.
    """

    dataset = XRayDataset(
        split=split,
        image_size=image_size,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


# ============================================================
# SANITY CHECK
# ============================================================

def run_sanity_check() -> None:
    """
    Load one batch from the training dataset and verify
    tensor shape, labels and pixel range.
    """

    print("=" * 70)
    print("X-RAY DATASET SANITY CHECK")
    print("=" * 70)

    print(f"Dataset root : {DATA_ROOT}")
    print(f"Image size   : {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Classes      : {CLASSES}")

    print("\nDataset summary:")

    summary = get_dataset_summary()

    for split, counts in summary.items():
        print(
            f"{split:>5} : "
            f"NORMAL={counts['NORMAL']:,} | "
            f"PNEUMONIA={counts['PNEUMONIA']:,} | "
            f"TOTAL={counts['TOTAL']:,}"
        )

    print("\nLoading training batch...")

    loader = create_dataloader(
        split="train",
        batch_size=8,
        image_size=IMAGE_SIZE,
        shuffle=False,
    )

    images, labels = next(iter(loader))

    print(f"Batch shape  : {tuple(images.shape)}")
    print(f"Labels shape : {tuple(labels.shape)}")
    print(
        f"Pixel range  : "
        f"{images.min().item():.4f} to "
        f"{images.max().item():.4f}"
    )

    print("\nClass labels in first batch:")

    for label in labels.tolist():
        print(
            f"{label} -> {INDEX_TO_CLASS[int(label)]}"
        )

    expected_shape = (
        images.shape[0],
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    if tuple(images.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected tensor shape: {tuple(images.shape)}"
        )

    if images.min() < -1.0001 or images.max() > 1.0001:
        raise RuntimeError(
            "Pixel normalization is outside [-1, 1]."
        )

    print("\n" + "=" * 70)
    print("SANITY CHECK PASSED")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_sanity_check()

"""
MRI ML Validation V5
====================

Uses the existing V4 synthetic MRI images. Does NOT retrain the GAN.

Experiments:
1. Real Training -> Real Testing
2. V4 Synthetic -> Real Testing
3. Real + V4 Synthetic -> Real Testing

V5 improvements:
- train/validation split using training data only
- moderate training augmentation
- stronger CNN
- dropout + AdamW weight decay
- ReduceLROnPlateau scheduler
- best validation checkpoint
- untouched real test set for final evaluation
- fixed seed and consistent grayscale preprocessing

NOTE:
These metrics measure ML/engineering utility only. They do not establish
medical realism, clinical validity, or diagnostic safety.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = PROJECT_ROOT / "data" / "imaging" / "MRI" / "Training_Preprocessed"
TEST_DIR = PROJECT_ROOT / "data" / "imaging" / "MRI" / "Testing"
SYNTHETIC_DIR = (
    PROJECT_ROOT / "outputs" / "mri" / "v4" / "evaluation" / "synthetic_samples"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mri" / "v5" / "ml_validation"
RESULTS_DIR = OUTPUT_DIR / "results"
MODELS_DIR = OUTPUT_DIR / "models"
CM_DIR = OUTPUT_DIR / "confusion_matrices"
CURVES_DIR = OUTPUT_DIR / "training_curves"

EXPECTED_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

IMAGE_SIZE = 128
NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
DROPOUT = 0.35
VAL_RATIO = 0.20
RANDOM_SEED = 42
NUM_WORKERS = 0
PIN_MEMORY = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cpu":
    CPU_THREADS = min(4, os.cpu_count() or 1)
    torch.set_num_threads(CPU_THREADS)
else:
    CPU_THREADS = None


# ============================================================
# REPRODUCIBILITY / SETUP
# ============================================================

def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_output_dirs() -> None:
    for directory in (
        OUTPUT_DIR,
        RESULTS_DIR,
        MODELS_DIR,
        CM_DIR,
        CURVES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def verify_directory(path: Path, name: str) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(
            f"\n{name} directory was not found:\n{path}\n"
        )


# ============================================================
# TRANSFORMS
# ============================================================

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.ToTensor(),
    ]
)

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(
            8,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0,
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.04, 0.04),
            scale=(0.95, 1.05),
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0,
        ),
        transforms.ToTensor(),
    ]
)


# ============================================================
# DATASETS
# ============================================================

class TransformSubset(Dataset):
    """Subset of an ImageFolder with its own transform."""

    def __init__(
        self,
        root: Path,
        indices: List[int],
        transform,
        classes: List[str],
    ):
        base = datasets.ImageFolder(str(root), transform=None)

        self.samples = [base.samples[i] for i in indices]
        self.targets = [label for _, label in self.samples]
        self.transform = transform
        self.classes = classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        with Image.open(path) as image:
            image = image.convert("L")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class ExplicitSamplesDataset(Dataset):
    """
    samples = [(path, class_index, source), ...]
    """

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, _source = self.samples[index]

        with Image.open(path) as image:
            image = image.convert("L")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def make_stratified_split(
    root: Path,
    val_ratio: float = VAL_RATIO,
    seed: int = RANDOM_SEED,
):
    base = datasets.ImageFolder(str(root), transform=None)

    if base.classes != EXPECTED_CLASSES:
        raise ValueError(
            f"Unexpected class order in {root}.\n"
            f"Expected: {EXPECTED_CLASSES}\n"
            f"Found:    {base.classes}"
        )

    rng = random.Random(seed)

    by_class = {i: [] for i in range(NUM_CLASSES)}

    for index, (_, label) in enumerate(base.samples):
        by_class[label].append(index)

    train_indices = []
    val_indices = []

    for class_index in range(NUM_CLASSES):
        indices = by_class[class_index][:]
        rng.shuffle(indices)

        val_count = max(1, int(len(indices) * val_ratio))

        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return (
        TransformSubset(
            root,
            train_indices,
            TRAIN_TRANSFORM,
            base.classes,
        ),
        TransformSubset(
            root,
            val_indices,
            EVAL_TRANSFORM,
            base.classes,
        ),
    )


def build_test_dataset():
    dataset = datasets.ImageFolder(
        str(TEST_DIR),
        transform=EVAL_TRANSFORM,
    )

    if dataset.classes != EXPECTED_CLASSES:
        raise ValueError(
            f"Unexpected test class order: {dataset.classes}"
        )

    return dataset


def build_combined_train_val():
    real_base = datasets.ImageFolder(str(TRAIN_DIR), transform=None)
    synth_base = datasets.ImageFolder(str(SYNTHETIC_DIR), transform=None)

    if real_base.classes != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected real classes: {real_base.classes}")

    if synth_base.classes != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected synthetic classes: {synth_base.classes}")

    rng = random.Random(RANDOM_SEED)

    train_samples = []
    val_samples = []

    for class_index in range(NUM_CLASSES):
        real_files = [
            path
            for path, label in real_base.samples
            if label == class_index
        ]

        synth_files = [
            path
            for path, label in synth_base.samples
            if label == class_index
        ]

        samples = (
            [(path, class_index, "real") for path in real_files]
            + [(path, class_index, "synthetic") for path in synth_files]
        )

        rng.shuffle(samples)

        val_count = max(1, int(len(samples) * VAL_RATIO))

        val_samples.extend(samples[:val_count])
        train_samples.extend(samples[val_count:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    return (
        ExplicitSamplesDataset(train_samples, TRAIN_TRANSFORM),
        ExplicitSamplesDataset(val_samples, EVAL_TRANSFORM),
    )


# ============================================================
# MODEL
# ============================================================

class MRIClassifierV5(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 384, 3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ============================================================
# LOADERS
# ============================================================

def make_loader(dataset, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )


# ============================================================
# TRAIN / VALIDATION
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return (
        running_loss / max(total, 1),
        correct / max(total, 1),
    )


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return (
        running_loss / max(total, 1),
        correct / max(total, 1),
    )


def save_curves(history: Dict, name: str):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{name} - Loss")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(CURVES_DIR / f"{name}_loss.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_accuracy"], label="Train Accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{name} - Accuracy")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(CURVES_DIR / f"{name}_accuracy.png", dpi=160)
    plt.close()


def train_classifier(name, train_dataset, val_dataset):
    print("\n" + "-" * 72)
    print(f"TRAINING: {name}")
    print("-" * 72)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = make_loader(train_dataset, shuffle=True)
    val_loader = make_loader(val_dataset, shuffle=False)

    model = MRIClassifierV5().to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }

    best_val_accuracy = -1.0
    best_epoch = 0
    best_state = None

    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
        )

        scheduler.step(val_acc)

        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["learning_rate"].append(lr)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        minutes = (time.time() - epoch_start) / 60.0

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc * 100:.2f}% | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc * 100:.2f}% | "
            f"LR {lr:.6f} | "
            f"{minutes:.2f} min"
        )

    total_minutes = (time.time() - start) / 60.0

    if best_state is None:
        raise RuntimeError(f"No best checkpoint created for {name}")

    model.load_state_dict(best_state)

    model_path = MODELS_DIR / f"{name}_best.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_accuracy,
            "classes": EXPECTED_CLASSES,
            "image_size": IMAGE_SIZE,
            "experiment": name,
        },
        model_path,
    )

    save_curves(history, name)

    with open(
        RESULTS_DIR / f"{name}_history.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "history": history,
                "best_epoch": best_epoch,
                "best_validation_accuracy": best_val_accuracy,
                "training_minutes": total_minutes,
            },
            f,
            indent=2,
        )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy * 100:.2f}% "
        f"(epoch {best_epoch})"
    )
    print(f"Saved: {model_path}")
    print(f"Training time: {total_minutes:.2f} minutes")

    return model


# ============================================================
# FINAL TEST
# ============================================================

@torch.no_grad()
def evaluate_real_test(model, test_dataset, name):
    print("\n" + "-" * 72)
    print(f"FINAL REAL TEST: {name}")
    print("-" * 72)

    loader = make_loader(test_dataset, shuffle=False)

    model.eval()

    y_true = []
    y_pred = []

    for images, labels in loader:
        images = images.to(DEVICE)

        logits = model(images)
        predictions = logits.argmax(1).cpu().numpy()

        y_pred.extend(predictions.tolist())
        y_true.extend(labels.numpy().tolist())

    accuracy = accuracy_score(y_true, y_pred)

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=EXPECTED_CLASSES,
        zero_division=0,
        output_dict=True,
    )

    print(f"Test samples      : {len(y_true)}")
    print(f"Accuracy           : {accuracy * 100:.2f}%")
    print(f"Macro Precision    : {macro_precision * 100:.2f}%")
    print(f"Macro Recall       : {macro_recall * 100:.2f}%")
    print(f"Macro F1           : {macro_f1 * 100:.2f}%")
    print(f"Weighted Precision : {weighted_precision * 100:.2f}%")
    print(f"Weighted Recall    : {weighted_recall * 100:.2f}%")
    print(f"Weighted F1        : {weighted_f1 * 100:.2f}%")

    print("\nPer-class:")
    for cls in EXPECTED_CLASSES:
        row = report[cls]
        print(
            f"  {cls:<12} "
            f"Precision {row['precision'] * 100:6.2f}% | "
            f"Recall {row['recall'] * 100:6.2f}% | "
            f"F1 {row['f1-score'] * 100:6.2f}%"
        )

    save_confusion_matrix(cm, name)

    result = {
        "experiment": name,
        "test_samples": len(y_true),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    with open(
        RESULTS_DIR / f"{name}_test_results.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f, indent=2)

    return result


def save_confusion_matrix(cm: np.ndarray, name: str):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm)

    plt.xticks(
        range(NUM_CLASSES),
        EXPECTED_CLASSES,
        rotation=30,
        ha="right",
    )
    plt.yticks(range(NUM_CLASSES), EXPECTED_CLASSES)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{name} - Confusion Matrix")

    for row in range(NUM_CLASSES):
        for col in range(NUM_CLASSES):
            plt.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
            )

    plt.colorbar()
    plt.tight_layout()

    plt.savefig(
        CM_DIR / f"{name}_confusion_matrix.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()


# ============================================================
# DATA SUMMARY
# ============================================================

def count_images(root: Path):
    counts = {}

    for cls in EXPECTED_CLASSES:
        folder = root / cls

        if not folder.exists():
            counts[cls] = 0
            continue

        counts[cls] = sum(
            1
            for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower()
            in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        )

    return counts


def print_dataset_summary():
    print("\n" + "=" * 72)
    print("DATASET SUMMARY")
    print("=" * 72)

    for name, path in (
        ("REAL TRAINING", TRAIN_DIR),
        ("REAL TESTING", TEST_DIR),
        ("V4 SYNTHETIC", SYNTHETIC_DIR),
    ):
        counts = count_images(path)

        print(f"\n{name}")
        print(f"Path: {path}")

        for cls in EXPECTED_CLASSES:
            print(f"  {cls:<12}: {counts[cls]:>5}")

        print(f"  {'TOTAL':<12}: {sum(counts.values()):>5}")


# ============================================================
# SUMMARY
# ============================================================

def retention(value, baseline):
    if baseline <= 0:
        return 0.0
    return (value / baseline) * 100.0


def save_summary(real, synthetic, combined):
    real_acc = real["accuracy"]
    synth_acc = synthetic["accuracy"]
    combined_acc = combined["accuracy"]

    synth_retention = retention(synth_acc, real_acc)
    combined_retention = retention(combined_acc, real_acc)

    synth_delta = (synth_acc - real_acc) * 100.0
    combined_delta = (combined_acc - real_acc) * 100.0

    if synth_retention >= 80:
        utility_status = "STRONG"
    elif synth_retention >= 60:
        utility_status = "MODERATE"
    else:
        utility_status = "LOW"

    if combined_delta > 2:
        augmentation_status = "POSITIVE"
    elif combined_delta >= -2:
        augmentation_status = "NEUTRAL"
    else:
        augmentation_status = "NEGATIVE"

    summary = {
        "real_test_accuracy": real_acc,
        "synthetic_test_accuracy": synth_acc,
        "combined_test_accuracy": combined_acc,
        "synthetic_utility_retention_percent": synth_retention,
        "combined_utility_retention_percent": combined_retention,
        "synthetic_accuracy_delta_percentage_points": synth_delta,
        "combined_accuracy_delta_percentage_points": combined_delta,
        "synthetic_utility_status": utility_status,
        "combined_augmentation_status": augmentation_status,
        "real_macro_f1": real["macro_f1"],
        "synthetic_macro_f1": synthetic["macro_f1"],
        "combined_macro_f1": combined["macro_f1"],
        "real_weighted_f1": real["weighted_f1"],
        "synthetic_weighted_f1": synthetic["weighted_f1"],
        "combined_weighted_f1": combined["weighted_f1"],
    }

    with open(
        RESULTS_DIR / "v5_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    rows = [
        {
            "Experiment": "Real -> Real Test",
            "Accuracy": real_acc * 100,
            "Macro F1": real["macro_f1"] * 100,
            "Weighted F1": real["weighted_f1"] * 100,
            "Utility Retention": 100.0,
        },
        {
            "Experiment": "V4 Synthetic -> Real Test",
            "Accuracy": synth_acc * 100,
            "Macro F1": synthetic["macro_f1"] * 100,
            "Weighted F1": synthetic["weighted_f1"] * 100,
            "Utility Retention": synth_retention,
        },
        {
            "Experiment": "Real + V4 Synthetic -> Real Test",
            "Accuracy": combined_acc * 100,
            "Macro F1": combined["macro_f1"] * 100,
            "Weighted F1": combined["weighted_f1"] * 100,
            "Utility Retention": combined_retention,
        },
    ]

    with open(
        RESULTS_DIR / "v5_summary.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    experiments = [
        "Real -> Real",
        "V4 Synthetic -> Real",
        "Real + V4 -> Real",
    ]

    accuracies = [
        real_acc * 100,
        synth_acc * 100,
        combined_acc * 100,
    ]

    f1_scores = [
        real["macro_f1"] * 100,
        synthetic["macro_f1"] * 100,
        combined["macro_f1"] * 100,
    ]

    x = np.arange(3)
    width = 0.35

    plt.figure(figsize=(11, 6))
    plt.bar(x - width / 2, accuracies, width, label="Accuracy")
    plt.bar(x + width / 2, f1_scores, width, label="Macro F1")

    plt.xticks(x, experiments, rotation=10)
    plt.ylabel("Score (%)")
    plt.title("MRI ML Validation V5")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis="y", alpha=0.2)

    for i, value in enumerate(accuracies):
        plt.text(i - width / 2, value + 1, f"{value:.1f}%", ha="center")

    for i, value in enumerate(f1_scores):
        plt.text(i + width / 2, value + 1, f"{value:.1f}%", ha="center")

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "v5_comparison.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()

    return summary


def print_final_summary(summary):
    print("\n" + "=" * 72)
    print("MRI ML VALIDATION V5 - FINAL SUMMARY")
    print("=" * 72)

    print(
        f"\nReal -> Real Test Accuracy    : "
        f"{summary['real_test_accuracy'] * 100:.2f}%"
    )
    print(
        f"V4 Synthetic -> Real Accuracy : "
        f"{summary['synthetic_test_accuracy'] * 100:.2f}%"
    )
    print(
        f"Real + V4 -> Real Accuracy    : "
        f"{summary['combined_test_accuracy'] * 100:.2f}%"
    )

    print(
        f"\nSynthetic Utility Retention  : "
        f"{summary['synthetic_utility_retention_percent']:.2f}%"
    )
    print(
        f"Combined Utility Retention   : "
        f"{summary['combined_utility_retention_percent']:.2f}%"
    )

    print(
        f"\nSynthetic Accuracy Delta     : "
        f"{summary['synthetic_accuracy_delta_percentage_points']:+.2f} pp"
    )
    print(
        f"Combined Accuracy Delta      : "
        f"{summary['combined_accuracy_delta_percentage_points']:+.2f} pp"
    )

    print(
        f"\nSynthetic Utility Status     : "
        f"{summary['synthetic_utility_status']}"
    )
    print(
        f"Combined Augmentation Status : "
        f"{summary['combined_augmentation_status']}"
    )

    print("\nMacro F1:")
    print(f"  Real          : {summary['real_macro_f1'] * 100:.2f}%")
    print(f"  V4 Synthetic  : {summary['synthetic_macro_f1'] * 100:.2f}%")
    print(f"  Real + V4     : {summary['combined_macro_f1'] * 100:.2f}%")

    print("\nWeighted F1:")
    print(f"  Real          : {summary['real_weighted_f1'] * 100:.2f}%")
    print(f"  V4 Synthetic  : {summary['synthetic_weighted_f1'] * 100:.2f}%")
    print(f"  Real + V4     : {summary['combined_weighted_f1'] * 100:.2f}%")

    print("\nIMPORTANT:")
    print(
        "These results measure ML utility only; they do not prove "
        "medical realism or clinical validity."
    )

    print(f"\nAll V5 outputs: {OUTPUT_DIR}")


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed()
    prepare_output_dirs()

    print("=" * 72)
    print("MRI ML VALIDATION V5")
    print("=" * 72)
    print(f"Device        : {DEVICE}")
    print(f"Image size    : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"Epochs        : {EPOCHS}")
    print(f"Learning rate : {LEARNING_RATE}")
    print(f"Weight decay  : {WEIGHT_DECAY}")
    print(f"Validation    : {VAL_RATIO * 100:.0f}%")
    print(f"Seed          : {RANDOM_SEED}")

    if CPU_THREADS is not None:
        print(f"CPU threads   : {CPU_THREADS}")

    verify_directory(TRAIN_DIR, "REAL TRAINING")
    verify_directory(TEST_DIR, "REAL TESTING")
    verify_directory(SYNTHETIC_DIR, "V4 SYNTHETIC")

    print_dataset_summary()

    # 1. Real -> Real
    real_train, real_val = make_stratified_split(TRAIN_DIR)
    real_test = build_test_dataset()

    real_model = train_classifier(
        "experiment_1_real_to_real",
        real_train,
        real_val,
    )

    real_result = evaluate_real_test(
        real_model,
        real_test,
        "experiment_1_real_to_real",
    )

    del real_model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # 2. V4 Synthetic -> Real
    synthetic_train, synthetic_val = make_stratified_split(SYNTHETIC_DIR)
    synthetic_test = build_test_dataset()

    synthetic_model = train_classifier(
        "experiment_2_v4_synthetic_to_real",
        synthetic_train,
        synthetic_val,
    )

    synthetic_result = evaluate_real_test(
        synthetic_model,
        synthetic_test,
        "experiment_2_v4_synthetic_to_real",
    )

    del synthetic_model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # 3. Real + V4 -> Real
    combined_train, combined_val = build_combined_train_val()
    combined_test = build_test_dataset()

    combined_model = train_classifier(
        "experiment_3_real_plus_v4_to_real",
        combined_train,
        combined_val,
    )

    combined_result = evaluate_real_test(
        combined_model,
        combined_test,
        "experiment_3_real_plus_v4_to_real",
    )

    del combined_model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    summary = save_summary(
        real_result,
        synthetic_result,
        combined_result,
    )

    print_final_summary(summary)

    print("\n" + "=" * 72)
    print("V5 VALIDATION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()

"""
MRI ML VALIDATION - TRAINING + SYNTHETIC DATA
==============================================

Purpose:
    Evaluate how useful the generated synthetic MRI training data is
    for downstream image classification.

FIVE EXPERIMENTS
----------------

1. Real Only
   Training:
       100% Real Training
   Testing:
       100% Real Testing

2. Synthetic Only
   Training:
       100% Synthetic Training
   Testing:
       100% Real Testing

3. Real + Synthetic
   Training:
       100% Real Training + 100% Synthetic Training
   Testing:
       100% Real Testing

4. Real 75% + Synthetic 25%
   Training:
       75% Real Training + 25% Synthetic Training
   Testing:
       100% Real Testing

5. Real 50% + Synthetic 50%
   Training:
       50% Real Training + 50% Synthetic Training
   Testing:
       100% Real Testing

IMPORTANT
---------
- Real Testing data is NEVER used for training.
- Real Testing data is used only for final evaluation.
- All images are converted to GRAYSCALE.
- All images are resized to 64 x 64.
- The same Real Testing dataset is used for every experiment.

INPUT
-----
Real Training:
    data/imaging/MRI/Training

Synthetic Training:
    outputs/mri/training_synthetic/synthetic

Real Testing:
    data/imaging/MRI/Testing

OUTPUT
------
outputs/mri/training_synthetic/ml_validation/

    ml_experiment_results.csv
    ml_experiment_summary.csv
    ml_utility_retention.csv

    real_only_confusion_matrix.png
    synthetic_only_confusion_matrix.png
    real_plus_synthetic_confusion_matrix.png
    real75_synthetic25_confusion_matrix.png
    real50_synthetic50_confusion_matrix.png

    ml_accuracy_comparison.png
    ml_precision_comparison.png
    ml_recall_comparison.png
    ml_f1_comparison.png
    ml_all_metrics_comparison.png
    ml_utility_retention.png
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

REAL_TRAIN_DIR = (
    BASE_DIR
    / "data"
    / "imaging"
    / "MRI"
    / "Training"
)

SYNTHETIC_TRAIN_DIR = (
    BASE_DIR
    / "outputs"
    / "mri"
    / "training_synthetic"
    / "synthetic"
)

REAL_TEST_DIR = (
    BASE_DIR
    / "data"
    / "imaging"
    / "MRI"
    / "Testing"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "mri"
    / "training_synthetic"
    / "ml_validation"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

IMAGE_SIZE = 64

NUM_CLASSES = 4

BATCH_SIZE = 32

EPOCHS = 5

LEARNING_RATE = 0.0003

NUM_WORKERS = 0

SEED = 42


# ============================================================
# DATA SETTINGS
# ============================================================

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

REAL_75_PERCENT = 0.75

REAL_50_PERCENT = 0.50

SYNTHETIC_25_PERCENT = 0.25

SYNTHETIC_50_PERCENT = 0.50


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CPU OPTIMIZATION
# ============================================================

if DEVICE.type == "cpu":

    try:

        cpu_threads = min(
            2,
            os.cpu_count() or 2
        )

        torch.set_num_threads(
            cpu_threads
        )

    except Exception:

        cpu_threads = 2

else:

    cpu_threads = torch.get_num_threads()


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print()

print("=" * 70)

print(
    "       MRI ML VALIDATION - "
    "FIVE TRAINING EXPERIMENTS"
)

print("=" * 70)

print()

print(
    f"Device        : {DEVICE}"
)

print(
    f"CPU threads   : {cpu_threads}"
)

print(
    f"Image size    : {IMAGE_SIZE} x {IMAGE_SIZE}"
)

print(
    f"Batch size    : {BATCH_SIZE}"
)

print(
    f"Epochs        : {EPOCHS}"
)

print(
    f"Learning rate : {LEARNING_RATE}"
)

print()

print(
    "IMPORTANT:"
)

print(
    "Real Testing images are NEVER used for training."
)

print(
    "The same Real Testing dataset is used for all experiments."
)

print(
    "All MRI images are converted to grayscale."
)

print()


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# GET IMAGE FILES
# ============================================================

def get_image_files(
    directory,
    class_name
):

    class_dir = (
        Path(directory)
        / class_name
    )

    if not class_dir.exists():

        return []

    files = []

    for file_path in class_dir.iterdir():

        if (
            file_path.is_file()
            and file_path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            files.append(
                file_path
            )

    return sorted(files)


# ============================================================
# DATASET RECORD CREATION
# ============================================================

def collect_dataset(
    directory,
    dataset_name
):

    directory = Path(directory)

    if not directory.exists():

        raise FileNotFoundError(
            f"\nDataset directory not found:\n{directory}"
        )

    records = []

    print()

    print("=" * 70)

    print(
        f"             {dataset_name}"
    )

    print("=" * 70)

    print()

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        files = get_image_files(
            directory,
            class_name
        )

        valid_files = []

        for file_path in files:

            try:

                # Verify image can actually be opened.

                with Image.open(
                    file_path
                ) as image:

                    image.verify()

                valid_files.append(
                    file_path
                )

            except Exception:

                pass

        print(
            f"{class_name:<15}: "
            f"{len(valid_files):>5} images"
        )

        for file_path in valid_files:

            records.append(
                (
                    file_path,
                    class_index
                )
            )

    print()

    print(
        f"Total images: {len(records)}"
    )

    print()

    return records


# ============================================================
# SELECT CLASS-BALANCED SUBSET
# ============================================================

def select_balanced_subset(
    records,
    fraction,
    dataset_label
):

    selected = []

    print()

    print(
        f"Selecting {fraction * 100:.0f}% "
        f"{dataset_label}..."
    )

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        class_records = [
            record
            for record in records
            if record[1] == class_index
        ]

        if len(class_records) == 0:

            continue

        count = max(
            1,
            int(
                len(class_records)
                * fraction
            )
        )

        rng = random.Random(
            SEED + class_index
        )

        selected_class = rng.sample(
            class_records,
            min(
                count,
                len(class_records)
            )
        )

        selected.extend(
            selected_class
        )

        print(
            f"{class_name:<15}: "
            f"{len(selected_class):>5} selected"
        )

    return selected


# ============================================================
# MRI DATASET
# ============================================================

class MRIDataset(Dataset):

    def __init__(
        self,
        records,
        augment=False
    ):

        self.records = records

        # ----------------------------------------------------
        # IMPORTANT:
        # Grayscale conversion guarantees 1 channel.
        # This fixes the previous:
        #
        # expected input to have 1 channels,
        # but got 3 channels
        # ----------------------------------------------------

        if augment:

            self.transform = transforms.Compose([

                transforms.Grayscale(
                    num_output_channels=1
                ),

                transforms.Resize(
                    (
                        IMAGE_SIZE,
                        IMAGE_SIZE
                    )
                ),

                transforms.RandomHorizontalFlip(
                    p=0.5
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    (0.5,),
                    (0.5,)
                ),
            ])

        else:

            self.transform = transforms.Compose([

                transforms.Grayscale(
                    num_output_channels=1
                ),

                transforms.Resize(
                    (
                        IMAGE_SIZE,
                        IMAGE_SIZE
                    )
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    (0.5,),
                    (0.5,)
                ),
            ])


    def __len__(self):

        return len(self.records)


    def __getitem__(
        self,
        index
    ):

        image_path, label = (
            self.records[index]
        )

        try:

            image = Image.open(
                image_path
            ).convert("L")

            image = self.transform(
                image
            )

            return (
                image,
                label
            )

        except Exception as error:

            print(
                f"Warning: failed to load "
                f"{image_path}: {error}"
            )

            # Return a valid black image
            # instead of crashing the entire run.

            image = torch.zeros(
                (
                    1,
                    IMAGE_SIZE,
                    IMAGE_SIZE
                ),
                dtype=torch.float32
            )

            return (
                image,
                label
            )


# ============================================================
# CNN MODEL
# ============================================================

class MRIClassifier(nn.Module):

    def __init__(
        self,
        num_classes=NUM_CLASSES
    ):

        super().__init__()

        self.features = nn.Sequential(

            # 1 -> 32

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),


            # 32 -> 64

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),


            # 64 -> 128

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),


            # 128 -> 256

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.3),

            nn.Linear(
                256,
                num_classes
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# CREATE DATA LOADER
# ============================================================

def create_loader(
    records,
    shuffle=True
):

    dataset = MRIDataset(
        records,
        augment=True
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )

    return loader


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    train_records,
    experiment_name
):

    print()

    print(
        "=" * 70
    )

    print(
        f"TRAINING: {experiment_name}"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Training images: "
        f"{len(train_records)}"
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if len(train_records) == 0:

        raise ValueError(
            f"No training images available "
            f"for experiment: {experiment_name}"
        )

    train_loader = create_loader(
        train_records,
        shuffle=True
    )

    model = MRIClassifier().to(
        DEVICE
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    history = []

    for epoch in range(
        EPOCHS
    ):

        model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        for images, labels in train_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad()

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        epoch_loss = (
            running_loss / total
        )

        epoch_accuracy = (
            correct / total
        )

        history.append({

            "Epoch":
                epoch + 1,

            "Loss":
                epoch_loss,

            "Accuracy":
                epoch_accuracy
        })

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Accuracy: "
            f"{epoch_accuracy * 100:.2f}%"
        )

    print()

    return model, history


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    test_records,
    experiment_name
):

    print()

    print(
        f"Evaluating {experiment_name}..."
    )

    test_dataset = MRIDataset(
        test_records,
        augment=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )

    model.eval()

    all_labels = []

    all_predictions = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(
                images
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
                .cpu()
                .numpy()
            )

            all_predictions.extend(
                predictions.tolist()
            )

            all_labels.extend(
                labels.numpy().tolist()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(
            range(NUM_CLASSES)
        )
    )

    print()

    print(
        f"{experiment_name} RESULTS"
    )

    print(
        "-" * 50
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print()

    print(
        "Classification Report"
    )

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    return {

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1_Score":
            f1,

        "Confusion_Matrix":
            cm
    }


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    cm,
    experiment_name,
    filename
):

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    image = ax.imshow(
        cm,
        interpolation="nearest"
    )

    ax.set_title(
        f"{experiment_name}\n"
        "Confusion Matrix",
        fontweight="bold"
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "True Class"
    )

    ax.set_xticks(
        range(NUM_CLASSES)
    )

    ax.set_yticks(
        range(NUM_CLASSES)
    )

    ax.set_xticklabels(
        CLASS_NAMES,
        rotation=30,
        ha="right"
    )

    ax.set_yticklabels(
        CLASS_NAMES
    )

    # --------------------------------------------------------
    # Add values inside cells
    # --------------------------------------------------------

    threshold = (
        cm.max() / 2
        if cm.size
        else 0
    )

    for i in range(
        NUM_CLASSES
    ):

        for j in range(
            NUM_CLASSES
        ):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[i, j] > threshold
                    else "black"
                ),
                fontweight="bold"
            )

    fig.colorbar(
        image,
        ax=ax
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / filename
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Confusion matrix saved:\n"
        f"{output_path}"
    )


# ============================================================
# SAVE METRIC GRAPH
# ============================================================

def save_metric_graph(
    results_df,
    metric,
    filename,
    title,
    ylabel
):

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    bars = ax.bar(
        results_df["Experiment"],
        results_df[metric] * 100
    )

    ax.set_title(
        title,
        fontweight="bold"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_xlabel(
        "Experiment"
    )

    ax.set_ylim(
        0,
        100
    )

    ax.tick_params(
        axis="x",
        rotation=25
    )

    # --------------------------------------------------------
    # Data labels
    # --------------------------------------------------------

    for bar, value in zip(
        bars,
        results_df[metric] * 100
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold"
        )

    ax.grid(
        axis="y",
        alpha=0.2
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / filename
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Graph saved:\n"
        f"{output_path}"
    )


# ============================================================
# SAVE ALL METRICS GRAPH
# ============================================================

def save_all_metrics_graph(
    results_df
):

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score"
    ]

    x = np.arange(
        len(results_df)
    )

    width = 0.18

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    for index, metric in enumerate(
        metrics
    ):

        positions = (
            x
            + (
                index
                - 1.5
            ) * width
        )

        bars = ax.bar(
            positions,
            results_df[metric] * 100,
            width,
            label=metric
        )

        for bar, value in zip(
            bars,
            results_df[metric] * 100
        ):

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    ax.set_title(
        "MRI ML Performance Comparison",
        fontweight="bold"
    )

    ax.set_ylabel(
        "Score (%)"
    )

    ax.set_xlabel(
        "Experiment"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        results_df["Experiment"],
        rotation=25,
        ha="right"
    )

    ax.set_ylim(
        0,
        105
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.2
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "ml_all_metrics_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"All-metrics graph saved:\n"
        f"{output_path}"
    )


# ============================================================
# UTILITY RETENTION
# ============================================================

def calculate_utility_retention(
    results_df
):

    # --------------------------------------------------------
    # Baseline = Real Only
    # --------------------------------------------------------

    real_baseline = float(
        results_df.loc[
            results_df["Experiment"]
            == "Real Only",
            "F1_Score"
        ].iloc[0]
    )

    results_df = results_df.copy()

    results_df[
        "Utility_Retention"
    ] = (
        results_df["F1_Score"]
        / real_baseline
    )

    results_df[
        "Performance_Gap"
    ] = (
        results_df["F1_Score"]
        - real_baseline
    )

    return results_df


# ============================================================
# SAVE UTILITY GRAPH
# ============================================================

def save_utility_graph(
    results_df
):

    graph_df = results_df[
        results_df["Experiment"]
        != "Real Only"
    ].copy()

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    bars = ax.bar(
        graph_df["Experiment"],
        graph_df[
            "Utility_Retention"
        ] * 100
    )

    ax.set_title(
        "MRI Synthetic Data Utility Retention",
        fontweight="bold"
    )

    ax.set_ylabel(
        "Utility Retention (%)"
    )

    ax.set_xlabel(
        "Experiment"
    )

    ax.axhline(
        100,
        linestyle="--",
        linewidth=1
    )

    ax.set_ylim(
        0,
        max(
            110,
            graph_df[
                "Utility_Retention"
            ].max()
            * 100
            + 10
        )
    )

    ax.tick_params(
        axis="x",
        rotation=25
    )

    for bar, value in zip(
        bars,
        graph_df[
            "Utility_Retention"
        ] * 100
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold"
        )

    ax.grid(
        axis="y",
        alpha=0.2
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "ml_utility_retention.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Utility graph saved:\n"
        f"{output_path}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results
):

    results_df = pd.DataFrame(
        results
    )

    results_df = calculate_utility_retention(
        results_df
    )

    # --------------------------------------------------------
    # Main CSV
    # --------------------------------------------------------

    main_columns = [

        "Experiment",

        "Real_Training_Images",

        "Synthetic_Training_Images",

        "Total_Training_Images",

        "Real_Training_Percentage",

        "Synthetic_Training_Percentage",

        "Test_Images",

        "Accuracy",

        "Precision",

        "Recall",

        "F1_Score",

        "Utility_Retention",

        "Performance_Gap",
    ]

    results_df[
        main_columns
    ].to_csv(
        OUTPUT_DIR
        / "ml_experiment_results.csv",
        index=False
    )

    # --------------------------------------------------------
    # Compact summary
    # --------------------------------------------------------

    summary_df = results_df[
        [
            "Experiment",
            "Accuracy",
            "Precision",
            "Recall",
            "F1_Score",
            "Utility_Retention"
        ]
    ].copy()

    summary_df[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1_Score",
            "Utility_Retention"
        ]
    ] *= 100

    summary_df.to_csv(
        OUTPUT_DIR
        / "ml_experiment_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # Utility CSV
    # --------------------------------------------------------

    utility_df = results_df[
        [
            "Experiment",
            "F1_Score",
            "Utility_Retention",
            "Performance_Gap"
        ]
    ].copy()

    utility_df[
        [
            "F1_Score",
            "Utility_Retention"
        ]
    ] *= 100

    utility_df.to_csv(
        OUTPUT_DIR
        / "ml_utility_retention.csv",
        index=False
    )

    print()

    print(
        "CSV results saved:"
    )

    print(
        OUTPUT_DIR
        / "ml_experiment_results.csv"
    )

    print(
        OUTPUT_DIR
        / "ml_experiment_summary.csv"
    )

    print(
        OUTPUT_DIR
        / "ml_utility_retention.csv"
    )

    return results_df


# ============================================================
# CLEAN OLD OUTPUT GRAPHS
# ============================================================

def clean_old_outputs():

    # Only remove generated graph/CSV files.
    # Never touch real or synthetic datasets.

    if not OUTPUT_DIR.exists():

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        return

    for path in OUTPUT_DIR.iterdir():

        if path.is_file():

            if path.suffix.lower() in {
                ".csv",
                ".png"
            }:

                try:

                    path.unlink()

                except Exception:

                    pass


# ============================================================
# MAIN
# ============================================================

def main():

    clean_old_outputs()

    # ========================================================
    # LOAD REAL TRAINING
    # ========================================================

    print()

    print("=" * 70)

    print(
        "                 LOADING DATASETS"
    )

    print("=" * 70)

    print()

    print(
        "Loading REAL training data..."
    )

    print(
        f"Path: {REAL_TRAIN_DIR}"
    )

    real_train_records = collect_dataset(
        REAL_TRAIN_DIR,
        "REAL TRAINING DATA"
    )

    # ========================================================
    # LOAD SYNTHETIC TRAINING
    # ========================================================

    print(
        "Loading SYNTHETIC training data..."
    )

    print(
        f"Path: {SYNTHETIC_TRAIN_DIR}"
    )

    synthetic_train_records = collect_dataset(
        SYNTHETIC_TRAIN_DIR,
        "SYNTHETIC TRAINING DATA"
    )

    # ========================================================
    # LOAD REAL TEST
    # ========================================================

    print(
        "Loading REAL testing data..."
    )

    print(
        f"Path: {REAL_TEST_DIR}"
    )

    real_test_records = collect_dataset(
        REAL_TEST_DIR,
        "REAL TEST DATA"
    )

    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if len(real_train_records) == 0:

        raise RuntimeError(
            "Real Training dataset is empty."
        )

    if len(synthetic_train_records) == 0:

        raise RuntimeError(
            "Synthetic Training dataset is empty."
        )

    if len(real_test_records) == 0:

        raise RuntimeError(
            "Real Testing dataset is empty."
        )

    # ========================================================
    # SELECT SUBSETS
    # ========================================================

    print()

    print("=" * 70)

    print(
        "             CREATING EXPERIMENT DATA"
    )

    print("=" * 70)

    real_75_records = select_balanced_subset(
        real_train_records,
        REAL_75_PERCENT,
        "REAL training data"
    )

    real_50_records = select_balanced_subset(
        real_train_records,
        REAL_50_PERCENT,
        "REAL training data"
    )

    synthetic_25_records = select_balanced_subset(
        synthetic_train_records,
        SYNTHETIC_25_PERCENT,
        "SYNTHETIC training data"
    )

    synthetic_50_records = select_balanced_subset(
        synthetic_train_records,
        SYNTHETIC_50_PERCENT,
        "SYNTHETIC training data"
    )

    # ========================================================
    # EXPERIMENT DEFINITIONS
    # ========================================================

    experiments = [

        {
            "name":
                "Real Only",

            "real":
                real_train_records,

            "synthetic":
                [],

            "real_percentage":
                100,

            "synthetic_percentage":
                0
        },

        {
            "name":
                "Synthetic Only",

            "real":
                [],

            "synthetic":
                synthetic_train_records,

            "real_percentage":
                0,

            "synthetic_percentage":
                100
        },

        {
            "name":
                "Real + Synthetic",

            "real":
                real_train_records,

            "synthetic":
                synthetic_train_records,

            "real_percentage":
                100,

            "synthetic_percentage":
                100
        },

        {
            "name":
                "Real 75% + Synthetic 25%",

            "real":
                real_75_records,

            "synthetic":
                synthetic_25_records,

            "real_percentage":
                75,

            "synthetic_percentage":
                25
        },

        {
            "name":
                "Real 50% + Synthetic 50%",

            "real":
                real_50_records,

            "synthetic":
                synthetic_50_records,

            "real_percentage":
                50,

            "synthetic_percentage":
                50
        }
    ]

    # ========================================================
    # RUN EXPERIMENTS
    # ========================================================

    all_results = []

    for experiment_index, experiment in enumerate(
        experiments,
        start=1
    ):

        print()

        print(
            "\n"
            + "#" * 70
        )

        print(
            f"EXPERIMENT {experiment_index}/5: "
            f"{experiment['name']}"
        )

        print(
            "#" * 70
        )

        # ----------------------------------------------------
        # Combine training records
        # ----------------------------------------------------

        train_records = (
            experiment["real"]
            + experiment["synthetic"]
        )

        # ----------------------------------------------------
        # Shuffle combined dataset
        # ----------------------------------------------------

        rng = random.Random(
            SEED + experiment_index
        )

        train_records = list(
            train_records
        )

        rng.shuffle(
            train_records
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model, history = train_model(
            train_records,
            experiment["name"]
        )

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        metrics = evaluate_model(
            model,
            real_test_records,
            experiment["name"]
        )

        # ----------------------------------------------------
        # Confusion matrix filename
        # ----------------------------------------------------

        filename_map = {

            "Real Only":
                "real_only_confusion_matrix.png",

            "Synthetic Only":
                "synthetic_only_confusion_matrix.png",

            "Real + Synthetic":
                "real_plus_synthetic_confusion_matrix.png",

            "Real 75% + Synthetic 25%":
                "real75_synthetic25_confusion_matrix.png",

            "Real 50% + Synthetic 50%":
                "real50_synthetic50_confusion_matrix.png",
        }

        save_confusion_matrix(
            metrics[
                "Confusion_Matrix"
            ],
            experiment["name"],
            filename_map[
                experiment["name"]
            ]
        )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        all_results.append({

            "Experiment":
                experiment["name"],

            "Real_Training_Images":
                len(
                    experiment["real"]
                ),

            "Synthetic_Training_Images":
                len(
                    experiment["synthetic"]
                ),

            "Total_Training_Images":
                len(
                    train_records
                ),

            "Real_Training_Percentage":
                experiment[
                    "real_percentage"
                ],

            "Synthetic_Training_Percentage":
                experiment[
                    "synthetic_percentage"
                ],

            "Test_Images":
                len(
                    real_test_records
                ),

            "Accuracy":
                metrics[
                    "Accuracy"
                ],

            "Precision":
                metrics[
                    "Precision"
                ],

            "Recall":
                metrics[
                    "Recall"
                ],

            "F1_Score":
                metrics[
                    "F1_Score"
                ],
        })

        # ----------------------------------------------------
        # Free memory
        # ----------------------------------------------------

        del model

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "             SAVING FINAL RESULTS"
    )

    print(
        "=" * 70
    )

    results_df = save_results(
        all_results
    )

    # ========================================================
    # DISPLAY FINAL TABLE
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "             FIVE-EXPERIMENT SUMMARY"
    )

    print(
        "=" * 70
    )

    display_df = results_df[
        [
            "Experiment",
            "Real_Training_Images",
            "Synthetic_Training_Images",
            "Accuracy",
            "Precision",
            "Recall",
            "F1_Score",
            "Utility_Retention"
        ]
    ].copy()

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
        "Utility_Retention"
    ]:

        display_df[column] *= 100

    print()

    print(
        display_df.to_string(
            index=False,
            formatters={

                "Accuracy":
                    "{:.2f}%".format,

                "Precision":
                    "{:.2f}%".format,

                "Recall":
                    "{:.2f}%".format,

                "F1_Score":
                    "{:.2f}%".format,

                "Utility_Retention":
                    "{:.2f}%".format,
            }
        )
    )

    # ========================================================
    # GENERATE GRAPHS
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "             CREATING COMPARISON GRAPHS"
    )

    print(
        "=" * 70
    )

    print()

    save_metric_graph(
        results_df,
        "Accuracy",
        "ml_accuracy_comparison.png",
        "MRI ML Accuracy Comparison",
        "Accuracy (%)"
    )

    save_metric_graph(
        results_df,
        "Precision",
        "ml_precision_comparison.png",
        "MRI ML Precision Comparison",
        "Precision (%)"
    )

    save_metric_graph(
        results_df,
        "Recall",
        "ml_recall_comparison.png",
        "MRI ML Recall Comparison",
        "Recall (%)"
    )

    save_metric_graph(
        results_df,
        "F1_Score",
        "ml_f1_comparison.png",
        "MRI ML F1 Score Comparison",
        "F1 Score (%)"
    )

    save_all_metrics_graph(
        results_df
    )

    save_utility_graph(
        results_df
    )

    # ========================================================
    # BEST EXPERIMENT
    # ========================================================

    best_index = results_df[
        "F1_Score"
    ].idxmax()

    best_row = results_df.loc[
        best_index
    ]

    print()

    print(
        "=" * 70
    )

    print(
        "             BEST EXPERIMENT"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Best Experiment : "
        f"{best_row['Experiment']}"
    )

    print(
        f"Accuracy        : "
        f"{best_row['Accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision       : "
        f"{best_row['Precision'] * 100:.2f}%"
    )

    print(
        f"Recall          : "
        f"{best_row['Recall'] * 100:.2f}%"
    )

    print(
        f"F1 Score        : "
        f"{best_row['F1_Score'] * 100:.2f}%"
    )

    print()

    # ========================================================
    # SYNTHETIC UTILITY INTERPRETATION
    # ========================================================

    real_baseline = float(
        results_df.loc[
            results_df["Experiment"]
            == "Real Only",
            "F1_Score"
        ].iloc[0]
    )

    best_synthetic_index = results_df[
        results_df["Experiment"]
        != "Real Only"
    ][
        "F1_Score"
    ].idxmax()

    best_synthetic = results_df.loc[
        best_synthetic_index
    ]

    utility = (
        best_synthetic["F1_Score"]
        / real_baseline
    ) * 100

    print(
        "Best Synthetic-Assisted Result:"
    )

    print(
        f"  {best_synthetic['Experiment']}"
    )

    print(
        f"  Utility Retention: "
        f"{utility:.2f}%"
    )

    print()

    # ========================================================
    # FINAL FILE LIST
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "       MRI ML VALIDATION COMPLETED"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Results directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    print(
        "Generated files:"
    )

    generated_files = sorted(
        OUTPUT_DIR.iterdir()
    )

    for index, file_path in enumerate(
        generated_files,
        start=1
    ):

        if file_path.is_file():

            print(
                f"  {index}. "
                f"{file_path.name}"
            )

    print()

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
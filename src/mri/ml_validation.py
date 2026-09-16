
# src/mri/ml_validation.py
#
# FIVE-EXPERIMENT MRI SYNTHETIC DATA VALIDATION
#
# Experiments:
# 1. Real Only
# 2. Synthetic Only
# 3. Real + 25% Synthetic
# 4. Real + 50% Synthetic
# 5. Real + 100% Synthetic
#
# Every experiment is evaluated on the SAME REAL TEST DATA.
#
# Outputs:
#   outputs/mri/ml_validation/
#       mri_five_experiment_results.csv
#       mri_utility_retention.csv
#       real_only_confusion_matrix.png
#       synthetic_only_confusion_matrix.png
#       real_plus_25pct_confusion_matrix.png
#       real_plus_50pct_confusion_matrix.png
#       real_plus_100pct_confusion_matrix.png
#       five_experiment_comparison.png
#
# IMPORTANT:
# Real and synthetic images are converted to RGB.
# Model therefore uses 3 input channels.

import os
import random
import time
import numpy as np
import pandas as pd
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
    classification_report
)

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

REAL_TRAIN_DIR = "data/imaging/MRI/Training"
SYNTHETIC_DIR = "outputs/mri/synthetic"
REAL_TEST_DIR = "data/imaging/MRI/Testing"

OUTPUT_DIR = "outputs/mri/ml_validation"

IMAGE_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.0003

NUM_CLASSES = 4

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

SEED = 42

# CPU optimization
CPU_THREADS = min(4, os.cpu_count() or 2)

torch.set_num_threads(CPU_THREADS)

if hasattr(torch, "set_num_interop_threads"):
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


# ============================================================
# DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# PRINT HEADER
# ============================================================

print()
print("=" * 70)
print("             MRI FIVE-EXPERIMENT ML VALIDATION")
print("=" * 70)
print()
print(f"Device       : {DEVICE}")
print(f"CPU threads  : {CPU_THREADS}")
print(f"Image size   : {IMAGE_SIZE} x {IMAGE_SIZE}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Epochs       : {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print()
print("IMPORTANT:")
print("Every experiment is evaluated on the same REAL test dataset.")
print()
print("Images are loaded as RGB (3 channels).")
print("=" * 70)


# ============================================================
# DATASET
# ============================================================

class MRIDataset(Dataset):

    def __init__(self, samples):

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        path, label = self.samples[index]

        try:

            # ALWAYS RGB
            image = Image.open(path).convert("RGB")

            image = image.resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                Image.Resampling.BILINEAR
            )

            image = np.asarray(image, dtype=np.float32) / 255.0

            # H,W,C -> C,H,W
            image = np.transpose(image, (2, 0, 1))

            image = torch.tensor(
                image,
                dtype=torch.float32
            )

            label = torch.tensor(
                label,
                dtype=torch.long
            )

            return image, label

        except Exception as e:

            print(f"Error loading image: {path}")
            print(e)

            # fallback black RGB image
            image = torch.zeros(
                (3, IMAGE_SIZE, IMAGE_SIZE),
                dtype=torch.float32
            )

            label = torch.tensor(
                label,
                dtype=torch.long
            )

            return image, label


# ============================================================
# DATA LOADING
# ============================================================

def load_samples(root_dir, dataset_name):

    print()
    print("=" * 70)
    print(f"LOADING {dataset_name.upper()}")
    print("=" * 70)
    print(f"Path: {root_dir}")

    samples = []

    if not os.path.exists(root_dir):

        raise FileNotFoundError(
            f"Dataset directory not found:\n{root_dir}"
        )

    for class_index, class_name in enumerate(CLASS_NAMES):

        class_dir = os.path.join(
            root_dir,
            class_name
        )

        if not os.path.exists(class_dir):

            print(
                f"{class_name:<14}: directory not found"
            )

            continue

        files = []

        for filename in os.listdir(class_dir):

            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            ):

                files.append(
                    os.path.join(
                        class_dir,
                        filename
                    )
                )

        files.sort()

        for path in files:

            samples.append(
                (path, class_index)
            )

        print(
            f"{class_name:<14}: {len(files):,} images"
        )

    print()
    print(f"Total images: {len(samples):,}")

    return samples


# ============================================================
# STRATIFIED SYNTHETIC SELECTION
# ============================================================

def select_synthetic_percentage(
    synthetic_samples,
    percentage
):

    selected = []

    print()
    print(
        f"Selecting {int(percentage * 100)}% synthetic data..."
    )

    for class_index, class_name in enumerate(CLASS_NAMES):

        class_samples = [
            item
            for item in synthetic_samples
            if item[1] == class_index
        ]

        random.shuffle(class_samples)

        count = int(
            len(class_samples) * percentage
        )

        selected_class = class_samples[:count]

        selected.extend(selected_class)

        print(
            f"{class_name:<14}: "
            f"{len(selected_class):,} selected"
        )

    return selected


# ============================================================
# CNN MODEL
# ============================================================

class MRIClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        # RGB INPUT = 3 CHANNELS
        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.30),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.20),

            nn.Linear(
                128,
                NUM_CLASSES
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# DATA LOADER
# ============================================================

def create_loader(
    samples,
    shuffle=True
):

    dataset = MRIDataset(samples)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False
    )

    return loader


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    train_samples,
    experiment_name
):

    print()
    print("=" * 70)
    print(
        f"TRAINING: {experiment_name.upper()}"
    )
    print("=" * 70)

    print(
        f"Training images: {len(train_samples):,}"
    )

    loader = create_loader(
        train_samples,
        shuffle=True
    )

    model = MRIClassifier().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.5
    )

    history = []

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        start_time = time.time()

        for images, labels in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            running_loss += (
                loss.item() *
                images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        scheduler.step()

        epoch_loss = (
            running_loss / total
        )

        epoch_accuracy = (
            correct / total
        )

        elapsed = (
            time.time() - start_time
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Accuracy: "
            f"{epoch_accuracy * 100:.2f}% | "
            f"Time: {elapsed / 60:.1f} min"
        )

        history.append({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "accuracy": epoch_accuracy
        })

    return model, history


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    test_samples,
    experiment_name
):

    print()
    print(
        f"EVALUATING {experiment_name.upper()}..."
    )

    loader = create_loader(
        test_samples,
        shuffle=False
    )

    model.eval()

    all_predictions = []

    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    print()
    print(
        f"{experiment_name.upper()} RESULTS"
    )
    print("-" * 50)

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
    print("Classification Report")

    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "labels": all_labels,
        "predictions": all_predictions
    }


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    labels,
    predictions,
    filename,
    title
):

    cm = confusion_matrix(
        labels,
        predictions,
        labels=list(range(NUM_CLASSES))
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    image = ax.imshow(cm)

    ax.set_title(
        title,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "Actual Class"
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

    for i in range(NUM_CLASSES):

        for j in range(NUM_CLASSES):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    fig.colorbar(
        image,
        ax=ax
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Confusion matrix saved: {path}"
    )


# ============================================================
# COMPARISON CHART
# ============================================================

def create_comparison_chart(results_df):

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    x = np.arange(
        len(results_df["Experiment"])
    )

    width = 0.18

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    for index, metric in enumerate(metrics):

        values = (
            results_df[metric]
            .values
        )

        ax.bar(
            x + (
                index - 1.5
            ) * width,
            values,
            width,
            label=metric
        )

    ax.set_title(
        "MRI Five-Experiment ML Performance Comparison",
        fontweight="bold",
        fontsize=14
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_xlabel(
        "Training Strategy"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        results_df["Experiment"],
        rotation=20,
        ha="right"
    )

    ax.set_ylim(
        0,
        1
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.legend()

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "five_experiment_comparison.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(
        f"Comparison chart saved: {path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING DATASETS")
    print("=" * 70)

    real_train = load_samples(
        REAL_TRAIN_DIR,
        "Real Training Data"
    )

    synthetic = load_samples(
        SYNTHETIC_DIR,
        "Synthetic Training Data"
    )

    real_test = load_samples(
        REAL_TEST_DIR,
        "Real Test Data"
    )

    # --------------------------------------------------------
    # CREATE EXPERIMENTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING FIVE EXPERIMENTS")
    print("=" * 70)

    synthetic_25 = select_synthetic_percentage(
        synthetic,
        0.25
    )

    synthetic_50 = select_synthetic_percentage(
        synthetic,
        0.50
    )

    experiments = {

        "Real Only":
            real_train,

        "Synthetic Only":
            synthetic,

        "Real + 25% Synthetic":
            real_train + synthetic_25,

        "Real + 50% Synthetic":
            real_train + synthetic_50,

        "Real + 100% Synthetic":
            real_train + synthetic
    }

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results = []

    # --------------------------------------------------------
    # FIVE EXPERIMENTS
    # --------------------------------------------------------

    for experiment_name, train_samples in experiments.items():

        print()
        print()
        print("#" * 70)

        print(
            f"EXPERIMENT: {experiment_name}"
        )

        print("#" * 70)

        model, history = train_model(
            train_samples,
            experiment_name
        )

        evaluation = evaluate_model(
            model,
            real_test,
            experiment_name
        )

        # Filename
        safe_name = (
            experiment_name
            .lower()
            .replace("+", "plus")
            .replace("%", "pct")
            .replace(" ", "_")
        )

        save_confusion_matrix(
            evaluation["labels"],
            evaluation["predictions"],
            f"{safe_name}_confusion_matrix.png",
            f"{experiment_name} - Confusion Matrix"
        )

        results.append({

            "Experiment":
                experiment_name,

            "Training Images":
                len(train_samples),

            "Accuracy":
                evaluation["accuracy"],

            "Precision":
                evaluation["precision"],

            "Recall":
                evaluation["recall"],

            "F1 Score":
                evaluation["f1"]
        })

        # Free memory
        del model

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # RESULTS DATAFRAME
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # UTILITY RETENTION
    # --------------------------------------------------------

    real_accuracy = (
        results_df.loc[
            results_df["Experiment"] == "Real Only",
            "Accuracy"
        ].iloc[0]
    )

    results_df["Utility Retention"] = (
        results_df["Accuracy"] /
        real_accuracy *
        100
    )

    # --------------------------------------------------------
    # SAVE MAIN CSV
    # --------------------------------------------------------

    results_csv = os.path.join(
        OUTPUT_DIR,
        "mri_five_experiment_results.csv"
    )

    results_df.to_csv(
        results_csv,
        index=False
    )

    print()
    print("=" * 70)
    print("FIVE-EXPERIMENT RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print()
    print(
        f"Results CSV saved: {results_csv}"
    )

    # --------------------------------------------------------
    # UTILITY RETENTION CSV
    # --------------------------------------------------------

    utility_df = results_df[
        [
            "Experiment",
            "Accuracy",
            "Utility Retention"
        ]
    ].copy()

    utility_csv = os.path.join(
        OUTPUT_DIR,
        "mri_utility_retention.csv"
    )

    utility_df.to_csv(
        utility_csv,
        index=False
    )

    print(
        f"Utility CSV saved: {utility_csv}"
    )

    # --------------------------------------------------------
    # COMPARISON CHART
    # --------------------------------------------------------

    create_comparison_chart(
        results_df
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MRI FIVE-EXPERIMENT VALIDATION COMPLETED")
    print("=" * 70)

    print()

    for _, row in results_df.iterrows():

        print(
            f"{row['Experiment']:<25} "
            f"Accuracy: {row['Accuracy']:.4f} | "
            f"F1: {row['F1 Score']:.4f} | "
            f"Utility: {row['Utility Retention']:.2f}%"
        )

    print()
    print("Output directory:")
    print(
        os.path.abspath(OUTPUT_DIR)
    )

    print()
    print("Generated files:")

    print(
        "1. mri_five_experiment_results.csv"
    )

    print(
        "2. mri_utility_retention.csv"
    )

    print(
        "3. real_only_confusion_matrix.png"
    )

    print(
        "4. synthetic_only_confusion_matrix.png"
    )

    print(
        "5. real_plus_25pct_confusion_matrix.png"
    )

    print(
        "6. real_plus_50pct_confusion_matrix.png"
    )

    print(
        "7. real_plus_100pct_confusion_matrix.png"
    )

    print(
        "8. five_experiment_comparison.png"
    )

    print()
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()


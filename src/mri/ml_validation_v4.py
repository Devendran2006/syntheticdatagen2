import os
import random
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset

from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_DIR = "data/imaging/MRI/Training_Preprocessed"

TEST_DIR = "data/imaging/MRI/Testing"

SYNTHETIC_DIR = (
    "outputs/mri/v4/evaluation/"
    "synthetic_samples"
)

OUTPUT_DIR = (
    "outputs/mri/v4/ml_validation"
)

RESULT_DIR = os.path.join(
    OUTPUT_DIR,
    "results"
)

CONFUSION_DIR = os.path.join(
    OUTPUT_DIR,
    "confusion_matrices"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

os.makedirs(
    CONFUSION_DIR,
    exist_ok=True
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

IMAGE_SIZE = 128

NUM_CLASSES = 4

BATCH_SIZE = 32

EPOCHS = 10

LEARNING_RATE = 0.001

RANDOM_SEED = 42

CPU_THREADS = min(
    4,
    os.cpu_count() or 1
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)

torch.set_num_threads(
    CPU_THREADS
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("MRI GAN V4 ML VALIDATION")
print("=" * 70)

print(
    f"Device              : {DEVICE}"
)

print(
    f"Image Size          : "
    f"{IMAGE_SIZE}x{IMAGE_SIZE}"
)

print(
    f"Batch Size          : {BATCH_SIZE}"
)

print(
    f"Classifier Epochs   : {EPOCHS}"
)

print(
    f"Learning Rate       : "
    f"{LEARNING_RATE}"
)

print(
    f"V4 Synthetic Source : "
    f"{SYNTHETIC_DIR}"
)

print("=" * 70)


# ============================================================
# PATH VALIDATION
# ============================================================

required_paths = {
    "Training data":
        TRAIN_DIR,

    "Real testing data":
        TEST_DIR,

    "V4 synthetic data":
        SYNTHETIC_DIR
}


print(
    "\nChecking required paths..."
)


for name, path in required_paths.items():

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\n{name} not found:\n{path}\n"
        )

    print(
        f"OK  {name:<25} : {path}"
    )


# ============================================================
# TRANSFORMS
# ============================================================

transform = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=1
    ),

    transforms.Resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    ),

    transforms.ToTensor()
])


# ============================================================
# DATASET LOADER
# ============================================================

def load_dataset(
    path,
    dataset_name
):

    print(
        f"\nLoading {dataset_name}..."
    )

    dataset = datasets.ImageFolder(
        root=path,
        transform=transform
    )

    print(
        f"{dataset_name} size : "
        f"{len(dataset)}"
    )

    print(
        f"Classes            : "
        f"{dataset.classes}"
    )

    return dataset


# ============================================================
# LOAD DATASETS
# ============================================================

real_train_dataset = load_dataset(
    TRAIN_DIR,
    "REAL TRAINING DATA"
)


real_test_dataset = load_dataset(
    TEST_DIR,
    "REAL TESTING DATA"
)


synthetic_dataset = load_dataset(
    SYNTHETIC_DIR,
    "V4 SYNTHETIC DATA"
)


# ============================================================
# VERIFY CLASS ORDER
# ============================================================

expected_classes = sorted(
    CLASS_NAMES
)


if sorted(
    real_train_dataset.classes
) != expected_classes:

    raise ValueError(
        "Real training class structure "
        "does not match expected classes."
    )


if sorted(
    real_test_dataset.classes
) != expected_classes:

    raise ValueError(
        "Real testing class structure "
        "does not match expected classes."
    )


if sorted(
    synthetic_dataset.classes
) != expected_classes:

    raise ValueError(
        "Synthetic class structure "
        "does not match expected classes."
    )


# ============================================================
# PRINT CLASS COUNTS
# ============================================================

def print_class_counts(
    dataset,
    dataset_name
):

    counts = {
        name: 0
        for name in CLASS_NAMES
    }


    for _, label in dataset.samples:

        class_name = dataset.classes[
            label
        ]

        if class_name in counts:

            counts[class_name] += 1


    print(
        f"\n{dataset_name} class distribution:"
    )


    for name in CLASS_NAMES:

        print(
            f"  {name:<15}: "
            f"{counts[name]}"
        )


    return counts


real_train_counts = print_class_counts(
    real_train_dataset,
    "REAL TRAINING"
)


real_test_counts = print_class_counts(
    real_test_dataset,
    "REAL TESTING"
)


synthetic_counts = print_class_counts(
    synthetic_dataset,
    "V4 SYNTHETIC"
)


# ============================================================
# BALANCE CHECK
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "DATASET BALANCE CHECK"
)

print(
    "=" * 70
)


for class_name in CLASS_NAMES:

    print(
        f"{class_name:<15} "
        f"Real Train = "
        f"{real_train_counts[class_name]:<5} "
        f"Real Test = "
        f"{real_test_counts[class_name]:<5} "
        f"Synthetic = "
        f"{synthetic_counts[class_name]}"
    )


# ============================================================
# SIMPLE CNN CLASSIFIER
# ============================================================

class MRIClassifier(
    nn.Module
):

    def __init__(
        self,
        num_classes=4
    ):

        super().__init__()

        self.features = nn.Sequential(

            # 128 -> 64

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                2
            ),


            # 64 -> 32

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                2
            ),


            # 32 -> 16

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                2
            ),


            # 16 -> 8

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                2
            )
        )


        self.classifier = nn.Sequential(

            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),

            nn.Flatten(),

            nn.Dropout(
                0.30
            ),

            nn.Linear(
                256,
                num_classes
            )
        )


    def forward(
        self,
        x
    ):

        x = self.features(
            x
        )

        x = self.classifier(
            x
        )

        return x


# ============================================================
# CREATE DATA LOADER
# ============================================================

def create_loader(
    dataset,
    shuffle
):

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False
    )


# ============================================================
# TRAIN CLASSIFIER
# ============================================================

def train_classifier(
    train_dataset,
    experiment_name
):

    print(
        "\n" + "=" * 70
    )

    print(
        f"TRAINING CLASSIFIER: "
        f"{experiment_name}"
    )

    print(
        "=" * 70
    )


    model = MRIClassifier(
        num_classes=NUM_CLASSES
    ).to(
        DEVICE
    )


    criterion = nn.CrossEntropyLoss()


    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    train_loader = create_loader(
        train_dataset,
        shuffle=True
    )


    history = []


    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()


        running_loss = 0.0

        correct = 0

        total = 0


        for images, labels in train_loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
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


            total += (
                labels.size(0)
            )


        epoch_loss = (
            running_loss /
            max(total, 1)
        )


        epoch_accuracy = (
            correct /
            max(total, 1)
        )


        history.append(
            {
                "epoch": epoch,

                "loss":
                    epoch_loss,

                "accuracy":
                    epoch_accuracy
            }
        )


        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"Loss: {epoch_loss:.4f} "
            f"Train Accuracy: "
            f"{epoch_accuracy * 100:.2f}%"
        )


    history_df = pd.DataFrame(
        history
    )


    history_path = os.path.join(
        RESULT_DIR,
        f"{experiment_name}_training_history.csv"
    )


    history_df.to_csv(
        history_path,
        index=False
    )


    return model


# ============================================================
# EVALUATE CLASSIFIER
# ============================================================

def evaluate_classifier(
    model,
    test_dataset,
    experiment_name
):

    print(
        "\n" + "-" * 70
    )

    print(
        f"EVALUATING: "
        f"{experiment_name}"
    )

    print(
        "-" * 70
    )


    model.eval()


    test_loader = create_loader(
        test_dataset,
        shuffle=False
    )


    all_predictions = []

    all_labels = []


    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                DEVICE
            )


            outputs = model(
                images
            )


            predictions = (
                outputs.argmax(
                    dim=1
                )
            )


            all_predictions.extend(
                predictions.cpu().numpy()
            )


            all_labels.extend(
                labels.numpy()
            )


    y_true = np.array(
        all_labels
    )


    y_pred = np.array(
        all_predictions
    )


    accuracy = accuracy_score(
        y_true,
        y_pred
    )


    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )


    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )


    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )


    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(NUM_CLASSES)
        )
    )


    print(
        f"Accuracy       : "
        f"{accuracy * 100:.2f}%"
    )


    print(
        f"Precision      : "
        f"{precision * 100:.2f}%"
    )


    print(
        f"Recall         : "
        f"{recall * 100:.2f}%"
    )


    print(
        f"Macro F1       : "
        f"{macro_f1 * 100:.2f}%"
    )


    print(
        f"Weighted F1    : "
        f"{weighted_f1 * 100:.2f}%"
    )


    print(
        "\nClassification Report:"
    )


    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0
    )


    print(
        report
    )


    # --------------------------------------------------------
    # Save classification report
    # --------------------------------------------------------

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )


    report_df = pd.DataFrame(
        report_dict
    ).transpose()


    report_path = os.path.join(
        RESULT_DIR,
        f"{experiment_name}_classification_report.csv"
    )


    report_df.to_csv(
        report_path
    )


    # --------------------------------------------------------
    # Confusion matrix CSV
    # --------------------------------------------------------

    cm_df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )


    cm_csv_path = os.path.join(
        RESULT_DIR,
        f"{experiment_name}_confusion_matrix.csv"
    )


    cm_df.to_csv(
        cm_csv_path
    )


    # --------------------------------------------------------
    # Confusion matrix image
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 7)
    )


    plt.imshow(
        cm,
        interpolation="nearest"
    )


    plt.title(
        f"{experiment_name} - Confusion Matrix"
    )


    plt.colorbar()


    tick_marks = np.arange(
        NUM_CLASSES
    )


    plt.xticks(
        tick_marks,
        CLASS_NAMES,
        rotation=45,
        ha="right"
    )


    plt.yticks(
        tick_marks,
        CLASS_NAMES
    )


    threshold = (
        cm.max() / 2.0
        if cm.max() > 0
        else 0
    )


    for i in range(
        NUM_CLASSES
    ):

        for j in range(
            NUM_CLASSES
        ):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                color=(
                    "white"
                    if cm[i, j] > threshold
                    else "black"
                )
            )


    plt.ylabel(
        "TRUE LABEL"
    )


    plt.xlabel(
        "PREDICTED LABEL"
    )


    plt.tight_layout()


    cm_image_path = os.path.join(
        CONFUSION_DIR,
        f"{experiment_name}_confusion_matrix.png"
    )


    plt.savefig(
        cm_image_path,
        dpi=150,
        bbox_inches="tight"
    )


    plt.close()


    return {
        "experiment":
            experiment_name,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "macro_f1":
            macro_f1,

        "weighted_f1":
            weighted_f1,

        "confusion_matrix":
            cm
    }


# ============================================================
# EXPERIMENT 1
# REAL TRAINING -> REAL TESTING
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "EXPERIMENT 1"
)

print(
    "REAL TRAINING -> REAL TESTING"
)

print(
    "=" * 70
)


real_model = train_classifier(
    real_train_dataset,
    "real_only"
)


real_result = evaluate_classifier(
    real_model,
    real_test_dataset,
    "real_only"
)


# ============================================================
# EXPERIMENT 2
# V4 SYNTHETIC -> REAL TESTING
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "EXPERIMENT 2"
)

print(
    "V4 SYNTHETIC -> REAL TESTING"
)

print(
    "=" * 70
)


synthetic_model = train_classifier(
    synthetic_dataset,
    "v4_synthetic_only"
)


synthetic_result = evaluate_classifier(
    synthetic_model,
    real_test_dataset,
    "v4_synthetic_only"
)


# ============================================================
# EXPERIMENT 3
# REAL + V4 SYNTHETIC -> REAL TESTING
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "EXPERIMENT 3"
)

print(
    "REAL + V4 SYNTHETIC -> REAL TESTING"
)

print(
    "=" * 70
)


combined_dataset = torch.utils.data.ConcatDataset(
    [
        real_train_dataset,
        synthetic_dataset
    ]
)


combined_model = train_classifier(
    combined_dataset,
    "real_plus_v4_synthetic"
)


combined_result = evaluate_classifier(
    combined_model,
    real_test_dataset,
    "real_plus_v4_synthetic"
)


# ============================================================
# UTILITY RETENTION
# ============================================================

real_accuracy = (
    real_result["accuracy"]
)

synthetic_accuracy = (
    synthetic_result["accuracy"]
)

combined_accuracy = (
    combined_result["accuracy"]
)


if real_accuracy > 0:

    synthetic_utility_retention = (
        synthetic_accuracy /
        real_accuracy
    ) * 100.0


    combined_utility_retention = (
        combined_accuracy /
        real_accuracy
    ) * 100.0

else:

    synthetic_utility_retention = 0.0

    combined_utility_retention = 0.0


# ============================================================
# SUMMARY TABLE
# ============================================================

summary_rows = [

    {
        "Experiment":
            "Real Only",

        "Accuracy":
            real_result["accuracy"] * 100,

        "Precision":
            real_result["precision"] * 100,

        "Recall":
            real_result["recall"] * 100,

        "Macro F1":
            real_result["macro_f1"] * 100,

        "Weighted F1":
            real_result["weighted_f1"] * 100,

        "Utility Retention":
            100.0
    },

    {
        "Experiment":
            "V4 Synthetic Only",

        "Accuracy":
            synthetic_result["accuracy"] * 100,

        "Precision":
            synthetic_result["precision"] * 100,

        "Recall":
            synthetic_result["recall"] * 100,

        "Macro F1":
            synthetic_result["macro_f1"] * 100,

        "Weighted F1":
            synthetic_result["weighted_f1"] * 100,

        "Utility Retention":
            synthetic_utility_retention
    },

    {
        "Experiment":
            "Real + V4 Synthetic",

        "Accuracy":
            combined_result["accuracy"] * 100,

        "Precision":
            combined_result["precision"] * 100,

        "Recall":
            combined_result["recall"] * 100,

        "Macro F1":
            combined_result["macro_f1"] * 100,

        "Weighted F1":
            combined_result["weighted_f1"] * 100,

        "Utility Retention":
            combined_utility_retention
    }
]


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    RESULT_DIR,
    "mri_gan_v4_ml_validation_summary.csv"
)


summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "FINAL MRI GAN V4 ML VALIDATION"
)

print(
    "=" * 70
)


print(
    "\nREAL ONLY"
)

print(
    f"Accuracy       : "
    f"{real_result['accuracy'] * 100:.2f}%"
)

print(
    f"Precision      : "
    f"{real_result['precision'] * 100:.2f}%"
)

print(
    f"Recall         : "
    f"{real_result['recall'] * 100:.2f}%"
)

print(
    f"Macro F1       : "
    f"{real_result['macro_f1'] * 100:.2f}%"
)

print(
    f"Weighted F1    : "
    f"{real_result['weighted_f1'] * 100:.2f}%"
)


print(
    "\nV4 SYNTHETIC ONLY"
)

print(
    f"Accuracy       : "
    f"{synthetic_result['accuracy'] * 100:.2f}%"
)

print(
    f"Precision      : "
    f"{synthetic_result['precision'] * 100:.2f}%"
)

print(
    f"Recall         : "
    f"{synthetic_result['recall'] * 100:.2f}%"
)

print(
    f"Macro F1       : "
    f"{synthetic_result['macro_f1'] * 100:.2f}%"
)

print(
    f"Weighted F1    : "
    f"{synthetic_result['weighted_f1'] * 100:.2f}%"
)

print(
    f"Utility Retention : "
    f"{synthetic_utility_retention:.2f}%"
)


print(
    "\nREAL + V4 SYNTHETIC"
)

print(
    f"Accuracy       : "
    f"{combined_result['accuracy'] * 100:.2f}%"
)

print(
    f"Precision      : "
    f"{combined_result['precision'] * 100:.2f}%"
)

print(
    f"Recall         : "
    f"{combined_result['recall'] * 100:.2f}%"
)

print(
    f"Macro F1       : "
    f"{combined_result['macro_f1'] * 100:.2f}%"
)

print(
    f"Weighted F1    : "
    f"{combined_result['weighted_f1'] * 100:.2f}%"
)

print(
    f"Utility Retention : "
    f"{combined_utility_retention:.2f}%"
)


# ============================================================
# COMPARISON TABLE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "COMPARISON"
)

print(
    "=" * 70
)


display_df = summary_df.copy()


for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "Macro F1",
    "Weighted F1",
    "Utility Retention"
]:

    display_df[column] = display_df[
        column
    ].map(
        lambda x:
        f"{x:.2f}%"
    )


print(
    display_df.to_string(
        index=False
    )
)


# ============================================================
# BAR CHART
# ============================================================

print(
    "\nCreating ML comparison chart..."
)


metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "Macro F1",
    "Weighted F1"
]


x = np.arange(
    len(metrics)
)


width = 0.25


plt.figure(
    figsize=(12, 7)
)


plt.bar(
    x - width,
    [
        real_result["accuracy"] * 100,
        real_result["precision"] * 100,
        real_result["recall"] * 100,
        real_result["macro_f1"] * 100,
        real_result["weighted_f1"] * 100
    ],
    width,
    label="Real Only"
)


plt.bar(
    x,
    [
        synthetic_result["accuracy"] * 100,
        synthetic_result["precision"] * 100,
        synthetic_result["recall"] * 100,
        synthetic_result["macro_f1"] * 100,
        synthetic_result["weighted_f1"] * 100
    ],
    width,
    label="V4 Synthetic Only"
)


plt.bar(
    x + width,
    [
        combined_result["accuracy"] * 100,
        combined_result["precision"] * 100,
        combined_result["recall"] * 100,
        combined_result["macro_f1"] * 100,
        combined_result["weighted_f1"] * 100
    ],
    width,
    label="Real + V4 Synthetic"
)


plt.xticks(
    x,
    metrics
)


plt.ylim(
    0,
    105
)


plt.ylabel(
    "SCORE (%)"
)


plt.title(
    "MRI GAN V4 - ML Utility Validation"
)


plt.legend()


plt.tight_layout()


comparison_chart = os.path.join(
    RESULT_DIR,
    "mri_gan_v4_ml_comparison.png"
)


plt.savefig(
    comparison_chart,
    dpi=150,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# UTILITY RETENTION CHART
# ============================================================

plt.figure(
    figsize=(9, 6)
)


utility_names = [
    "V4 Synthetic",
    "Real + V4 Synthetic"
]


utility_scores = [
    synthetic_utility_retention,
    combined_utility_retention
]


bars = plt.bar(
    utility_names,
    utility_scores
)


plt.ylim(
    0,
    max(
        110,
        max(utility_scores) + 10
    )
)


plt.ylabel(
    "UTILITY RETENTION (%)"
)


plt.title(
    "MRI GAN V4 - Utility Retention"
)


for bar, score in zip(
    bars,
    utility_scores
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{score:.2f}%",
        ha="center",
        va="bottom",
        fontweight="bold"
    )


plt.tight_layout()


utility_chart = os.path.join(
    RESULT_DIR,
    "mri_gan_v4_utility_retention.png"
)


plt.savefig(
    utility_chart,
    dpi=150,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# SAVE JSON SUMMARY
# ============================================================

json_summary = {

    "v4_epoch": 10,

    "real_only": {
        "accuracy":
            real_result["accuracy"],

        "precision":
            real_result["precision"],

        "recall":
            real_result["recall"],

        "macro_f1":
            real_result["macro_f1"],

        "weighted_f1":
            real_result["weighted_f1"]
    },

    "v4_synthetic_only": {
        "accuracy":
            synthetic_result["accuracy"],

        "precision":
            synthetic_result["precision"],

        "recall":
            synthetic_result["recall"],

        "macro_f1":
            synthetic_result["macro_f1"],

        "weighted_f1":
            synthetic_result["weighted_f1"],

        "utility_retention":
            synthetic_utility_retention
    },

    "real_plus_v4_synthetic": {
        "accuracy":
            combined_result["accuracy"],

        "precision":
            combined_result["precision"],

        "recall":
            combined_result["recall"],

        "macro_f1":
            combined_result["macro_f1"],

        "weighted_f1":
            combined_result["weighted_f1"],

        "utility_retention":
            combined_utility_retention
    }
}


json_path = os.path.join(
    RESULT_DIR,
    "mri_gan_v4_ml_validation.json"
)


with open(
    json_path,
    "w"
) as f:

    json.dump(
        json_summary,
        f,
        indent=4
    )


# ============================================================
# INTERPRETATION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "MRI GAN V4 ML INTERPRETATION"
)

print(
    "=" * 70
)


print(
    f"Real-only baseline accuracy : "
    f"{real_accuracy * 100:.2f}%"
)


print(
    f"V4 synthetic accuracy      : "
    f"{synthetic_accuracy * 100:.2f}%"
)


print(
    f"V4 utility retention       : "
    f"{synthetic_utility_retention:.2f}%"
)


print(
    f"Combined accuracy          : "
    f"{combined_accuracy * 100:.2f}%"
)


print(
    f"Combined utility retention : "
    f"{combined_utility_retention:.2f}%"
)


# ------------------------------------------------------------
# Synthetic utility status
# ------------------------------------------------------------

if synthetic_utility_retention >= 80:

    print(
        "\nSynthetic Utility Status : STRONG"
    )

    print(
        "V4 synthetic data retains "
        "a substantial amount of "
        "downstream classification utility."
    )

elif synthetic_utility_retention >= 60:

    print(
        "\nSynthetic Utility Status : MODERATE"
    )

    print(
        "V4 synthetic data shows "
        "meaningful downstream utility "
        "but remains below the real-data baseline."
    )

else:

    print(
        "\nSynthetic Utility Status : LOW"
    )

    print(
        "V4 synthetic data has limited "
        "downstream classification utility."
    )


# ------------------------------------------------------------
# Combined dataset status
# ------------------------------------------------------------

if combined_accuracy > real_accuracy:

    print(
        "\nAugmentation Effect : POSITIVE"
    )

    print(
        "Adding V4 synthetic data "
        "improved real-test performance."
    )

elif combined_accuracy >= (
    real_accuracy * 0.95
):

    print(
        "\nAugmentation Effect : STABLE"
    )

    print(
        "Adding V4 synthetic data "
        "maintained performance close "
        "to the real-only baseline."
    )

else:

    print(
        "\nAugmentation Effect : NEGATIVE"
    )

    print(
        "Adding V4 synthetic data "
        "reduced real-test performance."
    )


# ============================================================
# IMPORTANT DISCLAIMER
# ============================================================

print(
    "\nIMPORTANT:"
)

print(
    "This is an engineering ML utility experiment."
)

print(
    "It does NOT establish medical realism."
)

print(
    "It does NOT establish clinical validity."
)

print(
    "It does NOT establish diagnostic safety."
)

print(
    "The real Testing dataset was kept separate "
    "from GAN training."
)


# ============================================================
# OUTPUTS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "ML VALIDATION COMPLETE"
)

print(
    "=" * 70
)

print(
    f"Summary CSV          : "
    f"{summary_path}"
)

print(
    f"JSON Summary         : "
    f"{json_path}"
)

print(
    f"ML Comparison Chart  : "
    f"{comparison_chart}"
)

print(
    f"Utility Chart        : "
    f"{utility_chart}"
)

print(
    f"Confusion Matrices   : "
    f"{CONFUSION_DIR}"
)

print(
    f"Detailed Results     : "
    f"{RESULT_DIR}"
)

print(
    "=" * 70
)
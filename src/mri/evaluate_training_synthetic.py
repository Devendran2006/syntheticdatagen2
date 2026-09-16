"""
MRI TRAINING SYNTHETIC DATA EVALUATION
======================================

Purpose:
    Compare REAL MRI TRAINING data with the newly generated
    SYNTHETIC MRI TRAINING data.

IMPORTANT:
    - Real Training data is used only for comparison.
    - Synthetic Training data is evaluated only.
    - Real Testing data is NEVER used.
    - Real Testing data is NEVER modified.

Input:
    data/imaging/MRI/Training
    outputs/mri/training_synthetic/synthetic

Output:
    outputs/mri/training_synthetic/evaluation/

Generated:
    1. training_quality_summary.csv
    2. training_class_comparison.csv
    3. training_pixel_statistics.csv
    4. training_pixel_distribution.png
    5. training_mean_comparison.png
    6. training_std_comparison.png
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

REAL_TRAINING_DIR = (
    BASE_DIR
    / "data"
    / "imaging"
    / "MRI"
    / "Training"
)

SYNTHETIC_TRAINING_DIR = (
    BASE_DIR
    / "outputs"
    / "mri"
    / "training_synthetic"
    / "synthetic"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "mri"
    / "training_synthetic"
    / "evaluation"
)

IMAGE_SIZE = 64

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

VALID_EXTENSIONS = [
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp"
]

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# CREATE OUTPUT DIRECTORY
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
print("       MRI TRAINING SYNTHETIC DATA EVALUATION")
print("=" * 70)
print()

print("Real Training path:")
print(REAL_TRAINING_DIR)
print()

print("Synthetic Training path:")
print(SYNTHETIC_TRAINING_DIR)
print()

print("Image size:")
print(f"{IMAGE_SIZE} x {IMAGE_SIZE}")
print()

print("IMPORTANT:")
print("Real Testing data is NOT used.")
print("Real Testing data is NOT modified.")
print()


# ============================================================
# GET IMAGE FILES
# ============================================================

def get_image_files(directory):

    if not directory.exists():
        return []

    files = []

    for file in directory.iterdir():

        if file.is_file():

            if file.suffix.lower() in VALID_EXTENSIONS:
                files.append(file)

    return sorted(files)


# ============================================================
# LOAD IMAGE STATISTICS
# ============================================================

def calculate_image_statistics(
    image_path
):

    try:

        image = Image.open(
            image_path
        ).convert("L")

        image = image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        )

        array = (
            np.asarray(
                image,
                dtype=np.float32
            )
            / 255.0
        )

        return {
            "valid": True,
            "mean": float(array.mean()),
            "std": float(array.std()),
            "min": float(array.min()),
            "max": float(array.max())
        }

    except Exception:

        return {
            "valid": False,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0
        }


# ============================================================
# ANALYZE DATASET
# ============================================================

def analyze_dataset(
    dataset_dir,
    dataset_name
):

    print()
    print("=" * 70)
    print(
        f"             {dataset_name.upper()}"
    )
    print("=" * 70)
    print()

    if not dataset_dir.exists():

        raise FileNotFoundError(
            f"\nDataset directory not found:\n"
            f"{dataset_dir}"
        )

    rows = []

    total_files = 0
    total_valid = 0
    total_corrupted = 0

    for class_name in CLASS_NAMES:

        class_dir = (
            dataset_dir / class_name
        )

        if not class_dir.exists():

            print(
                f"{class_name:<15}: "
                "DIRECTORY NOT FOUND"
            )

            continue

        files = get_image_files(
            class_dir
        )

        total_files += len(files)

        means = []
        stds = []
        minimums = []
        maximums = []

        valid_count = 0
        corrupted_count = 0

        for image_path in files:

            stats = calculate_image_statistics(
                image_path
            )

            if stats["valid"]:

                valid_count += 1

                means.append(
                    stats["mean"]
                )

                stds.append(
                    stats["std"]
                )

                minimums.append(
                    stats["min"]
                )

                maximums.append(
                    stats["max"]
                )

            else:

                corrupted_count += 1

        total_valid += valid_count
        total_corrupted += corrupted_count

        mean_pixel = (
            float(np.mean(means))
            if means
            else 0.0
        )

        std_pixel = (
            float(np.mean(stds))
            if stds
            else 0.0
        )

        min_pixel = (
            float(np.mean(minimums))
            if minimums
            else 0.0
        )

        max_pixel = (
            float(np.mean(maximums))
            if maximums
            else 0.0
        )

        print(
            f"{class_name:<15}: "
            f"{valid_count:>5} valid images"
        )

        if corrupted_count > 0:

            print(
                f"{'':15}  "
                f"{corrupted_count} corrupted"
            )

        rows.append({

            "Dataset":
                dataset_name,

            "Class":
                class_name,

            "Images":
                len(files),

            "Valid_Images":
                valid_count,

            "Corrupted_Images":
                corrupted_count,

            "Mean_Pixel":
                mean_pixel,

            "Pixel_Std":
                std_pixel,

            "Min_Pixel":
                min_pixel,

            "Max_Pixel":
                max_pixel
        })

    print()

    print(
        f"Total files      : {total_files}"
    )

    print(
        f"Valid files      : {total_valid}"
    )

    print(
        f"Corrupted files  : {total_corrupted}"
    )

    print()

    return pd.DataFrame(rows)


# ============================================================
# CALCULATE SIMILARITY
# ============================================================

def calculate_similarity(
    real_df,
    synthetic_df
):

    print()
    print("=" * 70)
    print("             CALCULATING SIMILARITY")
    print("=" * 70)
    print()

    rows = []

    for class_name in CLASS_NAMES:

        real_rows = real_df[
            real_df["Class"] == class_name
        ]

        synthetic_rows = synthetic_df[
            synthetic_df["Class"] == class_name
        ]

        if real_rows.empty:
            continue

        if synthetic_rows.empty:
            continue

        real = real_rows.iloc[0]
        synthetic = synthetic_rows.iloc[0]

        real_mean = float(
            real["Mean_Pixel"]
        )

        synthetic_mean = float(
            synthetic["Mean_Pixel"]
        )

        real_std = float(
            real["Pixel_Std"]
        )

        synthetic_std = float(
            synthetic["Pixel_Std"]
        )

        # ----------------------------------------------------
        # Mean similarity
        # ----------------------------------------------------

        mean_difference = abs(
            real_mean -
            synthetic_mean
        )

        mean_similarity = max(
            0.0,
            1.0 - mean_difference
        )

        # ----------------------------------------------------
        # Standard deviation similarity
        # ----------------------------------------------------

        std_difference = abs(
            real_std -
            synthetic_std
        )

        std_similarity = max(
            0.0,
            1.0 - std_difference
        )

        # ----------------------------------------------------
        # Overall distribution score
        # ----------------------------------------------------

        distribution_score = (
            mean_similarity * 0.5
            +
            std_similarity * 0.5
        )

        rows.append({

            "Class":
                class_name,

            "Real_Images":
                int(real["Valid_Images"]),

            "Synthetic_Images":
                int(
                    synthetic[
                        "Valid_Images"
                    ]
                ),

            "Real_Mean_Pixel":
                real_mean,

            "Synthetic_Mean_Pixel":
                synthetic_mean,

            "Mean_Difference":
                mean_difference,

            "Mean_Similarity":
                mean_similarity,

            "Real_Pixel_Std":
                real_std,

            "Synthetic_Pixel_Std":
                synthetic_std,

            "Std_Difference":
                std_difference,

            "Std_Similarity":
                std_similarity,

            "Distribution_Score":
                distribution_score
        })

    result = pd.DataFrame(rows)

    return result


# ============================================================
# CREATE CLASS COUNT COMPARISON
# ============================================================

def create_class_comparison(
    real_df,
    synthetic_df
):

    rows = []

    for class_name in CLASS_NAMES:

        real_rows = real_df[
            real_df["Class"] == class_name
        ]

        synthetic_rows = synthetic_df[
            synthetic_df["Class"] == class_name
        ]

        real_count = 0
        synthetic_count = 0

        if not real_rows.empty:

            real_count = int(
                real_rows.iloc[0][
                    "Valid_Images"
                ]
            )

        if not synthetic_rows.empty:

            synthetic_count = int(
                synthetic_rows.iloc[0][
                    "Valid_Images"
                ]
            )

        rows.append({

            "Class":
                class_name,

            "Real_Training_Images":
                real_count,

            "Synthetic_Training_Images":
                synthetic_count,

            "Synthetic_to_Real_Ratio":
                (
                    synthetic_count /
                    real_count
                    if real_count > 0
                    else 0
                )
        })

    return pd.DataFrame(rows)


# ============================================================
# CREATE PIXEL STATISTICS TABLE
# ============================================================

def create_pixel_statistics(
    real_df,
    synthetic_df
):

    rows = []

    for class_name in CLASS_NAMES:

        real_rows = real_df[
            real_df["Class"] == class_name
        ]

        synthetic_rows = synthetic_df[
            synthetic_df["Class"] == class_name
        ]

        if real_rows.empty:
            continue

        if synthetic_rows.empty:
            continue

        real = real_rows.iloc[0]
        synthetic = synthetic_rows.iloc[0]

        rows.append({

            "Class":
                class_name,

            "Real_Mean_Pixel":
                real["Mean_Pixel"],

            "Synthetic_Mean_Pixel":
                synthetic["Mean_Pixel"],

            "Real_Pixel_Std":
                real["Pixel_Std"],

            "Synthetic_Pixel_Std":
                synthetic["Pixel_Std"],

            "Mean_Difference":
                abs(
                    real["Mean_Pixel"]
                    -
                    synthetic["Mean_Pixel"]
                ),

            "Std_Difference":
                abs(
                    real["Pixel_Std"]
                    -
                    synthetic["Pixel_Std"]
                )
        })

    return pd.DataFrame(rows)


# ============================================================
# SAVE QUALITY SUMMARY
# ============================================================

def save_quality_summary(
    real_df,
    synthetic_df,
    similarity_df
):

    real_total = int(
        real_df["Valid_Images"].sum()
    )

    synthetic_total = int(
        synthetic_df["Valid_Images"].sum()
    )

    overall_score = 0.0

    if not similarity_df.empty:

        overall_score = (
            similarity_df[
                "Distribution_Score"
            ].mean()
            * 100
        )

    mean_similarity = 0.0

    if not similarity_df.empty:

        mean_similarity = (
            similarity_df[
                "Mean_Similarity"
            ].mean()
            * 100
        )

    std_similarity = 0.0

    if not similarity_df.empty:

        std_similarity = (
            similarity_df[
                "Std_Similarity"
            ].mean()
            * 100
        )

    summary = pd.DataFrame([{

        "Real_Training_Images":
            real_total,

        "Synthetic_Training_Images":
            synthetic_total,

        "Number_of_Classes":
            len(CLASS_NAMES),

        "Mean_Similarity_Percent":
            mean_similarity,

        "Std_Similarity_Percent":
            std_similarity,

        "Overall_Distribution_Score":
            overall_score,

        "Evaluation_Type":
            "Pixel-level statistical comparison",

        "Testing_Data_Used":
            "No"
    }])

    path = (
        OUTPUT_DIR
        / "training_quality_summary.csv"
    )

    summary.to_csv(
        path,
        index=False
    )

    print(
        f"Quality summary saved:\n{path}"
    )

    return overall_score


# ============================================================
# CREATE BAR CHART
# ============================================================

def create_mean_comparison(
    similarity_df
):

    if similarity_df.empty:
        return

    plt.figure(
        figsize=(10, 6)
    )

    x = np.arange(
        len(similarity_df)
    )

    width = 0.35

    plt.bar(
        x - width / 2,
        similarity_df[
            "Real_Mean_Pixel"
        ],
        width,
        label="Real"
    )

    plt.bar(
        x + width / 2,
        similarity_df[
            "Synthetic_Mean_Pixel"
        ],
        width,
        label="Synthetic"
    )

    plt.xticks(
        x,
        similarity_df["Class"]
    )

    plt.ylabel(
        "Mean Pixel Value"
    )

    plt.xlabel(
        "MRI Class"
    )

    plt.title(
        "Real vs Synthetic MRI Training - Mean Pixel"
    )

    plt.legend()

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "training_mean_comparison.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Mean comparison saved:\n{path}"
    )


# ============================================================
# CREATE STD COMPARISON
# ============================================================

def create_std_comparison(
    similarity_df
):

    if similarity_df.empty:
        return

    plt.figure(
        figsize=(10, 6)
    )

    x = np.arange(
        len(similarity_df)
    )

    width = 0.35

    plt.bar(
        x - width / 2,
        similarity_df[
            "Real_Pixel_Std"
        ],
        width,
        label="Real"
    )

    plt.bar(
        x + width / 2,
        similarity_df[
            "Synthetic_Pixel_Std"
        ],
        width,
        label="Synthetic"
    )

    plt.xticks(
        x,
        similarity_df["Class"]
    )

    plt.ylabel(
        "Pixel Standard Deviation"
    )

    plt.xlabel(
        "MRI Class"
    )

    plt.title(
        "Real vs Synthetic MRI Training - Pixel Std"
    )

    plt.legend()

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "training_std_comparison.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"STD comparison saved:\n{path}"
    )


# ============================================================
# CREATE PIXEL DISTRIBUTION
# ============================================================

def create_pixel_distribution(
    real_df,
    synthetic_df
):

    if real_df.empty:
        return

    if synthetic_df.empty:
        return

    classes = []

    real_means = []

    synthetic_means = []

    for class_name in CLASS_NAMES:

        real_rows = real_df[
            real_df["Class"] == class_name
        ]

        synthetic_rows = synthetic_df[
            synthetic_df["Class"] == class_name
        ]

        if real_rows.empty:
            continue

        if synthetic_rows.empty:
            continue

        classes.append(
            class_name
        )

        real_means.append(
            float(
                real_rows.iloc[0][
                    "Mean_Pixel"
                ]
            )
        )

        synthetic_means.append(
            float(
                synthetic_rows.iloc[0][
                    "Mean_Pixel"
                ]
            )
        )

    if not classes:
        return

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        classes,
        real_means,
        marker="o",
        label="Real"
    )

    plt.plot(
        classes,
        synthetic_means,
        marker="o",
        label="Synthetic"
    )

    plt.xlabel(
        "MRI Class"
    )

    plt.ylabel(
        "Mean Pixel Value"
    )

    plt.title(
        "MRI Training Pixel Distribution Comparison"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "training_pixel_distribution.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Pixel distribution saved:\n{path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Analyze REAL Training
    # --------------------------------------------------------

    real_df = analyze_dataset(
        REAL_TRAINING_DIR,
        "Real Training"
    )

    # --------------------------------------------------------
    # 2. Analyze SYNTHETIC Training
    # --------------------------------------------------------

    synthetic_df = analyze_dataset(
        SYNTHETIC_TRAINING_DIR,
        "Synthetic Training"
    )

    # --------------------------------------------------------
    # 3. Save combined quality data
    # --------------------------------------------------------

    quality_df = pd.concat(
        [
            real_df,
            synthetic_df
        ],
        ignore_index=True
    )

    quality_path = (
        OUTPUT_DIR
        / "training_pixel_statistics.csv"
    )

    quality_df.to_csv(
        quality_path,
        index=False
    )

    print()
    print(
        f"Pixel statistics saved:\n"
        f"{quality_path}"
    )

    # --------------------------------------------------------
    # 4. Class count comparison
    # --------------------------------------------------------

    class_comparison = (
        create_class_comparison(
            real_df,
            synthetic_df
        )
    )

    class_path = (
        OUTPUT_DIR
        / "training_class_comparison.csv"
    )

    class_comparison.to_csv(
        class_path,
        index=False
    )

    print(
        f"Class comparison saved:\n"
        f"{class_path}"
    )

    # --------------------------------------------------------
    # 5. Similarity calculation
    # --------------------------------------------------------

    similarity_df = (
        calculate_similarity(
            real_df,
            synthetic_df
        )
    )

    # --------------------------------------------------------
    # 6. Save similarity into CSV
    # --------------------------------------------------------

    similarity_path = (
        OUTPUT_DIR
        / "training_similarity_scores.csv"
    )

    similarity_df.to_csv(
        similarity_path,
        index=False
    )

    print(
        f"Similarity scores saved:\n"
        f"{similarity_path}"
    )

    # --------------------------------------------------------
    # 7. Overall summary
    # --------------------------------------------------------

    overall_score = (
        save_quality_summary(
            real_df,
            synthetic_df,
            similarity_df
        )
    )

    # --------------------------------------------------------
    # 8. Graphs
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("             CREATING EVALUATION GRAPHS")
    print("=" * 70)
    print()

    create_mean_comparison(
        similarity_df
    )

    create_std_comparison(
        similarity_df
    )

    create_pixel_distribution(
        real_df,
        synthetic_df
    )

    # --------------------------------------------------------
    # 9. Display final table
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("       MRI TRAINING SYNTHETIC EVALUATION SUMMARY")
    print("=" * 70)
    print()

    if not similarity_df.empty:

        display_df = similarity_df[
            [
                "Class",
                "Real_Images",
                "Synthetic_Images",
                "Mean_Similarity",
                "Std_Similarity",
                "Distribution_Score"
            ]
        ].copy()

        display_df[
            "Mean_Similarity"
        ] = (
            display_df[
                "Mean_Similarity"
            ] * 100
        )

        display_df[
            "Std_Similarity"
        ] = (
            display_df[
                "Std_Similarity"
            ] * 100
        )

        display_df[
            "Distribution_Score"
        ] = (
            display_df[
                "Distribution_Score"
            ] * 100
        )

        print(
            display_df.to_string(
                index=False
            )
        )

    print()
    print("-" * 70)

    print(
        f"Overall Distribution Score: "
        f"{overall_score:.2f}%"
    )

    print()

    # --------------------------------------------------------
    # 10. Interpretation
    # --------------------------------------------------------

    if overall_score >= 90:

        print(
            "Interpretation: Excellent "
            "pixel-level similarity between "
            "real and synthetic training data."
        )

    elif overall_score >= 75:

        print(
            "Interpretation: Good "
            "pixel-level similarity between "
            "real and synthetic training data."
        )

    elif overall_score >= 60:

        print(
            "Interpretation: Moderate "
            "pixel-level similarity. "
            "Further GAN improvement may be useful."
        )

    else:

        print(
            "Interpretation: Low "
            "pixel-level similarity. "
            "Synthetic data quality requires improvement."
        )

    print()

    print("=" * 70)
    print(
        "       MRI TRAINING SYNTHETIC EVALUATION COMPLETED"
    )
    print("=" * 70)
    print()

    print("Results saved to:")
    print(OUTPUT_DIR)
    print()

    print("Generated files:")
    print("  1. training_quality_summary.csv")
    print("  2. training_class_comparison.csv")
    print("  3. training_pixel_statistics.csv")
    print("  4. training_similarity_scores.csv")
    print("  5. training_pixel_distribution.png")
    print("  6. training_mean_comparison.png")
    print("  7. training_std_comparison.png")
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
"""
MRI TRAINING SYNTHETIC VISUALIZATION
====================================

Purpose:
    Visually compare REAL MRI TRAINING images with
    SYNTHETIC MRI TRAINING images.

IMPORTANT:
    - Real Training data is used only for visualization.
    - Synthetic Training data is used only for visualization.
    - Real Testing data is NEVER used.
    - Real Testing data is NEVER modified.

Input:
    data/imaging/MRI/Training
    outputs/mri/training_synthetic/synthetic

Previous evaluation:
    outputs/mri/training_synthetic/evaluation/

Output:
    outputs/mri/training_synthetic/visualization/

Generated:
    1. real_vs_synthetic_training_grid.png
    2. real_training_grid.png
    3. synthetic_training_grid.png
    4. training_class_pixel_comparison.png
    5. training_distribution_comparison.png
    6. visualization_summary.csv
"""

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

EVALUATION_DIR = (
    BASE_DIR
    / "outputs"
    / "mri"
    / "training_synthetic"
    / "evaluation"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "mri"
    / "training_synthetic"
    / "visualization"
)

IMAGE_SIZE = 64

SAMPLES_PER_CLASS = 8

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
print("       MRI TRAINING SYNTHETIC VISUALIZATION")
print("=" * 70)
print()

print("Real Training:")
print(REAL_TRAINING_DIR)
print()

print("Synthetic Training:")
print(SYNTHETIC_TRAINING_DIR)
print()

print("Output:")
print(OUTPUT_DIR)
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
# LOAD IMAGE
# ============================================================

def load_image(image_path):

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

        return array

    except Exception:

        return None


# ============================================================
# SAMPLE IMAGES
# ============================================================

def sample_images(
    directory,
    class_name,
    count
):

    class_dir = (
        directory / class_name
    )

    files = get_image_files(
        class_dir
    )

    if not files:

        return []

    if len(files) <= count:

        selected = files

    else:

        selected = random.sample(
            files,
            count
        )

    images = []

    for image_path in selected:

        image = load_image(
            image_path
        )

        if image is not None:

            images.append(
                image
            )

    return images


# ============================================================
# CREATE REAL VS SYNTHETIC GRID
# ============================================================

def create_comparison_grid():

    print("=" * 70)
    print("       CREATING REAL VS SYNTHETIC IMAGE GRID")
    print("=" * 70)
    print()

    fig, axes = plt.subplots(
        len(CLASS_NAMES) * 2,
        SAMPLES_PER_CLASS,
        figsize=(16, 12)
    )

    row_index = 0

    for class_name in CLASS_NAMES:

        real_images = sample_images(
            REAL_TRAINING_DIR,
            class_name,
            SAMPLES_PER_CLASS
        )

        synthetic_images = sample_images(
            SYNTHETIC_TRAINING_DIR,
            class_name,
            SAMPLES_PER_CLASS
        )

        # ----------------------------------------------------
        # REAL ROW
        # ----------------------------------------------------

        for col in range(
            SAMPLES_PER_CLASS
        ):

            ax = axes[
                row_index,
                col
            ]

            ax.axis("off")

            if col < len(real_images):

                ax.imshow(
                    real_images[col],
                    cmap="gray",
                    vmin=0,
                    vmax=1
                )

            if col == 0:

                ax.set_ylabel(
                    f"{class_name}\nREAL",
                    fontsize=10
                )

        row_index += 1

        # ----------------------------------------------------
        # SYNTHETIC ROW
        # ----------------------------------------------------

        for col in range(
            SAMPLES_PER_CLASS
        ):

            ax = axes[
                row_index,
                col
            ]

            ax.axis("off")

            if col < len(synthetic_images):

                ax.imshow(
                    synthetic_images[col],
                    cmap="gray",
                    vmin=0,
                    vmax=1
                )

            if col == 0:

                ax.set_ylabel(
                    f"{class_name}\nSYNTHETIC",
                    fontsize=10
                )

        row_index += 1

    fig.suptitle(
        "Real vs Synthetic MRI Training Images",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.96
        ]
    )

    output_path = (
        OUTPUT_DIR
        / "real_vs_synthetic_training_grid.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Comparison grid saved:\n"
        f"{output_path}"
    )

    print()


# ============================================================
# CREATE REAL TRAINING GRID
# ============================================================

def create_real_grid():

    print("Creating real training image grid...")

    fig, axes = plt.subplots(
        len(CLASS_NAMES),
        SAMPLES_PER_CLASS,
        figsize=(16, 8)
    )

    for row, class_name in enumerate(
        CLASS_NAMES
    ):

        images = sample_images(
            REAL_TRAINING_DIR,
            class_name,
            SAMPLES_PER_CLASS
        )

        for col in range(
            SAMPLES_PER_CLASS
        ):

            ax = axes[
                row,
                col
            ]

            ax.axis("off")

            if col < len(images):

                ax.imshow(
                    images[col],
                    cmap="gray",
                    vmin=0,
                    vmax=1
                )

            if col == 0:

                ax.set_ylabel(
                    class_name,
                    fontsize=11
                )

    fig.suptitle(
        "Real MRI Training Dataset Samples",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.95
        ]
    )

    output_path = (
        OUTPUT_DIR
        / "real_training_grid.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Real grid saved:\n"
        f"{output_path}"
    )

    print()


# ============================================================
# CREATE SYNTHETIC TRAINING GRID
# ============================================================

def create_synthetic_grid():

    print(
        "Creating synthetic training image grid..."
    )

    fig, axes = plt.subplots(
        len(CLASS_NAMES),
        SAMPLES_PER_CLASS,
        figsize=(16, 8)
    )

    for row, class_name in enumerate(
        CLASS_NAMES
    ):

        images = sample_images(
            SYNTHETIC_TRAINING_DIR,
            class_name,
            SAMPLES_PER_CLASS
        )

        for col in range(
            SAMPLES_PER_CLASS
        ):

            ax = axes[
                row,
                col
            ]

            ax.axis("off")

            if col < len(images):

                ax.imshow(
                    images[col],
                    cmap="gray",
                    vmin=0,
                    vmax=1
                )

            if col == 0:

                ax.set_ylabel(
                    class_name,
                    fontsize=11
                )

    fig.suptitle(
        "Synthetic MRI Training Dataset Samples",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.95
        ]
    )

    output_path = (
        OUTPUT_DIR
        / "synthetic_training_grid.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Synthetic grid saved:\n"
        f"{output_path}"
    )

    print()


# ============================================================
# LOAD PREVIOUS EVALUATION
# ============================================================

def load_evaluation_results():

    path = (
        EVALUATION_DIR
        / "training_similarity_scores.csv"
    )

    if not path.exists():

        print(
            "WARNING: Previous similarity file "
            "not found."
        )

        return None

    try:

        df = pd.read_csv(
            path
        )

        return df

    except Exception as error:

        print(
            f"WARNING: Could not read:\n"
            f"{path}"
        )

        print(error)

        return None


# ============================================================
# CLASS PIXEL COMPARISON
# ============================================================

def create_pixel_comparison(
    similarity_df
):

    if similarity_df is None:

        return

    if similarity_df.empty:

        return

    print(
        "Creating class pixel comparison..."
    )

    classes = (
        similarity_df["Class"]
        .tolist()
    )

    real_mean = (
        similarity_df[
            "Real_Mean_Pixel"
        ]
        .tolist()
    )

    synthetic_mean = (
        similarity_df[
            "Synthetic_Mean_Pixel"
        ]
        .tolist()
    )

    real_std = (
        similarity_df[
            "Real_Pixel_Std"
        ]
        .tolist()
    )

    synthetic_std = (
        similarity_df[
            "Synthetic_Pixel_Std"
        ]
        .tolist()
    )

    x = np.arange(
        len(classes)
    )

    width = 0.35

    # --------------------------------------------------------
    # Mean
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        x - width / 2,
        real_mean,
        width,
        label="Real"
    )

    plt.bar(
        x + width / 2,
        synthetic_mean,
        width,
        label="Synthetic"
    )

    plt.xticks(
        x,
        classes
    )

    plt.xlabel(
        "MRI Class"
    )

    plt.ylabel(
        "Mean Pixel Value"
    )

    plt.title(
        "Real vs Synthetic MRI Training Pixel Mean",
        fontweight="bold"
    )

    plt.legend()

    plt.tight_layout()

    mean_path = (
        OUTPUT_DIR
        / "training_class_pixel_comparison.png"
    )

    plt.savefig(
        mean_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Pixel comparison saved:\n"
        f"{mean_path}"
    )


# ============================================================
# DISTRIBUTION SCORE GRAPH
# ============================================================

def create_distribution_comparison(
    similarity_df
):

    if similarity_df is None:

        return

    if similarity_df.empty:

        return

    print(
        "Creating distribution score graph..."
    )

    classes = (
        similarity_df["Class"]
        .tolist()
    )

    scores = (
        similarity_df[
            "Distribution_Score"
        ]
        .to_numpy()
        * 100
    )

    plt.figure(
        figsize=(10, 6)
    )

    bars = plt.bar(
        classes,
        scores
    )

    plt.axhline(
        90,
        linestyle="--",
        label="90% Reference"
    )

    plt.xlabel(
        "MRI Class"
    )

    plt.ylabel(
        "Distribution Similarity (%)"
    )

    plt.title(
        "Synthetic MRI Training Distribution Similarity",
        fontweight="bold"
    )

    plt.ylim(
        0,
        105
    )

    plt.legend()

    # --------------------------------------------------------
    # Data labels
    # --------------------------------------------------------

    for bar, score in zip(
        bars,
        scores
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height()
            + 1,
            f"{score:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "training_distribution_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Distribution graph saved:\n"
        f"{output_path}"
    )


# ============================================================
# CREATE VISUALIZATION SUMMARY
# ============================================================

def create_summary(
    similarity_df
):

    rows = []

    if similarity_df is not None:

        for _, row in similarity_df.iterrows():

            rows.append({

                "Class":
                    row["Class"],

                "Real_Training_Images":
                    row["Real_Images"],

                "Synthetic_Training_Images":
                    row["Synthetic_Images"],

                "Mean_Similarity_Percent":
                    row[
                        "Mean_Similarity"
                    ] * 100,

                "Std_Similarity_Percent":
                    row[
                        "Std_Similarity"
                    ] * 100,

                "Distribution_Score_Percent":
                    row[
                        "Distribution_Score"
                    ] * 100
            })

    if rows:

        df = pd.DataFrame(
            rows
        )

        overall_score = (
            df[
                "Distribution_Score_Percent"
            ].mean()
        )

        df["Overall_Score_Percent"] = (
            overall_score
        )

    else:

        df = pd.DataFrame()

    output_path = (
        OUTPUT_DIR
        / "visualization_summary.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Visualization summary saved:\n"
        f"{output_path}"
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Validate directories
    # --------------------------------------------------------

    if not REAL_TRAINING_DIR.exists():

        raise FileNotFoundError(
            f"\nReal Training directory not found:\n"
            f"{REAL_TRAINING_DIR}"
        )

    if not SYNTHETIC_TRAINING_DIR.exists():

        raise FileNotFoundError(
            f"\nSynthetic Training directory not found:\n"
            f"{SYNTHETIC_TRAINING_DIR}\n\n"
            "Run generate_training_synthetic.py first."
        )

    # --------------------------------------------------------
    # Count datasets
    # --------------------------------------------------------

    print("=" * 70)
    print("             DATASET VERIFICATION")
    print("=" * 70)
    print()

    real_total = 0
    synthetic_total = 0

    for class_name in CLASS_NAMES:

        real_files = get_image_files(
            REAL_TRAINING_DIR / class_name
        )

        synthetic_files = get_image_files(
            SYNTHETIC_TRAINING_DIR / class_name
        )

        real_count = len(
            real_files
        )

        synthetic_count = len(
            synthetic_files
        )

        real_total += real_count
        synthetic_total += synthetic_count

        print(
            f"{class_name:<15} "
            f"Real: {real_count:>5}   "
            f"Synthetic: {synthetic_count:>5}"
        )

    print()

    print(
        f"Total Real Training Images: "
        f"{real_total}"
    )

    print(
        f"Total Synthetic Training Images: "
        f"{synthetic_total}"
    )

    print()

    # --------------------------------------------------------
    # Create image grids
    # --------------------------------------------------------

    create_comparison_grid()

    create_real_grid()

    create_synthetic_grid()

    # --------------------------------------------------------
    # Load evaluation results
    # --------------------------------------------------------

    similarity_df = (
        load_evaluation_results()
    )

    # --------------------------------------------------------
    # Create graphs
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("             CREATING ANALYSIS GRAPHS")
    print("=" * 70)
    print()

    create_pixel_comparison(
        similarity_df
    )

    create_distribution_comparison(
        similarity_df
    )

    # --------------------------------------------------------
    # Create summary
    # --------------------------------------------------------

    summary_df = create_summary(
        similarity_df
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("       MRI TRAINING VISUALIZATION COMPLETED")
    print("=" * 70)
    print()

    if (
        similarity_df is not None
        and not similarity_df.empty
    ):

        overall_score = (
            similarity_df[
                "Distribution_Score"
            ].mean()
            * 100
        )

        print(
            f"Overall Distribution Score: "
            f"{overall_score:.2f}%"
        )

        print()

        if overall_score >= 90:

            print(
                "Interpretation: Excellent "
                "pixel-level similarity."
            )

        elif overall_score >= 75:

            print(
                "Interpretation: Good "
                "pixel-level similarity."
            )

        elif overall_score >= 60:

            print(
                "Interpretation: Moderate "
                "pixel-level similarity."
            )

        else:

            print(
                "Interpretation: Low "
                "pixel-level similarity."
            )

    print()

    print(
        "Visualization results:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    print("Generated files:")

    print(
        "  1. real_vs_synthetic_training_grid.png"
    )

    print(
        "  2. real_training_grid.png"
    )

    print(
        "  3. synthetic_training_grid.png"
    )

    print(
        "  4. training_class_pixel_comparison.png"
    )

    print(
        "  5. training_distribution_comparison.png"
    )

    print(
        "  6. visualization_summary.csv"
    )

    print()

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
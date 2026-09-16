import os
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# =========================================================
# PATHS
# =========================================================

REAL_DIR = "data/imaging/MRI/Training"

SYNTHETIC_DIR = "outputs/mri/synthetic"

OUTPUT_DIR = "outputs/mri/evaluation"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# SETTINGS
# =========================================================

CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

SAMPLES_PER_CLASS = 5

IMAGE_SIZE = (128, 128)

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# =========================================================
# IMAGE FILE FINDER
# =========================================================

def get_images(folder):

    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp"
    ]

    files = []

    for extension in extensions:

        files.extend(
            glob.glob(
                os.path.join(
                    folder,
                    extension
                )
            )
        )

    return files


# =========================================================
# LOAD IMAGE
# =========================================================

def load_image(path):

    try:

        image = Image.open(path).convert("L")

        image = image.resize(
            IMAGE_SIZE
        )

        return np.array(image) / 255.0

    except Exception:

        return None


# =========================================================
# GET CLASS IMAGES
# =========================================================

def get_class_images(
    base_dir,
    class_name
):

    folder = os.path.join(
        base_dir,
        class_name
    )

    if not os.path.exists(folder):

        return []

    return get_images(folder)


# =========================================================
# DATASET INSPECTION
# =========================================================

def inspect_dataset():

    print()
    print("=" * 70)
    print("             MRI VISUAL QUALITY INSPECTION")
    print("=" * 70)

    results = []

    for class_name in CLASSES:

        real_images = get_class_images(
            REAL_DIR,
            class_name
        )

        synthetic_images = get_class_images(
            SYNTHETIC_DIR,
            class_name
        )

        print()
        print(
            f"{class_name:<15} "
            f"Real: {len(real_images):>5,}   "
            f"Synthetic: {len(synthetic_images):>5,}"
        )

        # -------------------------------------------------
        # CHECK SYNTHETIC IMAGES
        # -------------------------------------------------

        valid = 0
        corrupted = 0

        for path in synthetic_images:

            image = load_image(path)

            if image is None:

                corrupted += 1

            else:

                valid += 1

        results.append(
            {
                "Class": class_name,
                "Real Images": len(real_images),
                "Synthetic Images": len(synthetic_images),
                "Valid Synthetic": valid,
                "Corrupted Synthetic": corrupted
            }
        )

    result_df = pd.DataFrame(
        results
    )

    print()
    print("-" * 70)

    print(
        result_df.to_string(
            index=False
        )
    )

    result_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "mri_visual_inspection.csv"
        ),
        index=False
    )

    return result_df


# =========================================================
# REAL VS SYNTHETIC GRID
# =========================================================

def create_comparison_grid():

    print()
    print(
        "Creating Real vs Synthetic comparison..."
    )

    rows = len(CLASSES)

    columns = SAMPLES_PER_CLASS * 2

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(18, 10)
    )

    for row, class_name in enumerate(
        CLASSES
    ):

        real_images = get_class_images(
            REAL_DIR,
            class_name
        )

        synthetic_images = get_class_images(
            SYNTHETIC_DIR,
            class_name
        )

        # -------------------------------------------------
        # RANDOM SAMPLE
        # -------------------------------------------------

        real_sample = random.sample(
            real_images,
            min(
                SAMPLES_PER_CLASS,
                len(real_images)
            )
        )

        synthetic_sample = random.sample(
            synthetic_images,
            min(
                SAMPLES_PER_CLASS,
                len(synthetic_images)
            )
        )

        # -------------------------------------------------
        # REAL IMAGES
        # -------------------------------------------------

        for i in range(
            SAMPLES_PER_CLASS
        ):

            ax = axes[
                row,
                i
            ]

            ax.axis(
                "off"
            )

            if i < len(real_sample):

                image = load_image(
                    real_sample[i]
                )

                if image is not None:

                    ax.imshow(
                        image,
                        cmap="gray"
                    )

            if i == 0:

                ax.set_title(
                    f"{class_name.upper()}\nREAL",
                    fontsize=10,
                    fontweight="bold"
                )

        # -------------------------------------------------
        # SYNTHETIC IMAGES
        # -------------------------------------------------

        for i in range(
            SAMPLES_PER_CLASS
        ):

            column = (
                SAMPLES_PER_CLASS
                + i
            )

            ax = axes[
                row,
                column
            ]

            ax.axis(
                "off"
            )

            if i < len(synthetic_sample):

                image = load_image(
                    synthetic_sample[i]
                )

                if image is not None:

                    ax.imshow(
                        image,
                        cmap="gray"
                    )

            if i == 0:

                ax.set_title(
                    f"{class_name.upper()}\nSYNTHETIC",
                    fontsize=10,
                    fontweight="bold"
                )

    fig.suptitle(
        "MRI Real vs Synthetic Visual Comparison",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "real_vs_synthetic_mri_samples.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Comparison saved to:\n{output_path}"
    )


# =========================================================
# PIXEL STATISTICS
# =========================================================

def calculate_pixel_statistics():

    print()
    print(
        "Calculating pixel statistics..."
    )

    results = []

    for dataset_name, base_dir in [
        ("Real", REAL_DIR),
        ("Synthetic", SYNTHETIC_DIR)
    ]:

        for class_name in CLASSES:

            images = get_class_images(
                base_dir,
                class_name
            )

            means = []

            stds = []

            for path in images:

                image = load_image(
                    path
                )

                if image is not None:

                    means.append(
                        image.mean()
                    )

                    stds.append(
                        image.std()
                    )

            if means:

                results.append(
                    {
                        "Dataset": dataset_name,
                        "Class": class_name,
                        "Mean Pixel": np.mean(
                            means
                        ),
                        "Pixel Std": np.mean(
                            stds
                        ),
                        "Images Used": len(
                            means
                        )
                    }
                )

    statistics_df = pd.DataFrame(
        results
    )

    print()

    print(
        statistics_df.to_string(
            index=False
        )
    )

    statistics_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "mri_pixel_statistics.csv"
        ),
        index=False
    )

    return statistics_df


# =========================================================
# PIXEL DISTRIBUTION PLOT
# =========================================================

def create_pixel_distribution():

    print()
    print(
        "Creating pixel distribution comparison..."
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8)
    )

    axes = axes.flatten()

    for index, class_name in enumerate(
        CLASSES
    ):

        ax = axes[index]

        # ---------------------------------------------
        # REAL
        # ---------------------------------------------

        real_images = get_class_images(
            REAL_DIR,
            class_name
        )

        real_pixels = []

        for path in real_images[
            :100
        ]:

            image = load_image(
                path
            )

            if image is not None:

                real_pixels.extend(
                    image.flatten()[
                        ::20
                    ]
                )

        # ---------------------------------------------
        # SYNTHETIC
        # ---------------------------------------------

        synthetic_images = get_class_images(
            SYNTHETIC_DIR,
            class_name
        )

        synthetic_pixels = []

        for path in synthetic_images[
            :100
        ]:

            image = load_image(
                path
            )

            if image is not None:

                synthetic_pixels.extend(
                    image.flatten()[
                        ::20
                    ]
                )

        # ---------------------------------------------
        # PLOT
        # ---------------------------------------------

        if real_pixels:

            ax.hist(
                real_pixels,
                bins=40,
                alpha=0.5,
                label="Real"
            )

        if synthetic_pixels:

            ax.hist(
                synthetic_pixels,
                bins=40,
                alpha=0.5,
                label="Synthetic"
            )

        ax.set_title(
            class_name.upper()
        )

        ax.set_xlabel(
            "Pixel Intensity"
        )

        ax.set_ylabel(
            "Frequency"
        )

        ax.legend()

    fig.suptitle(
        "MRI Pixel Distribution — Real vs Synthetic",
        fontsize=15,
        fontweight="bold"
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "mri_pixel_distribution_visual.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Pixel distribution saved to:\n{output_path}"
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("             MRI VISUAL QUALITY EVALUATION")
    print("=" * 70)

    # -----------------------------------------------------
    # STEP 1
    # -----------------------------------------------------

    inspection_df = inspect_dataset()

    # -----------------------------------------------------
    # STEP 2
    # -----------------------------------------------------

    create_comparison_grid()

    # -----------------------------------------------------
    # STEP 3
    # -----------------------------------------------------

    calculate_pixel_statistics()

    # -----------------------------------------------------
    # STEP 4
    # -----------------------------------------------------

    create_pixel_distribution()

    print()
    print("=" * 70)
    print("       MRI VISUAL EVALUATION COMPLETED")
    print("=" * 70)

    print()
    print(
        "Results saved to:"
    )

    print(
        os.path.abspath(
            OUTPUT_DIR
        )
    )

    print()
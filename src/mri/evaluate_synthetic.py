import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

REAL_DIR = os.path.join(
    BASE_DIR,
    "data",
    "imaging",
    "MRI",
    "Training"
)

SYNTHETIC_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "mri",
    "synthetic"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "mri",
    "evaluation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# CLASSES
# =========================================================

CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# =========================================================
# IMAGE LOADER
# =========================================================

def get_images(directory, class_name):

    class_dir = os.path.join(
        directory,
        class_name
    )

    if not os.path.exists(class_dir):

        return []

    files = []

    for file in os.listdir(class_dir):

        if file.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp"
            )
        ):

            files.append(
                os.path.join(
                    class_dir,
                    file
                )
            )

    return files


# =========================================================
# DATASET ANALYSIS
# =========================================================

def analyse_dataset(
    base_dir,
    dataset_name
):

    print()
    print(
        f"========== {dataset_name.upper()} =========="
    )

    results = []

    all_pixels = []

    for class_name in CLASSES:

        files = get_images(
            base_dir,
            class_name
        )

        print(
            f"{class_name:<15}: "
            f"{len(files):,} images"
        )

        pixel_means = []
        pixel_stds = []
        image_sizes = []

        for file in files:

            image = cv2.imread(
                file,
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:

                continue

            image = image.astype(
                np.float32
            ) / 255.0

            pixel_means.append(
                image.mean()
            )

            pixel_stds.append(
                image.std()
            )

            image_sizes.append(
                image.shape
            )

            # Sample pixels for distribution
            if len(all_pixels) < 1000000:

                sampled = image.flatten()

                if len(sampled) > 1000:

                    sampled = np.random.choice(
                        sampled,
                        1000,
                        replace=False
                    )

                all_pixels.extend(
                    sampled.tolist()
                )


        if pixel_means:

            results.append(
                {
                    "Dataset": dataset_name,
                    "Class": class_name,
                    "Images": len(files),
                    "Mean Pixel": np.mean(
                        pixel_means
                    ),
                    "Pixel Std": np.mean(
                        pixel_stds
                    ),
                    "Image Size": str(
                        image_sizes[0]
                    )
                }
            )

    return (
        pd.DataFrame(results),
        np.array(all_pixels)
    )


# =========================================================
# REAL DATA
# =========================================================

real_results, real_pixels = analyse_dataset(
    REAL_DIR,
    "Real"
)


# =========================================================
# SYNTHETIC DATA
# =========================================================

synthetic_results, synthetic_pixels = analyse_dataset(
    SYNTHETIC_DIR,
    "Synthetic"
)


# =========================================================
# COMBINE RESULTS
# =========================================================

results = pd.concat(
    [
        real_results,
        synthetic_results
    ],
    ignore_index=True
)


# =========================================================
# DISPLAY
# =========================================================

print()
print("=" * 70)
print("MRI DATASET QUALITY SUMMARY")
print("=" * 70)

print()

print(
    results.to_string(
        index=False
    )
)


# =========================================================
# SAVE RESULTS
# =========================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "mri_quality_summary.csv"
)

results.to_csv(
    results_path,
    index=False
)


# =========================================================
# CLASS COUNT COMPARISON
# =========================================================

count_table = results[
    [
        "Dataset",
        "Class",
        "Images"
    ]
]

pivot_counts = count_table.pivot(
    index="Class",
    columns="Dataset",
    values="Images"
)


print()
print("=" * 70)
print("CLASS-WISE IMAGE COUNT")
print("=" * 70)

print()

print(
    pivot_counts
)


# =========================================================
# PIXEL MEAN COMPARISON
# =========================================================

mean_table = results[
    [
        "Dataset",
        "Class",
        "Mean Pixel"
    ]
]

pivot_mean = mean_table.pivot(
    index="Class",
    columns="Dataset",
    values="Mean Pixel"
)


print()
print("=" * 70)
print("CLASS-WISE PIXEL MEAN")
print("=" * 70)

print()

print(
    pivot_mean
)


# =========================================================
# PIXEL STD COMPARISON
# =========================================================

std_table = results[
    [
        "Dataset",
        "Class",
        "Pixel Std"
    ]
]

pivot_std = std_table.pivot(
    index="Class",
    columns="Dataset",
    values="Pixel Std"
)


print()
print("=" * 70)
print("CLASS-WISE PIXEL STANDARD DEVIATION")
print("=" * 70)

print()

print(
    pivot_std
)


# =========================================================
# PIXEL DISTRIBUTION
# =========================================================

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    real_pixels,
    bins=50,
    alpha=0.5,
    label="Real"
)

plt.hist(
    synthetic_pixels,
    bins=50,
    alpha=0.5,
    label="Synthetic"
)

plt.xlabel(
    "Normalized Pixel Intensity"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Real vs Synthetic MRI Pixel Distribution"
)

plt.legend()

plt.tight_layout()

distribution_path = os.path.join(
    OUTPUT_DIR,
    "pixel_distribution.png"
)

plt.savefig(
    distribution_path,
    dpi=150
)

plt.close()


# =========================================================
# MEAN COMPARISON CHART
# =========================================================

chart_df = results.copy()

plt.figure(
    figsize=(10, 6)
)

for dataset in [
    "Real",
    "Synthetic"
]:

    data = chart_df[
        chart_df["Dataset"] == dataset
    ]

    plt.plot(
        data["Class"],
        data["Mean Pixel"],
        marker="o",
        label=dataset
    )

plt.xlabel(
    "MRI Class"
)

plt.ylabel(
    "Mean Pixel Intensity"
)

plt.title(
    "Real vs Synthetic MRI Mean Pixel Intensity"
)

plt.legend()

plt.tight_layout()

mean_chart_path = os.path.join(
    OUTPUT_DIR,
    "mean_pixel_comparison.png"
)

plt.savefig(
    mean_chart_path,
    dpi=150
)

plt.close()


# =========================================================
# FINAL
# =========================================================

print()
print("=" * 70)
print("MRI SYNTHETIC DATA EVALUATION COMPLETED")
print("=" * 70)

print()

print(
    "Results saved to:"
)

print(
    OUTPUT_DIR
)

print()

print(
    f"Summary CSV: {results_path}"
)

print(
    f"Pixel distribution: {distribution_path}"
)

print(
    f"Mean comparison: {mean_chart_path}"
)

print()
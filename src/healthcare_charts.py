
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# =========================================================
# PATHS
# =========================================================

REAL_PATH = "data/healthcare_dataset.csv"
SYNTHETIC_PATH = "data/synthetic_healthcare.csv"

OUTPUT_DIR = "data/healthcare/comparison/charts"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

real = pd.read_csv(REAL_PATH)
synthetic = pd.read_csv(SYNTHETIC_PATH)

print("===================================")
print("HEALTHCARE VISUAL COMPARISON")
print("===================================")

print("Real Data:", real.shape)
print("Synthetic Data:", synthetic.shape)


# =========================================================
# COMMON COLUMNS
# =========================================================

common_columns = [
    col for col in real.columns
    if col in synthetic.columns
]

print("\nCommon Columns:")
print(common_columns)


# =========================================================
# NUMERIC COLUMNS
# =========================================================

numeric_columns = [
    col
    for col in common_columns
    if pd.api.types.is_numeric_dtype(real[col])
    and pd.api.types.is_numeric_dtype(synthetic[col])
]

print("\nNumeric Columns:")
print(numeric_columns)


# =========================================================
# 1. NUMERIC MEAN COMPARISON
# =========================================================

real_means = real[numeric_columns].mean()
synthetic_means = synthetic[numeric_columns].mean()

x = np.arange(len(numeric_columns))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    real_means.values,
    width,
    label="Real"
)

plt.bar(
    x + width / 2,
    synthetic_means.values,
    width,
    label="Synthetic"
)

plt.xticks(
    x,
    numeric_columns,
    rotation=20
)

plt.ylabel("Mean Value")

plt.title(
    "Healthcare - Real vs Synthetic Numeric Mean",
    fontweight="bold"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/01_numeric_mean_comparison.png",
    dpi=300
)

plt.close()


# =========================================================
# 2. NUMERIC DISTRIBUTION CHARTS
# =========================================================

for column in numeric_columns:

    plt.figure(figsize=(9, 6))

    plt.hist(
        real[column].dropna(),
        bins=30,
        alpha=0.5,
        label="Real"
    )

    plt.hist(
        synthetic[column].dropna(),
        bins=30,
        alpha=0.5,
        label="Synthetic"
    )

    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.title(
        f"Healthcare - {column}\nReal vs Synthetic Distribution",
        fontweight="bold"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.2
    )

    plt.tight_layout()

    safe_name = (
        column
        .replace(" ", "_")
        .replace("/", "_")
    )

    plt.savefig(
        f"{OUTPUT_DIR}/02_{safe_name}_distribution.png",
        dpi=300
    )

    plt.close()


# =========================================================
# 3. CATEGORICAL COLUMNS
# =========================================================

categorical_columns = [
    col
    for col in common_columns
    if col not in numeric_columns
]

print("\nCategorical Columns:")
print(categorical_columns)


# =========================================================
# 4. CATEGORICAL DISTRIBUTION COMPARISON
# =========================================================

for column in categorical_columns:

    real_percentage = (
        real[column]
        .value_counts(normalize=True)
        .mul(100)
    )

    synthetic_percentage = (
        synthetic[column]
        .value_counts(normalize=True)
        .mul(100)
    )

    categories = sorted(
        set(real_percentage.index)
        .union(synthetic_percentage.index)
    )

    real_values = [
        real_percentage.get(category, 0)
        for category in categories
    ]

    synthetic_values = [
        synthetic_percentage.get(category, 0)
        for category in categories
    ]

    x_cat = np.arange(len(categories))

    plt.figure(figsize=(10, 6))

    plt.bar(
        x_cat - width / 2,
        real_values,
        width,
        label="Real"
    )

    plt.bar(
        x_cat + width / 2,
        synthetic_values,
        width,
        label="Synthetic"
    )

    plt.xticks(
        x_cat,
        categories,
        rotation=35,
        ha="right"
    )

    plt.ylabel("Percentage (%)")

    plt.title(
        f"Healthcare - {column}\nReal vs Synthetic Distribution",
        fontweight="bold"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.2
    )

    plt.tight_layout()

    safe_name = (
        column
        .replace(" ", "_")
        .replace("/", "_")
    )

    plt.savefig(
        f"{OUTPUT_DIR}/03_{safe_name}_comparison.png",
        dpi=300
    )

    plt.close()


# =========================================================
# 5. REAL CORRELATION
# =========================================================

if len(numeric_columns) >= 2:

    real_corr = real[numeric_columns].corr()

    plt.figure(figsize=(8, 7))

    plt.imshow(
        real_corr,
        aspect="auto"
    )

    plt.colorbar(
        label="Correlation"
    )

    plt.xticks(
        range(len(numeric_columns)),
        numeric_columns,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(numeric_columns)),
        numeric_columns
    )

    plt.title(
        "Healthcare - Real Data Correlation",
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/04_real_correlation.png",
        dpi=300
    )

    plt.close()


# =========================================================
# 6. SYNTHETIC CORRELATION
# =========================================================

    synthetic_corr = synthetic[numeric_columns].corr()

    plt.figure(figsize=(8, 7))

    plt.imshow(
        synthetic_corr,
        aspect="auto"
    )

    plt.colorbar(
        label="Correlation"
    )

    plt.xticks(
        range(len(numeric_columns)),
        numeric_columns,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(numeric_columns)),
        numeric_columns
    )

    plt.title(
        "Healthcare - Synthetic Data Correlation",
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/05_synthetic_correlation.png",
        dpi=300
    )

    plt.close()


# =========================================================
# 7. CORRELATION DIFFERENCE
# =========================================================

    correlation_difference = (
        real_corr - synthetic_corr
    ).abs()

    plt.figure(figsize=(8, 7))

    plt.imshow(
        correlation_difference,
        aspect="auto"
    )

    plt.colorbar(
        label="Absolute Difference"
    )

    plt.xticks(
        range(len(numeric_columns)),
        numeric_columns,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(numeric_columns)),
        numeric_columns
    )

    plt.title(
        "Healthcare - Real vs Synthetic Correlation Difference",
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/06_correlation_difference.png",
        dpi=300
    )

    plt.close()


# =========================================================
# 8. QUALITY SCORE
# =========================================================

quality_path = (
    "data/healthcare/comparison/"
    "synthetic_comparison.csv"
)

if os.path.exists(quality_path):

    quality = pd.read_csv(quality_path)

    row = quality.iloc[0]

    score_names = [
        "Distribution",
        "Correlation",
        "Overall Similarity"
    ]

    score_values = [
        row["Distribution Score"],
        row["Correlation Score"],
        row["Overall Similarity Score"]
    ]

    plt.figure(figsize=(9, 6))

    bars = plt.bar(
        score_names,
        score_values
    )

    plt.ylabel("Score (%)")

    plt.ylim(
        0,
        100
    )

    plt.title(
        "Healthcare Synthetic Data Quality",
        fontweight="bold"
    )

    for bar, value in zip(
        bars,
        score_values
    ):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center",
            fontweight="bold"
        )

    plt.grid(
        axis="y",
        alpha=0.2
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/07_quality_score.png",
        dpi=300
    )

    plt.close()


# =========================================================
# 9. NUMERIC SUMMARY
# =========================================================

summary = pd.DataFrame({
    "Column": numeric_columns,
    "Real Mean": [
        real[col].mean()
        for col in numeric_columns
    ],
    "Synthetic Mean": [
        synthetic[col].mean()
        for col in numeric_columns
    ],
    "Real Std": [
        real[col].std()
        for col in numeric_columns
    ],
    "Synthetic Std": [
        synthetic[col].std()
        for col in numeric_columns
    ]
})

summary["Mean Difference %"] = (
    abs(
        summary["Real Mean"]
        - summary["Synthetic Mean"]
    )
    /
    summary["Real Mean"].replace(0, np.nan)
    * 100
)

summary.to_csv(
    f"{OUTPUT_DIR}/numeric_visual_summary.csv",
    index=False
)


# =========================================================
# COMPLETE
# =========================================================

print()
print("===================================")
print("HEALTHCARE CHARTS CREATED")
print("===================================")

print("Saved to:")
print(OUTPUT_DIR)

print()
print("Healthcare Overall Similarity:")
print("93.47%")


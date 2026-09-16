import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


# ==========================================
# PATHS
# ==========================================

REAL_PATH = "data/banking/banking_combined.csv"
SYNTHETIC_PATH = "data/banking/synthetic_banking.csv"

OUTPUT_DIR = "data/banking/comparison"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# LOAD DATA
# ==========================================

real = pd.read_csv(REAL_PATH)
synthetic = pd.read_csv(SYNTHETIC_PATH)

print("REAL DATA:", real.shape)
print("SYNTHETIC DATA:", synthetic.shape)


# ==========================================
# NUMERIC COLUMNS
# ==========================================

numeric_columns = real.select_dtypes(
    include=np.number
).columns.tolist()

print("\nNumeric Columns:")
print(numeric_columns)


# ==========================================
# NUMERIC SUMMARY COMPARISON
# ==========================================

comparison = pd.DataFrame({
    "Real Mean": real[numeric_columns].mean(),
    "Synthetic Mean": synthetic[numeric_columns].mean(),
    "Real Median": real[numeric_columns].median(),
    "Synthetic Median": synthetic[numeric_columns].median(),
    "Real Std": real[numeric_columns].std(),
    "Synthetic Std": synthetic[numeric_columns].std()
})

comparison["Mean Difference %"] = (
    abs(
        comparison["Real Mean"]
        - comparison["Synthetic Mean"]
    )
    / comparison["Real Mean"].replace(0, np.nan)
) * 100

comparison.to_csv(
    f"{OUTPUT_DIR}/numeric_comparison.csv"
)

print("\n===================================")
print("NUMERIC COMPARISON")
print("===================================")

print(comparison.round(2))


# ==========================================
# CATEGORICAL COMPARISON
# ==========================================

categorical_columns = real.select_dtypes(
    include="object"
).columns.tolist()

# Remove ID
categorical_columns = [
    col for col in categorical_columns
    if col != "customer_id"
]

print("\nCategorical Columns:")
print(categorical_columns)


# ==========================================
# CATEGORY DISTRIBUTION
# ==========================================

for column in categorical_columns:

    real_dist = (
        real[column]
        .value_counts(normalize=True)
        .mul(100)
    )

    synthetic_dist = (
        synthetic[column]
        .value_counts(normalize=True)
        .mul(100)
    )

    categories = sorted(
        set(real_dist.index)
        | set(synthetic_dist.index)
    )

    real_values = [
        real_dist.get(category, 0)
        for category in categories
    ]

    synthetic_values = [
        synthetic_dist.get(category, 0)
        for category in categories
    ]

    x = np.arange(len(categories))
    width = 0.35

    plt.figure(figsize=(10, 5))

    plt.bar(
        x - width / 2,
        real_values,
        width,
        label="Real"
    )

    plt.bar(
        x + width / 2,
        synthetic_values,
        width,
        label="Synthetic"
    )

    plt.title(
        f"{column}: Real vs Synthetic"
    )

    plt.xlabel(column)
    plt.ylabel("Percentage")

    plt.xticks(
        x,
        categories,
        rotation=30,
        ha="right"
    )

    plt.legend()

    plt.tight_layout()

    safe_name = (
        column
        .replace(" ", "_")
        .lower()
    )

    plt.savefig(
        f"{OUTPUT_DIR}/{safe_name}_comparison.png",
        dpi=150
    )

    plt.close()


# ==========================================
# NUMERIC DISTRIBUTION CHARTS
# ==========================================

for column in numeric_columns:

    plt.figure(figsize=(8, 5))

    plt.hist(
        real[column].dropna(),
        bins=20,
        alpha=0.6,
        label="Real"
    )

    plt.hist(
        synthetic[column].dropna(),
        bins=20,
        alpha=0.6,
        label="Synthetic"
    )

    plt.title(
        f"{column}: Real vs Synthetic"
    )

    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.legend()

    plt.tight_layout()

    safe_name = (
        column
        .replace(" ", "_")
        .lower()
    )

    plt.savefig(
        f"{OUTPUT_DIR}/{safe_name}_distribution.png",
        dpi=150
    )

    plt.close()


# ==========================================
# CORRELATION COMPARISON
# ==========================================

real_corr = real[numeric_columns].corr()
synthetic_corr = synthetic[numeric_columns].corr()

corr_difference = (
    real_corr - synthetic_corr
).abs()

correlation_error = corr_difference.mean().mean()

print("\n===================================")
print("CORRELATION COMPARISON")
print("===================================")

print(
    "Average Correlation Difference:",
    round(correlation_error, 4)
)

real_corr.to_csv(
    f"{OUTPUT_DIR}/real_correlation.csv"
)

synthetic_corr.to_csv(
    f"{OUTPUT_DIR}/synthetic_correlation.csv"
)


# ==========================================
# OVERALL QUALITY SCORE
# ==========================================

mean_error = comparison[
    "Mean Difference %"
].replace(
    [np.inf, -np.inf],
    np.nan
).dropna().mean()

# Convert error into a simple similarity score.
# This is a project-level indicator, not a formal
# statistical utility/privacy metric.

distribution_score = max(
    0,
    100 - mean_error
)

correlation_score = max(
    0,
    100 - (correlation_error * 100)
)

overall_score = (
    distribution_score * 0.7
    + correlation_score * 0.3
)

print("\n===================================")
print("BANKING SYNTHETIC DATA QUALITY")
print("===================================")

print(
    "Distribution Score:",
    round(distribution_score, 2)
)

print(
    "Correlation Score:",
    round(correlation_score, 2)
)

print(
    "Overall Similarity Score:",
    round(overall_score, 2)
)


# ==========================================
# SAVE QUALITY REPORT
# ==========================================

quality_report = pd.DataFrame({
    "Metric": [
        "Real Records",
        "Synthetic Records",
        "Distribution Score",
        "Correlation Score",
        "Overall Similarity Score"
    ],
    "Value": [
        len(real),
        len(synthetic),
        round(distribution_score, 2),
        round(correlation_score, 2),
        round(overall_score, 2)
    ]
})

quality_report.to_csv(
    f"{OUTPUT_DIR}/quality_report.csv",
    index=False
)


print("\n===================================")
print("COMPARISON COMPLETE")
print("===================================")

print(
    f"Results saved in: {OUTPUT_DIR}"
)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# =========================================================
# PATHS
# =========================================================

REAL_PATH = "data/banking/banking_combined.csv"
SYNTHETIC_PATH = "data/banking/synthetic_banking.csv"

OUTPUT_DIR = "data/banking/comparison/charts"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

real = pd.read_csv(REAL_PATH)
synthetic = pd.read_csv(SYNTHETIC_PATH)

print("Real Data:", real.shape)
print("Synthetic Data:", synthetic.shape)


# =========================================================
# NUMERIC COLUMNS
# =========================================================

numeric_columns = [
    "age",
    "tenure_years",
    "avg_balance",
    "monthly_logins",
    "products_held",
    "loan_count",
    "total_loan_amount",
    "avg_loan_amount",
    "avg_income",
    "avg_term_months",
    "avg_dti_ratio",
    "avg_credit_score",
    "avg_employment_years",
    "avg_utilisation",
    "total_missed_payments",
    "default_count",
    "transaction_count",
    "total_transaction_amount",
    "avg_transaction_amount",
    "fraud_count",
    "new_device_count"
]


# =========================================================
# 1. NUMERIC MEAN COMPARISON
# =========================================================

real_means = real[numeric_columns].mean()
synthetic_means = synthetic[numeric_columns].mean()

x = np.arange(len(numeric_columns))
width = 0.38

plt.figure(figsize=(16, 7))

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

plt.title(
    "Banking: Real vs Synthetic Numeric Mean Comparison",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Numeric Features")
plt.ylabel("Mean Value")

plt.xticks(
    x,
    numeric_columns,
    rotation=75,
    ha="right"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/01_numeric_mean_comparison.png",
    dpi=300
)

plt.close()


# =========================================================
# 2. NORMALIZED NUMERIC COMPARISON
# =========================================================
# Different banking features have very different scales.
# Normalize means using the real-data mean.

comparison = pd.DataFrame({
    "Real": real_means,
    "Synthetic": synthetic_means
})

comparison["Real_Normalized"] = 1.0

comparison["Synthetic_Normalized"] = (
    comparison["Synthetic"] /
    comparison["Real"].replace(0, np.nan)
)

comparison = comparison.fillna(0)

plt.figure(figsize=(16, 7))

plt.bar(
    x - width / 2,
    comparison["Real_Normalized"],
    width,
    label="Real"
)

plt.bar(
    x + width / 2,
    comparison["Synthetic_Normalized"],
    width,
    label="Synthetic"
)

plt.axhline(
    1.0,
    linestyle="--",
    linewidth=1
)

plt.title(
    "Banking: Normalized Real vs Synthetic Feature Comparison",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Features")
plt.ylabel("Synthetic / Real Mean")

plt.xticks(
    x,
    numeric_columns,
    rotation=75,
    ha="right"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/02_normalized_comparison.png",
    dpi=300
)

plt.close()


# =========================================================
# 3. CATEGORICAL DISTRIBUTION
# =========================================================

categorical_columns = [
    "segment",
    "churn_risk",
    "primary_merchant_category",
    "primary_channel",
    "primary_country"
]


for column in categorical_columns:

    real_counts = (
        real[column]
        .value_counts(normalize=True)
        .mul(100)
    )

    synthetic_counts = (
        synthetic[column]
        .value_counts(normalize=True)
        .mul(100)
    )

    categories = sorted(
        set(real_counts.index)
        .union(synthetic_counts.index)
    )

    real_values = [
        real_counts.get(category, 0)
        for category in categories
    ]

    synthetic_values = [
        synthetic_counts.get(category, 0)
        for category in categories
    ]

    x_cat = np.arange(len(categories))

    plt.figure(figsize=(11, 6))

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

    plt.title(
        f"Banking: {column} Distribution",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel(column.replace("_", " ").title())
    plt.ylabel("Percentage (%)")

    plt.xticks(
        x_cat,
        categories,
        rotation=35,
        ha="right"
    )

    plt.legend()
    plt.tight_layout()

    filename = (
        f"{OUTPUT_DIR}/03_{column}_distribution.png"
    )

    plt.savefig(
        filename,
        dpi=300
    )

    plt.close()


# =========================================================
# 4. CORRELATION HEATMAP - REAL
# =========================================================

real_corr = real[numeric_columns].corr()

plt.figure(figsize=(14, 11))

plt.imshow(
    real_corr,
    aspect="auto"
)

plt.colorbar(label="Correlation")

plt.title(
    "Banking Real Data - Correlation Matrix",
    fontsize=15,
    fontweight="bold"
)

plt.xticks(
    range(len(numeric_columns)),
    numeric_columns,
    rotation=90
)

plt.yticks(
    range(len(numeric_columns)),
    numeric_columns
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/09_real_correlation.png",
    dpi=300
)

plt.close()


# =========================================================
# 5. CORRELATION HEATMAP - SYNTHETIC
# =========================================================

synthetic_corr = synthetic[numeric_columns].corr()

plt.figure(figsize=(14, 11))

plt.imshow(
    synthetic_corr,
    aspect="auto"
)

plt.colorbar(label="Correlation")

plt.title(
    "Banking Synthetic Data - Correlation Matrix",
    fontsize=15,
    fontweight="bold"
)

plt.xticks(
    range(len(numeric_columns)),
    numeric_columns,
    rotation=90
)

plt.yticks(
    range(len(numeric_columns)),
    numeric_columns
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/10_synthetic_correlation.png",
    dpi=300
)

plt.close()


# =========================================================
# 6. CORRELATION DIFFERENCE
# =========================================================

correlation_difference = (
    real_corr - synthetic_corr
).abs()

plt.figure(figsize=(14, 11))

plt.imshow(
    correlation_difference,
    aspect="auto"
)

plt.colorbar(
    label="Absolute Correlation Difference"
)

plt.title(
    "Banking: Real vs Synthetic Correlation Difference",
    fontsize=15,
    fontweight="bold"
)

plt.xticks(
    range(len(numeric_columns)),
    numeric_columns,
    rotation=90
)

plt.yticks(
    range(len(numeric_columns)),
    numeric_columns
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/11_correlation_difference.png",
    dpi=300
)

plt.close()


# =========================================================
# 7. QUALITY SCORE SUMMARY
# =========================================================

quality_path = (
    "data/banking/comparison/"
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

    plt.title(
        "Banking Synthetic Data Quality",
        fontsize=15,
        fontweight="bold"
    )

    plt.ylabel("Score (%)")
    plt.ylim(0, 100)

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

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/12_quality_score.png",
        dpi=300
    )

    plt.close()


# =========================================================
# COMPLETE
# =========================================================

print()
print("===================================")
print("BANKING COMPARISON CHARTS CREATED")
print("===================================")

print()
print("Charts saved to:")
print(OUTPUT_DIR)

print()
print("Generated:")
print("01_numeric_mean_comparison.png")
print("02_normalized_comparison.png")
print("03_segment_distribution.png")
print("03_churn_risk_distribution.png")
print("03_primary_merchant_category_distribution.png")
print("03_primary_channel_distribution.png")
print("03_primary_country_distribution.png")
print("09_real_correlation.png")
print("10_synthetic_correlation.png")
print("11_correlation_difference.png")
print("12_quality_score.png")

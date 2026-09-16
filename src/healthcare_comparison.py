
import pandas as pd
import numpy as np
import os

# =========================================================
# LOAD DATA
# =========================================================

REAL_PATH = "data/healthcare_dataset.csv"
SYNTHETIC_PATH = "data/synthetic_healthcare.csv"

real = pd.read_csv(REAL_PATH)
synthetic = pd.read_csv(SYNTHETIC_PATH)

print("REAL DATA:", real.shape)
print("SYNTHETIC DATA:", synthetic.shape)


# =========================================================
# ALIGN COMMON COLUMNS
# =========================================================

common_columns = [
    col for col in real.columns
    if col in synthetic.columns
]

real = real[common_columns].copy()
synthetic = synthetic[common_columns].copy()

print("\nCommon Columns:")
print(common_columns)


# =========================================================
# NUMERIC COLUMNS
# =========================================================

numeric_columns = real.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

print("\n===================================")
print("NUMERIC COMPARISON")
print("===================================")

numeric_results = []

for column in numeric_columns:

    real_mean = real[column].mean()
    synthetic_mean = synthetic[column].mean()

    real_std = real[column].std()
    synthetic_std = synthetic[column].std()

    if real_mean != 0:
        mean_difference = (
            abs(real_mean - synthetic_mean)
            / abs(real_mean)
        ) * 100
    else:
        mean_difference = 0

    numeric_results.append({
        "Column": column,
        "Real Mean": round(real_mean, 2),
        "Synthetic Mean": round(synthetic_mean, 2),
        "Real Std": round(real_std, 2),
        "Synthetic Std": round(synthetic_std, 2),
        "Mean Difference %": round(mean_difference, 2)
    })

numeric_comparison = pd.DataFrame(numeric_results)

print(numeric_comparison.to_string(index=False))


# =========================================================
# CATEGORICAL COLUMNS
# =========================================================

categorical_columns = [
    col for col in common_columns
    if col not in numeric_columns
]

print("\nCategorical Columns:")
print(categorical_columns)


# =========================================================
# DISTRIBUTION SCORE
# =========================================================

distribution_scores = []

for column in numeric_columns:

    real_mean = real[column].mean()
    synthetic_mean = synthetic[column].mean()

    real_std = real[column].std()
    synthetic_std = synthetic[column].std()

    mean_similarity = 1 - (
        abs(real_mean - synthetic_mean)
        / (abs(real_mean) + 1e-8)
    )

    std_similarity = 1 - (
        abs(real_std - synthetic_std)
        / (abs(real_std) + 1e-8)
    )

    mean_similarity = max(0, mean_similarity)
    std_similarity = max(0, std_similarity)

    score = (
        (mean_similarity * 0.5)
        + (std_similarity * 0.5)
    )

    distribution_scores.append(score)


if distribution_scores:
    distribution_score = (
        np.mean(distribution_scores) * 100
    )
else:
    distribution_score = 0


# =========================================================
# CATEGORICAL DISTRIBUTION
# =========================================================

categorical_scores = []

for column in categorical_columns:

    real_dist = (
        real[column]
        .value_counts(normalize=True)
    )

    synthetic_dist = (
        synthetic[column]
        .value_counts(normalize=True)
    )

    categories = set(real_dist.index).union(
        set(synthetic_dist.index)
    )

    difference = 0

    for category in categories:

        real_value = real_dist.get(category, 0)
        synthetic_value = synthetic_dist.get(category, 0)

        difference += abs(
            real_value - synthetic_value
        )

    categorical_similarity = max(
        0,
        1 - (difference / 2)
    )

    categorical_scores.append(
        categorical_similarity
    )


if categorical_scores:

    categorical_score = (
        np.mean(categorical_scores) * 100
    )

    distribution_score = (
        distribution_score + categorical_score
    ) / 2


# =========================================================
# CORRELATION COMPARISON
# =========================================================

print("\n===================================")
print("CORRELATION COMPARISON")
print("===================================")

if len(numeric_columns) >= 2:

    real_corr = real[numeric_columns].corr()
    synthetic_corr = synthetic[numeric_columns].corr()

    common_corr_columns = [
        col for col in numeric_columns
        if col in synthetic_corr.columns
    ]

    real_corr = real_corr.loc[
        common_corr_columns,
        common_corr_columns
    ]

    synthetic_corr = synthetic_corr.loc[
        common_corr_columns,
        common_corr_columns
    ]

    correlation_difference = (
        abs(real_corr - synthetic_corr)
    )

    upper_triangle = np.triu(
        np.ones(correlation_difference.shape),
        k=1
    ).astype(bool)

    avg_correlation_difference = (
        correlation_difference
        .where(upper_triangle)
        .stack()
        .mean()
    )

    correlation_score = (
        1 - avg_correlation_difference
    ) * 100

    correlation_score = max(
        0,
        correlation_score
    )

else:

    correlation_score = 0
    avg_correlation_difference = 1


print(
    f"Average Correlation Difference: "
    f"{avg_correlation_difference:.4f}"
)


# =========================================================
# OVERALL SIMILARITY SCORE
# =========================================================

overall_similarity_score = (
    distribution_score * 0.5
    + correlation_score * 0.5
)


# =========================================================
# DISPLAY QUALITY
# =========================================================

print("\n===================================")
print("HEALTHCARE SYNTHETIC DATA QUALITY")
print("===================================")

print(
    f"Distribution Score: "
    f"{distribution_score:.2f}"
)

print(
    f"Correlation Score: "
    f"{correlation_score:.2f}"
)

print(
    f"Overall Similarity Score: "
    f"{overall_similarity_score:.2f}"
)


# =========================================================
# SAVE RESULTS
# =========================================================

output_dir = "data/healthcare/comparison"

os.makedirs(
    output_dir,
    exist_ok=True
)

numeric_comparison.to_csv(
    f"{output_dir}/numeric_comparison.csv",
    index=False
)

summary = pd.DataFrame([{
    "Domain": "Healthcare",
    "Distribution Score": round(
        distribution_score, 2
    ),
    "Correlation Score": round(
        correlation_score, 2
    ),
    "Overall Similarity Score": round(
        overall_similarity_score, 2
    ),
    "Real Records": len(real),
    "Synthetic Records": len(synthetic)
}])

summary.to_csv(
    f"{output_dir}/synthetic_comparison.csv",
    index=False
)


# =========================================================
# COMPLETE
# =========================================================

print("\n===================================")
print("COMPARISON COMPLETE")
print("===================================")

print(
    "Results saved in:",
    output_dir
)


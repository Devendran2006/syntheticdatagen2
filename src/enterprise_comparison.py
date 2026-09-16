import pandas as pd
import numpy as np
import os

# =========================================================
# LOAD DATA
# =========================================================

real = pd.read_csv(
    "data/enterprise/enterprise_combined.csv"
)

synthetic = pd.read_csv(
    "data/enterprise/synthetic_enterprise.csv"
)

print("REAL DATA:", real.shape)
print("SYNTHETIC DATA:", synthetic.shape)


# =========================================================
# NUMERIC COLUMNS
# =========================================================

numeric_columns = real.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

# Remove ID-like columns
numeric_columns = [
    col for col in numeric_columns
    if col != "employee_id"
]


# =========================================================
# NUMERIC COMPARISON
# =========================================================

comparison = []

for col in numeric_columns:

    real_mean = real[col].mean()
    synthetic_mean = synthetic[col].mean()

    real_std = real[col].std()
    synthetic_std = synthetic[col].std()

    if real_mean != 0:
        mean_difference = (
            abs(real_mean - synthetic_mean)
            / abs(real_mean)
        ) * 100
    else:
        mean_difference = 0

    comparison.append({
        "Column": col,
        "Real Mean": round(real_mean, 2),
        "Synthetic Mean": round(synthetic_mean, 2),
        "Real Std": round(real_std, 2),
        "Synthetic Std": round(synthetic_std, 2),
        "Mean Difference %": round(mean_difference, 2)
    })


comparison_df = pd.DataFrame(comparison)


# =========================================================
# CATEGORICAL COLUMNS
# =========================================================

categorical_columns = real.select_dtypes(
    include=["object"]
).columns.tolist()

categorical_columns = [
    col for col in categorical_columns
    if col != "employee_id"
]


# =========================================================
# CORRELATION COMPARISON
# =========================================================

real_corr = real[numeric_columns].corr()
synthetic_corr = synthetic[numeric_columns].corr()

correlation_difference = (
    real_corr - synthetic_corr
).abs().mean().mean()

correlation_score = max(
    0,
    100 - (correlation_difference * 100)
)


# =========================================================
# DISTRIBUTION SCORE
# =========================================================

distribution_errors = []

for col in numeric_columns:

    real_mean = real[col].mean()
    synthetic_mean = synthetic[col].mean()

    real_std = real[col].std()
    synthetic_std = synthetic[col].std()

    mean_error = 0

    if real_mean != 0:
        mean_error = abs(
            real_mean - synthetic_mean
        ) / abs(real_mean)

    std_error = 0

    if real_std != 0:
        std_error = abs(
            real_std - synthetic_std
        ) / abs(real_std)

    error = (mean_error + std_error) / 2

    distribution_errors.append(error)


average_distribution_error = np.mean(
    distribution_errors
)

distribution_score = max(
    0,
    100 - (average_distribution_error * 100)
)


# =========================================================
# OVERALL SCORE
# =========================================================

overall_score = (
    distribution_score * 0.5
    + correlation_score * 0.5
)


# =========================================================
# DISPLAY RESULTS
# =========================================================

print()
print("===================================")
print("NUMERIC COMPARISON")
print("===================================")

print(
    comparison_df.to_string(index=False)
)


print()
print("Categorical Columns:")
print(categorical_columns)


print()
print("===================================")
print("CORRELATION COMPARISON")
print("===================================")

print(
    "Average Correlation Difference:",
    round(correlation_difference, 4)
)


print()
print("===================================")
print("ENTERPRISE SYNTHETIC DATA QUALITY")
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


# =========================================================
# SAVE RESULTS
# =========================================================

output_dir = "data/enterprise/comparison"

os.makedirs(
    output_dir,
    exist_ok=True
)

comparison_df.to_csv(
    f"{output_dir}/numeric_comparison.csv",
    index=False
)

summary = pd.DataFrame({
    "Metric": [
        "Distribution Score",
        "Correlation Score",
        "Overall Similarity Score"
    ],
    "Score": [
        round(distribution_score, 2),
        round(correlation_score, 2),
        round(overall_score, 2)
    ]
})

summary.to_csv(
    f"{output_dir}/quality_summary.csv",
    index=False
)


print()
print("===================================")
print("COMPARISON COMPLETE")
print("===================================")

print(
    "Results saved in:",
    output_dir
)
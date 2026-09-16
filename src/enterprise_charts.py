
import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================================================
# PATHS
# =========================================================

REAL_PATH = "data/enterprise/enterprise_combined.csv"
SYNTHETIC_PATH = "data/enterprise/synthetic_enterprise.csv"

OUTPUT_DIR = "data/enterprise/comparison/charts"

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
    "salary",
    "overtime_pct",
    "last_promotion_years",
    "engagement_score",
    "project_count",
    "avg_planned_days",
    "avg_actual_days",
    "total_budget",
    "total_spend",
    "avg_team_size",
    "avg_scope_changes",
    "status_on_track",
    "status_overrun",
    "total_assets",
    "avg_asset_age_years",
    "avg_runtime_hours",
    "avg_temperature",
    "avg_vibration",
    "avg_tickets_90d",
    "avg_last_service_days",
    "asset_risk_low_count",
    "asset_risk_medium_count",
    "asset_risk_high_count"
]

# =========================================================
# 1. EMPLOYEE METRICS
# =========================================================

employee_metrics = [
    "age",
    "tenure_years",
    "salary",
    "overtime_pct",
    "last_promotion_years",
    "engagement_score"
]

real_means = real[employee_metrics].mean()
synthetic_means = synthetic[employee_metrics].mean()

comparison = pd.DataFrame({
    "Real": real_means,
    "Synthetic": synthetic_means
})

ax = comparison.plot(
    kind="bar",
    figsize=(12, 6)
)

ax.set_title(
    "Enterprise Employee Metrics - Real vs Synthetic",
    fontsize=14,
    fontweight="bold"
)

ax.set_ylabel("Average Value")
ax.set_xlabel("Employee Metrics")

plt.xticks(rotation=45, ha="right")
plt.legend(title="Dataset")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/employee_metrics_real_vs_synthetic.png",
    dpi=300
)

plt.close()

# =========================================================
# 2. PROJECT METRICS
# =========================================================

project_metrics = [
    "project_count",
    "avg_planned_days",
    "avg_actual_days",
    "total_budget",
    "total_spend",
    "avg_team_size",
    "avg_scope_changes",
    "status_on_track",
    "status_overrun"
]

real_means = real[project_metrics].mean()
synthetic_means = synthetic[project_metrics].mean()

comparison = pd.DataFrame({
    "Real": real_means,
    "Synthetic": synthetic_means
})

ax = comparison.plot(
    kind="bar",
    figsize=(13, 6)
)

ax.set_title(
    "Enterprise Project Metrics - Real vs Synthetic",
    fontsize=14,
    fontweight="bold"
)

ax.set_ylabel("Average / Total Value")
ax.set_xlabel("Project Metrics")

plt.xticks(rotation=45, ha="right")
plt.legend(title="Dataset")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/project_metrics_real_vs_synthetic.png",
    dpi=300
)

plt.close()

# =========================================================
# 3. ASSET METRICS
# =========================================================

asset_metrics = [
    "total_assets",
    "avg_asset_age_years",
    "avg_runtime_hours",
    "avg_temperature",
    "avg_vibration",
    "avg_tickets_90d",
    "avg_last_service_days",
    "asset_risk_low_count",
    "asset_risk_medium_count",
    "asset_risk_high_count"
]

real_means = real[asset_metrics].mean()
synthetic_means = synthetic[asset_metrics].mean()

comparison = pd.DataFrame({
    "Real": real_means,
    "Synthetic": synthetic_means
})

ax = comparison.plot(
    kind="bar",
    figsize=(13, 6)
)

ax.set_title(
    "Enterprise Asset Metrics - Real vs Synthetic",
    fontsize=14,
    fontweight="bold"
)

ax.set_ylabel("Average / Count")
ax.set_xlabel("Asset Metrics")

plt.xticks(rotation=45, ha="right")
plt.legend(title="Dataset")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/asset_metrics_real_vs_synthetic.png",
    dpi=300
)

plt.close()

# =========================================================
# 4. DEPARTMENT DISTRIBUTION
# =========================================================

real_department = real["department"].value_counts().sort_index()
synthetic_department = synthetic["department"].value_counts().reindex(
    real_department.index,
    fill_value=0
)

comparison = pd.DataFrame({
    "Real": real_department,
    "Synthetic": synthetic_department
})

ax = comparison.plot(
    kind="bar",
    figsize=(10, 6)
)

ax.set_title(
    "Department Distribution - Real vs Synthetic",
    fontsize=14,
    fontweight="bold"
)

ax.set_ylabel("Employee Count")
ax.set_xlabel("Department")

plt.xticks(rotation=0)
plt.legend(title="Dataset")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/department_distribution_real_vs_synthetic.png",
    dpi=300
)

plt.close()

# =========================================================
# 5. CORRELATION HEATMAP - REAL
# =========================================================

plt.figure(figsize=(14, 10))

corr = real[numeric_columns].corr()

plt.imshow(corr, aspect="auto")

plt.colorbar()

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=90,
    fontsize=7
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns,
    fontsize=7
)

plt.title(
    "Enterprise Real Data Correlation Matrix",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/real_correlation_heatmap.png",
    dpi=300
)

plt.close()

# =========================================================
# 6. CORRELATION HEATMAP - SYNTHETIC
# =========================================================

plt.figure(figsize=(14, 10))

corr = synthetic[numeric_columns].corr()

plt.imshow(corr, aspect="auto")

plt.colorbar()

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=90,
    fontsize=7
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns,
    fontsize=7
)

plt.title(
    "Enterprise Synthetic Data Correlation Matrix",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/synthetic_correlation_heatmap.png",
    dpi=300
)

plt.close()

# =========================================================
# COMPLETE
# =========================================================

print()
print("===================================")
print("ENTERPRISE CHARTS CREATED")
print("===================================")

print("Charts saved to:")
print(OUTPUT_DIR)

print()
print("Created Charts:")

for file in os.listdir(OUTPUT_DIR):
    print("-", file)


import pandas as pd
import os

# =========================================================
# LOAD DATA
# =========================================================

employees = pd.read_csv("data/enterprise/Employees.csv")
projects = pd.read_csv("data/enterprise/Projects.csv")
assets = pd.read_csv("data/enterprise/Assets_Maintenance.csv")

print("Employees:", employees.shape)
print("Projects:", projects.shape)
print("Assets:", assets.shape)


# =========================================================
# EMPLOYEE DATA
# Keep employee-level records
# =========================================================

combined = employees.copy()


# =========================================================
# PROJECT AGGREGATION BY DEPARTMENT
# =========================================================

project_summary = projects.groupby("department").agg(
    project_count=("project_id", "count"),
    avg_planned_days=("planned_days", "mean"),
    avg_actual_days=("actual_days", "mean"),
    total_budget=("budget", "sum"),
    total_spend=("spend", "sum"),
    avg_team_size=("team_size", "mean"),
    avg_scope_changes=("scope_changes", "mean")
).reset_index()


# =========================================================
# PROJECT STATUS BY DEPARTMENT
# =========================================================

status_table = projects.pivot_table(
    index="department",
    columns="status",
    values="project_id",
    aggfunc="count",
    fill_value=0
).reset_index()

status_table.columns.name = None

if "On Track" not in status_table.columns:
    status_table["On Track"] = 0

if "Overrun" not in status_table.columns:
    status_table["Overrun"] = 0

status_table = status_table.rename(
    columns={
        "On Track": "status_on_track",
        "Overrun": "status_overrun"
    }
)

status_table = status_table[
    [
        "department",
        "status_on_track",
        "status_overrun"
    ]
]


# =========================================================
# MERGE PROJECT INFORMATION
# INTO EACH EMPLOYEE
# =========================================================

combined = combined.merge(
    project_summary,
    on="department",
    how="left"
)

combined = combined.merge(
    status_table,
    on="department",
    how="left"
)


# =========================================================
# ASSET SUMMARY
# =========================================================

asset_summary = {
    "total_assets": len(assets),

    "avg_asset_age_years":
        assets["age_years"].mean(),

    "avg_runtime_hours":
        assets["runtime_hours"].mean(),

    "avg_temperature":
        assets["avg_temp_c"].mean(),

    "avg_vibration":
        assets["vibration_mm_s"].mean(),

    "avg_tickets_90d":
        assets["tickets_90d"].mean(),

    "avg_last_service_days":
        assets["last_service_days"].mean(),

    "asset_risk_low_count":
        (assets["failure_risk"] == "Low").sum(),

    "asset_risk_medium_count":
        (assets["failure_risk"] == "Medium").sum(),

    "asset_risk_high_count":
        (assets["failure_risk"] == "High").sum()
}


# =========================================================
# ADD ASSET INFORMATION
# TO EACH EMPLOYEE RECORD
# =========================================================

for column, value in asset_summary.items():
    combined[column] = value


# =========================================================
# ROUND NUMERIC VALUES
# =========================================================

numeric_columns = combined.select_dtypes(
    include=["float64", "float32"]
).columns

combined[numeric_columns] = combined[
    numeric_columns
].round(2)


# =========================================================
# HANDLE MISSING VALUES
# =========================================================

numeric_columns = combined.select_dtypes(
    include=["number"]
).columns

combined[numeric_columns] = combined[
    numeric_columns
].fillna(0)


# =========================================================
# SAVE DATASET
# =========================================================

output_path = "data/enterprise/enterprise_combined.csv"

combined.to_csv(
    output_path,
    index=False
)


# =========================================================
# DISPLAY RESULT
# =========================================================

print()
print("===================================")
print("ENTERPRISE DATASET CREATED")
print("===================================")

print("Shape:")
print(combined.shape)

print()
print("Columns:")
print(combined.columns.tolist())

print()
print("First 5 records:")
print(
    combined.head().to_string(index=False)
)

print()
print("Saved to:")
print(output_path)
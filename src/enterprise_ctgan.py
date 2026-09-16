import os
import joblib
import pandas as pd

from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_PATH = (
    "data/enterprise/enterprise_combined.csv"
)

SYNTHETIC_PATH = (
    "data/enterprise/"
    "synthetic_enterprise_large.csv"
)

EXPANDED_PATH = (
    "data/enterprise/"
    "enterprise_expanded.csv"
)

MODEL_PATH = (
    "models/"
    "enterprise_ctgan_large.pkl"
)


# Number of synthetic records
#
# Example:
# Real = 1000
# Multiplier = 2
# Synthetic = 2000
# Expanded = 3000

SYNTHETIC_MULTIPLIER = 2


# CTGAN training epochs

EPOCHS = 100


# =========================================================
# LOAD ENTERPRISE DATA
# =========================================================

print("=" * 70)
print("ENTERPRISE SYNTHETIC DATA EXPANSION")
print("=" * 70)

print("\nLoading original enterprise dataset...")


df = pd.read_csv(
    INPUT_PATH
)


print(
    f"Original Dataset Shape: "
    f"{df.shape}"
)

print(
    f"Original Records: "
    f"{len(df):,}"
)

print(
    f"Original Columns: "
    f"{len(df.columns)}"
)


# =========================================================
# CALCULATE SYNTHETIC ROW COUNT
# =========================================================

SYNTHETIC_ROWS = (
    len(df) * SYNTHETIC_MULTIPLIER
)

EXPANDED_ROWS = (
    len(df) + SYNTHETIC_ROWS
)


print("\n" + "=" * 70)
print("DATASET EXPANSION CONFIGURATION")
print("=" * 70)


print(
    f"Original Records    : "
    f"{len(df):,}"
)

print(
    f"Synthetic Multiplier: "
    f"{SYNTHETIC_MULTIPLIER}x"
)

print(
    f"Synthetic Records   : "
    f"{SYNTHETIC_ROWS:,}"
)

print(
    f"Expanded Records    : "
    f"{EXPANDED_ROWS:,}"
)


# =========================================================
# REMOVE EMPLOYEE ID
# =========================================================

print("\n" + "=" * 70)
print("PREPARING CTGAN TRAINING DATA")
print("=" * 70)


if "employee_id" not in df.columns:

    raise ValueError(
        "employee_id column not found "
        "in enterprise dataset."
    )


training_df = df.drop(
    columns=["employee_id"]
).copy()


print(
    f"\nTraining Shape: "
    f"{training_df.shape}"
)


# =========================================================
# CHECK EMPTY COLUMNS
# =========================================================

empty_columns = [
    column
    for column in training_df.columns
    if training_df[column].isna().all()
]


if empty_columns:

    print(
        "\nRemoving completely empty columns:"
    )

    for column in empty_columns:

        print(
            f" - {column}"
        )

    training_df = training_df.drop(
        columns=empty_columns
    )


# =========================================================
# HANDLE MISSING VALUES
# =========================================================

print("\nHandling missing values...")


for column in training_df.columns:

    if pd.api.types.is_numeric_dtype(
        training_df[column]
    ):

        median_value = (
            training_df[column].median()
        )

        training_df[column] = (
            training_df[column]
            .fillna(median_value)
        )

    else:

        training_df[column] = (
            training_df[column]
            .fillna("Unknown")
        )


# =========================================================
# CREATE METADATA
# =========================================================

print("\n" + "=" * 70)
print("CREATING SDV METADATA")
print("=" * 70)


metadata = SingleTableMetadata()


metadata.detect_from_dataframe(
    data=training_df
)


print(
    "Metadata detected successfully."
)


# =========================================================
# CREATE CTGAN
# =========================================================

print("\n" + "=" * 70)
print("CREATING ENTERPRISE CTGAN")
print("=" * 70)


print(
    f"Epochs: "
    f"{EPOCHS}"
)

print(
    f"Target Synthetic Records: "
    f"{SYNTHETIC_ROWS:,}"
)


synthesizer = CTGANSynthesizer(
    metadata,
    epochs=EPOCHS,
    verbose=True
)


# =========================================================
# TRAIN CTGAN
# =========================================================

print("\n" + "=" * 70)
print("TRAINING ENTERPRISE CTGAN")
print("=" * 70)


print(
    "Training started..."
)


synthesizer.fit(
    training_df
)


print(
    "\nCTGAN training completed successfully."
)


# =========================================================
# GENERATE SYNTHETIC DATA
# =========================================================

print("\n" + "=" * 70)
print("GENERATING SYNTHETIC ENTERPRISE DATA")
print("=" * 70)


print(
    f"Generating "
    f"{SYNTHETIC_ROWS:,} synthetic records..."
)


synthetic_df = synthesizer.sample(
    num_rows=SYNTHETIC_ROWS
)


print(
    "\nSynthetic Shape:"
)

print(
    synthetic_df.shape
)


# =========================================================
# ADD SYNTHETIC EMPLOYEE IDs
# =========================================================

print("\n" + "=" * 70)
print("CREATING SYNTHETIC EMPLOYEE IDs")
print("=" * 70)


synthetic_df.insert(
    0,
    "employee_id",
    [
        f"SYN-EMP-{i + 1}"
        for i in range(SYNTHETIC_ROWS)
    ]
)


# =========================================================
# POST-GENERATION VALIDATION
# =========================================================

print("\n" + "=" * 70)
print("POST-GENERATION VALIDATION")
print("=" * 70)


missing_after_generation = (
    synthetic_df.isnull()
    .sum()
)


missing_after_generation = (
    missing_after_generation[
        missing_after_generation > 0
    ]
)


print("\nMissing Values:")


if len(missing_after_generation) == 0:

    print(
        "No missing values found."
    )

else:

    print(
        missing_after_generation
    )


# =========================================================
# NUMERIC RANGE CHECK
# =========================================================

print("\n" + "=" * 70)
print("NUMERIC RANGE COMPARISON")
print("=" * 70)


numeric_columns = (
    training_df
    .select_dtypes(
        include="number"
    )
    .columns
)


for column in numeric_columns:

    real_min = (
        training_df[column].min()
    )

    real_max = (
        training_df[column].max()
    )

    synthetic_min = (
        synthetic_df[column].min()
    )

    synthetic_max = (
        synthetic_df[column].max()
    )

    print(
        f"{column:30} "
        f"Real [{real_min:.2f}, "
        f"{real_max:.2f}] "
        f"Synthetic [{synthetic_min:.2f}, "
        f"{synthetic_max:.2f}]"
    )


# =========================================================
# SAVE SYNTHETIC DATA
# =========================================================

print("\n" + "=" * 70)
print("SAVING SYNTHETIC ENTERPRISE DATA")
print("=" * 70)


os.makedirs(
    os.path.dirname(SYNTHETIC_PATH),
    exist_ok=True
)


synthetic_df.to_csv(
    SYNTHETIC_PATH,
    index=False
)


print(
    "\nSynthetic dataset saved:"
)

print(
    SYNTHETIC_PATH
)


# =========================================================
# CREATE EXPANDED DATASET
# =========================================================

print("\n" + "=" * 70)
print("CREATING EXPANDED ENTERPRISE DATASET")
print("=" * 70)


expanded_df = pd.concat(
    [
        df,
        synthetic_df
    ],
    ignore_index=True
)


print(
    f"\nExpanded Dataset Shape: "
    f"{expanded_df.shape}"
)


# =========================================================
# SAVE EXPANDED DATASET
# =========================================================

expanded_df.to_csv(
    EXPANDED_PATH,
    index=False
)


print(
    "\nExpanded dataset saved:"
)

print(
    EXPANDED_PATH
)


# =========================================================
# SAVE CTGAN MODEL
# =========================================================

print("\n" + "=" * 70)
print("SAVING CTGAN MODEL")
print("=" * 70)


os.makedirs(
    "models",
    exist_ok=True
)


joblib.dump(
    synthesizer,
    MODEL_PATH
)


print(
    "\nCTGAN model saved:"
)

print(
    MODEL_PATH
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("ENTERPRISE DATA EXPANSION COMPLETE")
print("=" * 70)


print(
    f"\nOriginal Records   : "
    f"{len(df):,}"
)

print(
    f"Synthetic Records  : "
    f"{len(synthetic_df):,}"
)

print(
    f"Expanded Records   : "
    f"{len(expanded_df):,}"
)

print(
    f"Original Columns   : "
    f"{len(df.columns)}"
)

print(
    f"Synthetic Columns  : "
    f"{len(synthetic_df.columns)}"
)

print(
    f"Expanded Columns   : "
    f"{len(expanded_df.columns)}"
)


print("\nFiles created:")

print(
    f"1. {SYNTHETIC_PATH}"
)

print(
    f"2. {EXPANDED_PATH}"
)

print(
    f"3. {MODEL_PATH}"
)


print(
    "\nEnterprise synthetic data expansion "
    "completed successfully."
)
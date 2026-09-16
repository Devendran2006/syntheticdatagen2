import os
import joblib
import pandas as pd

from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer


# =========================================================
# CONFIGURATION
# =========================================================

REAL_PATH = "data/healthcare_dataset.csv"

SYNTHETIC_PATH = "data/synthetic_healthcare_large.csv"

EXPANDED_PATH = "data/healthcare_expanded.csv"

MODEL_PATH = "models/healthcare_ctgan_large.pkl"

# Number of synthetic records to generate
SYNTHETIC_ROWS = 111000

# CTGAN training epochs
EPOCHS = 150


# =========================================================
# HIGH-CARDINALITY COLUMNS
# =========================================================
#
# These columns contain thousands of unique values.
# Training CTGAN directly on them creates huge one-hot
# matrices and causes memory errors.
#
# They will NOT be given to CTGAN.
# They will be recreated separately in the final dataset.
# =========================================================

HIGH_CARDINALITY_COLUMNS = [
    "Name",
    "Doctor",
    "Hospital"
]


# =========================================================
# LOAD ORIGINAL DATA
# =========================================================

print("=" * 70)
print("HEALTHCARE SYNTHETIC DATA EXPANSION")
print("=" * 70)

print("\nLoading original healthcare dataset...")

real_df = pd.read_csv(
    REAL_PATH
)

print(
    f"Original Dataset Shape: {real_df.shape}"
)

print(
    f"Original Records: {len(real_df):,}"
)

print(
    f"Original Columns: {len(real_df.columns)}"
)


# =========================================================
# CHECK HIGH-CARDINALITY COLUMNS
# =========================================================

print("\n" + "=" * 70)
print("HIGH-CARDINALITY COLUMN CHECK")
print("=" * 70)

for column in HIGH_CARDINALITY_COLUMNS:

    if column in real_df.columns:

        unique_count = real_df[column].nunique(
            dropna=False
        )

        print(
            f"{column}: {unique_count:,} unique values"
        )


# =========================================================
# CREATE CTGAN TRAINING DATA
# =========================================================

print("\n" + "=" * 70)
print("PREPARING CTGAN TRAINING DATA")
print("=" * 70)

ctgan_df = real_df.copy()


# Remove high-cardinality columns only
# from the CTGAN training dataset.

columns_to_remove = [
    column
    for column in HIGH_CARDINALITY_COLUMNS
    if column in ctgan_df.columns
]

ctgan_df = ctgan_df.drop(
    columns=columns_to_remove
)


print(
    "\nColumns excluded from CTGAN:"
)

for column in columns_to_remove:

    print(
        f" - {column}"
    )


print(
    f"\nCTGAN Training Shape: {ctgan_df.shape}"
)

print(
    f"CTGAN Training Columns: {len(ctgan_df.columns)}"
)


# =========================================================
# REMOVE EMPTY COLUMNS IF ANY
# =========================================================

empty_columns = [
    column
    for column in ctgan_df.columns
    if ctgan_df[column].isna().all()
]

if empty_columns:

    print(
        "\nRemoving completely empty columns:"
    )

    for column in empty_columns:

        print(
            f" - {column}"
        )

    ctgan_df = ctgan_df.drop(
        columns=empty_columns
    )


# =========================================================
# METADATA
# =========================================================

print("\n" + "=" * 70)
print("CREATING SDV METADATA")
print("=" * 70)

metadata = SingleTableMetadata()

metadata.detect_from_dataframe(
    ctgan_df
)

print(
    "Metadata detected successfully."
)


# =========================================================
# CREATE CTGAN
# =========================================================

print("\n" + "=" * 70)
print("CREATING CTGAN MODEL")
print("=" * 70)

print(
    f"Epochs: {EPOCHS}"
)

print(
    f"Target Synthetic Records: {SYNTHETIC_ROWS:,}"
)


synthesizer = CTGANSynthesizer(
    metadata=metadata,
    epochs=EPOCHS,
    verbose=True
)


# =========================================================
# TRAIN CTGAN
# =========================================================

print("\n" + "=" * 70)
print("TRAINING CTGAN")
print("=" * 70)

print(
    "Training started..."
)

synthesizer.fit(
    ctgan_df
)

print(
    "\nCTGAN training completed successfully."
)


# =========================================================
# GENERATE SYNTHETIC DATA
# =========================================================

print("\n" + "=" * 70)
print("GENERATING SYNTHETIC DATA")
print("=" * 70)

print(
    f"Generating {SYNTHETIC_ROWS:,} synthetic records..."
)

synthetic_core = synthesizer.sample(
    num_rows=SYNTHETIC_ROWS
)

print(
    "\nSynthetic Core Shape:"
)

print(
    synthetic_core.shape
)


# =========================================================
# RECREATE HIGH-CARDINALITY COLUMNS
# =========================================================
#
# The original Name / Doctor / Hospital columns are sampled
# from the original value distributions.
#
# This prevents CTGAN from creating enormous one-hot matrices.
# =========================================================

print("\n" + "=" * 70)
print("RECREATING HIGH-CARDINALITY COLUMNS")
print("=" * 70)

for column in HIGH_CARDINALITY_COLUMNS:

    if column not in real_df.columns:
        continue

    print(
        f"Generating synthetic values for: {column}"
    )

    # Preserve original frequency distribution
    value_distribution = (
        real_df[column]
        .value_counts(
            normalize=True,
            dropna=False
        )
    )

    values = value_distribution.index.tolist()

    probabilities = value_distribution.values

    synthetic_values = pd.Series(
        pd.Series(values)
        .sample(
            n=SYNTHETIC_ROWS,
            replace=True,
            weights=probabilities,
            random_state=42
        )
        .values
    )

    synthetic_core[column] = synthetic_values.values


# =========================================================
# RESTORE ORIGINAL COLUMN ORDER
# =========================================================

print("\n" + "=" * 70)
print("RESTORING ORIGINAL COLUMN STRUCTURE")
print("=" * 70)

final_columns = [
    column
    for column in real_df.columns
    if column in synthetic_core.columns
]

synthetic_df = synthetic_core[
    final_columns
].copy()


print(
    f"Final Synthetic Shape: {synthetic_df.shape}"
)


# =========================================================
# VERIFY COLUMN STRUCTURE
# =========================================================

print("\nOriginal Columns:")
print(
    list(real_df.columns)
)

print("\nSynthetic Columns:")
print(
    list(synthetic_df.columns)
)


if list(real_df.columns) == list(synthetic_df.columns):

    print(
        "\nColumn structure verification: PASSED"
    )

else:

    print(
        "\nWARNING: Column structure is different."
    )


# =========================================================
# SAVE SYNTHETIC DATA
# =========================================================

os.makedirs(
    "data",
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
print("CREATING EXPANDED HEALTHCARE DATASET")
print("=" * 70)

expanded_df = pd.concat(
    [
        real_df,
        synthetic_df
    ],
    ignore_index=True
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
print("HEALTHCARE DATA EXPANSION COMPLETE")
print("=" * 70)

print(
    f"\nOriginal Records   : {len(real_df):,}"
)

print(
    f"Synthetic Records  : {len(synthetic_df):,}"
)

print(
    f"Expanded Records   : {len(expanded_df):,}"
)

print(
    f"Original Columns   : {len(real_df.columns)}"
)

print(
    f"Synthetic Columns  : {len(synthetic_df.columns)}"
)

print(
    f"Expanded Columns   : {len(expanded_df.columns)}"
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

print("\nProcess completed successfully.")
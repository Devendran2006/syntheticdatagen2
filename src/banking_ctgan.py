import os
import joblib
import pandas as pd

from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_PATH = "data/banking/banking_combined.csv"

SYNTHETIC_PATH = "data/banking/synthetic_banking_large.csv"

EXPANDED_PATH = "data/banking/banking_expanded.csv"

METADATA_PATH = "data/banking/banking_metadata.json"

MODEL_PATH = "models/banking_ctgan_large.pkl"


# Number of synthetic records
# Example:
# Real = 1000
# Multiplier = 2
# Synthetic = 2000
# Expanded = 3000

SYNTHETIC_MULTIPLIER = 2


# CTGAN training epochs

EPOCHS = 400


# =========================================================
# LOAD BANKING DATA
# =========================================================

print("=" * 70)
print("BANKING SYNTHETIC DATA EXPANSION")
print("=" * 70)

print("\nLoading original banking dataset...")

df = pd.read_csv(
    INPUT_PATH
)

print(
    f"Original Dataset Shape: {df.shape}"
)

print(
    f"Original Records: {len(df):,}"
)

print(
    f"Original Columns: {len(df.columns)}"
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
    f"Original Records   : {len(df):,}"
)

print(
    f"Synthetic Multiplier: {SYNTHETIC_MULTIPLIER}x"
)

print(
    f"Synthetic Records  : {SYNTHETIC_ROWS:,}"
)

print(
    f"Expanded Records   : {EXPANDED_ROWS:,}"
)


# =========================================================
# REMOVE CUSTOMER ID
# =========================================================

print("\n" + "=" * 70)
print("PREPARING CTGAN TRAINING DATA")
print("=" * 70)

if "customer_id" not in df.columns:

    raise ValueError(
        "customer_id column not found in banking dataset."
    )

train_df = df.drop(
    columns=["customer_id"]
).copy()


print(
    f"\nTraining Shape: {train_df.shape}"
)


# =========================================================
# DEFINE DATA TYPES
# =========================================================

categorical_columns = [
    "segment",
    "churn_risk",
    "primary_merchant_category",
    "primary_channel",
    "primary_country"
]


numerical_columns = [
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
# CHECK REQUIRED COLUMNS
# =========================================================

required_columns = (
    categorical_columns +
    numerical_columns
)

missing_columns = [
    column
    for column in required_columns
    if column not in train_df.columns
]


if missing_columns:

    raise ValueError(
        "Missing columns in banking_combined.csv: "
        f"{missing_columns}"
    )


# =========================================================
# FIX NUMERICAL DATA TYPES
# =========================================================

print("\nConverting numerical columns...")

for column in numerical_columns:

    train_df[column] = pd.to_numeric(
        train_df[column],
        errors="coerce"
    )


# =========================================================
# FIX CATEGORICAL DATA TYPES
# =========================================================

print("Converting categorical columns...")

for column in categorical_columns:

    train_df[column] = (
        train_df[column]
        .astype(str)
    )


# =========================================================
# HANDLE MISSING VALUES
# =========================================================

print("\nHandling missing values...")

for column in numerical_columns:

    median_value = train_df[column].median()

    train_df[column] = (
        train_df[column]
        .fillna(median_value)
    )


for column in categorical_columns:

    train_df[column] = (
        train_df[column]
        .fillna("Unknown")
    )


# =========================================================
# VERIFY MISSING VALUES
# =========================================================

remaining_missing = (
    train_df.isnull()
    .sum()
    .sum()
)


print(
    f"Remaining Missing Values: "
    f"{remaining_missing}"
)


if remaining_missing > 0:

    raise ValueError(
        "Missing values still exist in training data."
    )


# =========================================================
# CREATE METADATA
# =========================================================

print("\n" + "=" * 70)
print("CREATING SDV METADATA")
print("=" * 70)

metadata = Metadata.detect_from_dataframe(
    data=train_df
)

print(
    "Metadata detected successfully."
)


# =========================================================
# SAVE METADATA
# =========================================================

os.makedirs(
    os.path.dirname(METADATA_PATH),
    exist_ok=True
)

if os.path.exists(METADATA_PATH):

    os.remove(
        METADATA_PATH
    )


metadata.save_to_json(
    METADATA_PATH
)

print(
    "\nMetadata saved:"
)

print(
    METADATA_PATH
)


# =========================================================
# CREATE CTGAN
# =========================================================

print("\n" + "=" * 70)
print("CREATING BANKING CTGAN")
print("=" * 70)

print(
    f"Epochs: {EPOCHS}"
)

print(
    f"Target Synthetic Records: "
    f"{SYNTHETIC_ROWS:,}"
)

print(
    f"Categorical Columns: "
    f"{len(categorical_columns)}"
)

print(
    f"Numerical Columns: "
    f"{len(numerical_columns)}"
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
print("TRAINING BANKING CTGAN")
print("=" * 70)

print(
    "Training started..."
)

synthesizer.fit(
    train_df
)

print(
    "\nCTGAN training completed successfully."
)


# =========================================================
# GENERATE SYNTHETIC DATA
# =========================================================

print("\n" + "=" * 70)
print("GENERATING SYNTHETIC BANKING DATA")
print("=" * 70)

print(
    f"Generating {SYNTHETIC_ROWS:,} synthetic records..."
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
# CREATE SYNTHETIC CUSTOMER IDS
# =========================================================

print("\n" + "=" * 70)
print("CREATING SYNTHETIC CUSTOMER IDs")
print("=" * 70)


synthetic_df.insert(
    0,
    "customer_id",
    [
        f"SYN-CU-{i + 1}"
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


for column in numerical_columns:

    real_min = train_df[column].min()
    real_max = train_df[column].max()

    synthetic_min = (
        synthetic_df[column].min()
    )

    synthetic_max = (
        synthetic_df[column].max()
    )

    print(
        f"{column:30} "
        f"Real [{real_min:.2f}, {real_max:.2f}] "
        f"Synthetic [{synthetic_min:.2f}, "
        f"{synthetic_max:.2f}]"
    )


# =========================================================
# SAVE SYNTHETIC DATA
# =========================================================

print("\n" + "=" * 70)
print("SAVING SYNTHETIC BANKING DATA")
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
print("CREATING EXPANDED BANKING DATASET")
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
print("BANKING DATA EXPANSION COMPLETE")
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
    f"4. {METADATA_PATH}"
)


print(
    "\nBanking synthetic data expansion "
    "completed successfully."
)
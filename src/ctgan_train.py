import os
import pandas as pd
import joblib

from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer


# =========================================================
# CONFIGURATION
# =========================================================

REAL_DATA_PATH = "data/ctgan_ready.csv"

SYNTHETIC_OUTPUT_PATH = "outputs/synthetic_healthcare.csv"

MODEL_OUTPUT_PATH = "models/ctgan_model.pkl"

# Number of synthetic records to generate
SYNTHETIC_ROWS = 100000

# CTGAN training epochs
EPOCHS = 150


# =========================================================
# LOAD REAL DATA
# =========================================================

print("=" * 60)
print("HEALTHCARE CTGAN SYNTHETIC DATA GENERATION")
print("=" * 60)

df = pd.read_csv(REAL_DATA_PATH)

print()
print("Real Dataset Shape:", df.shape)
print("Real Records:", len(df))
print("Columns:", len(df.columns))


# =========================================================
# METADATA
# =========================================================

print()
print("Detecting Metadata...")

metadata = SingleTableMetadata()

metadata.detect_from_dataframe(
    df
)

print("Metadata Detection Completed")


# =========================================================
# CREATE CTGAN
# =========================================================

print()
print("Creating CTGAN Model...")

synthesizer = CTGANSynthesizer(
    metadata=metadata,
    epochs=EPOCHS,
    verbose=True
)


# =========================================================
# TRAIN CTGAN
# =========================================================

print()
print("=" * 60)
print("TRAINING CTGAN")
print("=" * 60)

print("Epochs:", EPOCHS)

synthesizer.fit(df)

print()
print("CTGAN Training Completed")


# =========================================================
# GENERATE SYNTHETIC DATA
# =========================================================

print()
print("=" * 60)
print("GENERATING SYNTHETIC DATA")
print("=" * 60)

print("Real Records:", len(df))
print("Synthetic Records Requested:", SYNTHETIC_ROWS)

synthetic_data = synthesizer.sample(
    num_rows=SYNTHETIC_ROWS
)


# =========================================================
# CREATE OUTPUT DIRECTORIES
# =========================================================

os.makedirs(
    "outputs",
    exist_ok=True
)

os.makedirs(
    "models",
    exist_ok=True
)


# =========================================================
# SAVE SYNTHETIC DATA
# =========================================================

synthetic_data.to_csv(
    SYNTHETIC_OUTPUT_PATH,
    index=False
)


# =========================================================
# SAVE TRAINED MODEL
# =========================================================

joblib.dump(
    synthesizer,
    MODEL_OUTPUT_PATH
)


# =========================================================
# FINAL VALIDATION
# =========================================================

print()
print("=" * 60)
print("GENERATION COMPLETE")
print("=" * 60)

print()
print("Real Dataset:")
print("Records :", len(df))
print("Columns :", len(df.columns))

print()
print("Synthetic Dataset:")
print("Records :", len(synthetic_data))
print("Columns :", len(synthetic_data.columns))

print()
print("Saved Synthetic Data:")
print(SYNTHETIC_OUTPUT_PATH)

print()
print("Saved CTGAN Model:")
print(MODEL_OUTPUT_PATH)

print()
print("First 5 Synthetic Records:")
print(
    synthetic_data.head()
)

print()
print("Synthetic Dataset Shape:")
print(
    synthetic_data.shape
)

print()
print("=" * 60)
print("HEALTHCARE SYNTHETIC DATA GENERATION SUCCESSFUL")
print("=" * 60)
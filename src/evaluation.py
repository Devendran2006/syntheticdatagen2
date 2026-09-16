import pandas as pd

real_df = pd.read_csv("data/ctgan_ready.csv")
synthetic_df = pd.read_csv("data/synthetic_healthcare.csv")

print("REAL DATA")
print(real_df.describe(include='all'))

print("\nSYNTHETIC DATA")
print(synthetic_df.describe(include='all'))



# Load datasets
real_df = pd.read_csv("data/ctgan_ready.csv")
synthetic_df = pd.read_csv("data/synthetic_healthcare.csv")

# -------------------------
# Medical Condition Compare
# -------------------------
print("\n===== REAL CONDITION COUNTS =====")
print(real_df["Medical Condition"].value_counts())

print("\n===== SYNTHETIC CONDITION COUNTS =====")
print(synthetic_df["Medical Condition"].value_counts())

# -------------------------
# Gender Compare
# -------------------------
print("\n===== REAL GENDER COUNTS =====")
print(real_df["Gender"].value_counts())

print("\n===== SYNTHETIC GENDER COUNTS =====")
print(synthetic_df["Gender"].value_counts())

# -------------------------
# Age Statistics Compare
# -------------------------
print("\n===== REAL AGE STATISTICS =====")
print(real_df["Age"].describe())

print("\n===== SYNTHETIC AGE STATISTICS =====")
print(synthetic_df["Age"].describe())
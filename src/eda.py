import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/clean_healthcare.csv")

print(df.shape)
print(df.info())

print("\nMedical Condition Counts")
print(df["Medical Condition"].value_counts())

print("\nGender Counts")
print(df["Gender"].value_counts())

print("\nAdmission Type Counts")
print(df["Admission Type"].value_counts())

import os

# Create results folder if it doesn't exist
os.makedirs("results", exist_ok=True)

# Generate summary statistics
summary = df.describe(include="all")

# Save to CSV
summary.to_csv("results/statistical_summary.csv")

print("Statistical summary saved successfully!")
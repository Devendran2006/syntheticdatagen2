import pandas as pd
import os

real = pd.read_csv("data/ctgan_ready.csv")
synthetic = pd.read_csv("data/synthetic_healthcare.csv")

duplicates = synthetic.merge(
    real,
    how="inner"
)

duplicate_count = len(duplicates)

privacy_score = (
    1 - duplicate_count / len(synthetic)
)

print("\n===== PRIVACY SCORE =====")
print(round(privacy_score * 100, 2), "%")

os.makedirs("outputs", exist_ok=True)

with open(
    "outputs/privacy_score.txt",
    "w"
) as f:
    f.write(
        f"Privacy Score: {round(privacy_score*100,2)}%"
    )

print("\nDuplicate Records Found:")
print(duplicate_count)
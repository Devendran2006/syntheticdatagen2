import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

real = pd.read_csv("data/ctgan_ready.csv")
synthetic = pd.read_csv("data/synthetic_healthcare.csv")

for col in real.select_dtypes(include="object").columns:
    le = LabelEncoder()

    real[col] = le.fit_transform(real[col])

    synthetic[col] = le.fit_transform(
        synthetic[col]
    )

real_corr = real.corr(numeric_only=True)
synthetic_corr = synthetic.corr(numeric_only=True)

difference = (
    real_corr - synthetic_corr
).abs()

score = 1 - difference.mean().mean()

print("\n===== CORRELATION SIMILARITY =====")
print(round(score * 100, 2), "%")

os.makedirs("outputs", exist_ok=True)

with open(
    "outputs/correlation_score.txt",
    "w"
) as f:
    f.write(
        f"Correlation Similarity: {round(score*100,2)}%"
    )
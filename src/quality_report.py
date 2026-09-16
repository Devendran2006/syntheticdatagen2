import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdmetrics.reports.single_table import QualityReport
import os

os.makedirs("outputs", exist_ok=True)

real_data = pd.read_csv("data/ctgan_ready.csv")
synthetic_data = pd.read_csv("data/synthetic_healthcare.csv")

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)

report = QualityReport()

print("Generating Quality Report...")

report.generate(
    real_data,
    synthetic_data,
    metadata.to_dict()
)

score = report.get_score()

print("\n===== OVERALL QUALITY SCORE =====")
print(round(score * 100, 2), "%")

properties = report.get_properties()

print("\n===== PROPERTY SCORES =====")
print(properties)

with open(
    "outputs/quality_score.txt",
    "w"
) as f:
    f.write(f"Quality Score: {round(score*100,2)}%")

properties.to_csv(
    "outputs/quality_properties.csv",
    index=False
)

print("\nSaved to outputs/")
import pandas as pd

df = pd.read_csv("data/healthcare_dataset.csv")

pii_columns = [
    "Name",
    "Doctor",
    "Hospital",
    "Insurance Provider",
    "Room Number",
    "Date of Admission",
    "Discharge Date"
]

print("Detected PII Columns:")
for col in pii_columns:
    if col in df.columns:
        print(col)
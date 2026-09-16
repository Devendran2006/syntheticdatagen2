import pandas as pd

df = pd.read_csv("data/healthcare_dataset.csv")

# Create IDs
df["Patient_ID"] = ["P" + str(i).zfill(5) for i in range(1, len(df)+1)]

# Remove PII columns
pii_columns = [
    "Name",
    "Doctor",
    "Hospital"
]

df.drop(columns=pii_columns, inplace=True)

# Move Patient_ID to first column
cols = ["Patient_ID"] + [c for c in df.columns if c != "Patient_ID"]
df = df[cols]

df.to_csv("data/anonymized_healthcare.csv", index=False)

print("Anonymized dataset saved successfully.")
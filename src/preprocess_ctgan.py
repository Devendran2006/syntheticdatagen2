import pandas as pd

# Read anonymized dataset
df = pd.read_csv("data/anonymized_healthcare.csv")

# Convert date columns
df["Date of Admission"] = pd.to_datetime(df["Date of Admission"])
df["Discharge Date"] = pd.to_datetime(df["Discharge Date"])

# Create Length_of_Stay column
df["Length_of_Stay"] = (
    df["Discharge Date"] - df["Date of Admission"]
).dt.days

# Remove unnecessary columns
df.drop(
    columns=[
        "Patient_ID",
        "Date of Admission",
        "Discharge Date",
        "Room Number"
    ],
    inplace=True
)

# Save CTGAN-ready dataset
df.to_csv("data/ctgan_ready.csv", index=False)

print("ctgan_ready.csv created successfully!")
print(df.head())
print(df.shape)
import pandas as pd
df=pd.read_csv("data/anonymized_healthcare.csv")
print(df.isnull().sum())

print(df.duplicated().sum())
print(df.dtypes)
df["Date of Admission"] = pd.to_datetime(df["Date of Admission"])
df["Discharge Date"] = pd.to_datetime(df["Discharge Date"])
df.to_csv("data/clean_healthcare.csv", index=False)
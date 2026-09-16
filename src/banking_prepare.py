import pandas as pd
import os

# ==========================================
# PATHS
# ==========================================

CUSTOMERS_PATH = "data/banking/Customers.csv"
LOANS_PATH = "data/banking/Loans.csv"
TRANSACTIONS_PATH = "data/banking/Transactions.csv"

OUTPUT_DIR = "data/banking"
OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "banking_combined.csv"
)

# ==========================================
# LOAD DATA
# ==========================================

customers = pd.read_csv(CUSTOMERS_PATH)
loans = pd.read_csv(LOANS_PATH)
transactions = pd.read_csv(TRANSACTIONS_PATH)

print("Customers:", customers.shape)
print("Loans:", loans.shape)
print("Transactions:", transactions.shape)

# ==========================================
# LOAN AGGREGATION
# ==========================================

loan_summary = loans.groupby("customer_id").agg(
    loan_count=("loan_id", "count"),
    total_loan_amount=("loan_amount", "sum"),
    avg_loan_amount=("loan_amount", "mean"),
    avg_income=("income", "mean"),
    avg_term_months=("term_months", "mean"),
    avg_dti_ratio=("dti_ratio", "mean"),
    avg_credit_score=("credit_score", "mean"),
    avg_employment_years=("employment_years", "mean"),
    avg_utilisation=("utilisation", "mean"),
    total_missed_payments=("missed_payments", "sum"),
    default_count=("default", "sum")
).reset_index()

# ==========================================
# TRANSACTION AGGREGATION
# ==========================================

transaction_summary = transactions.groupby("customer_id").agg(
    transaction_count=("txn_id", "count"),
    total_transaction_amount=("amount", "sum"),
    avg_transaction_amount=("amount", "mean"),
    fraud_count=("is_fraud", "sum"),
    new_device_count=("device_new", lambda x: (x == "Yes").sum())
).reset_index()

# ==========================================
# MOST COMMON TRANSACTION CATEGORIES
# ==========================================

merchant_mode = (
    transactions.groupby("customer_id")["merchant_category"]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
    .reset_index()
    .rename(columns={
        "merchant_category": "primary_merchant_category"
    })
)

channel_mode = (
    transactions.groupby("customer_id")["channel"]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
    .reset_index()
    .rename(columns={
        "channel": "primary_channel"
    })
)

country_mode = (
    transactions.groupby("customer_id")["country"]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
    .reset_index()
    .rename(columns={
        "country": "primary_country"
    })
)

# ==========================================
# MERGE EVERYTHING
# ==========================================

combined = customers.merge(
    loan_summary,
    on="customer_id",
    how="left"
)

combined = combined.merge(
    transaction_summary,
    on="customer_id",
    how="left"
)

combined = combined.merge(
    merchant_mode,
    on="customer_id",
    how="left"
)

combined = combined.merge(
    channel_mode,
    on="customer_id",
    how="left"
)

combined = combined.merge(
    country_mode,
    on="customer_id",
    how="left"
)

# ==========================================
# HANDLE CUSTOMERS WITHOUT LOANS/TRANSACTIONS
# ==========================================

numeric_columns = combined.select_dtypes(
    include=["number"]
).columns

combined[numeric_columns] = combined[numeric_columns].fillna(0)

categorical_columns = combined.select_dtypes(
    include=["object"]
).columns

for col in categorical_columns:
    if col != "customer_id":
        combined[col] = combined[col].fillna("Unknown")

# ==========================================
# SAVE
# ==========================================

combined.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n===================================")
print("BANKING DATASET CREATED")
print("===================================")

print("Shape:", combined.shape)

print("\nColumns:")
print(combined.columns.tolist())

print("\nSaved to:")
print(OUTPUT_PATH)

print("\nFirst 5 records:")
print(combined.head())

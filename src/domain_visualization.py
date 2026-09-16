
import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================================================
# PATHS
# =========================================================

DOMAINS = {
    "Healthcare": {
        "real": "data/healthcare_dataset.csv",
        "synthetic": "data/synthetic_healthcare.csv",
        "output": "data/healthcare/comparison/charts"
    },

    "Banking": {
        "real": "data/banking/banking_combined.csv",
        "synthetic": "data/banking/synthetic_banking.csv",
        "output": "data/banking/comparison/charts"
    },

    "Enterprise": {
        "real": "data/enterprise/enterprise_combined.csv",
        "synthetic": "data/enterprise/synthetic_enterprise.csv",
        "output": "data/enterprise/comparison/charts"
    }
}


# =========================================================
# NUMERIC COLUMNS TO COMPARE
# =========================================================

NUMERIC_COLUMNS = {
    "Healthcare": [
        "Age",
        "Billing Amount"
    ],

    "Banking": [
        "age",
        "tenure_years",
        "avg_balance",
        "monthly_logins",
        "products_held",
        "loan_count",
        "total_loan_amount",
        "avg_income",
        "avg_credit_score",
        "avg_utilisation",
        "transaction_count",
        "total_transaction_amount"
    ],

    "Enterprise": [
        "age",
        "tenure_years",
        "salary",
        "overtime_pct",
        "engagement_score",
        "project_count",
        "avg_planned_days",
        "avg_actual_days",
        "total_budget",
        "total_spend",
        "avg_team_size"
    ]
}


# =========================================================
# FUNCTION
# =========================================================

def create_domain_charts(domain, config):

    print()
    print("===================================")
    print(f"{domain.upper()} VISUALIZATION")
    print("===================================")

    real = pd.read_csv(config["real"])
    synthetic = pd.read_csv(config["synthetic"])

    output_dir = config["output"]
    os.makedirs(output_dir, exist_ok=True)

    columns = NUMERIC_COLUMNS[domain]

    # Keep only columns existing in both datasets
    columns = [
        col for col in columns
        if col in real.columns and col in synthetic.columns
    ]

    print("Columns used:")
    print(columns)

    # =====================================================
    # 1. REAL VS SYNTHETIC MEAN
    # =====================================================

    real_means = []
    synthetic_means = []

    for col in columns:
        real_means.append(real[col].mean())
        synthetic_means.append(synthetic[col].mean())

    comparison = pd.DataFrame({
        "Column": columns,
        "Real": real_means,
        "Synthetic": synthetic_means
    })

    # Normalize each column so different units
    # can be compared in one chart
    normalized = []

    for _, row in comparison.iterrows():

        maximum = max(
            abs(row["Real"]),
            abs(row["Synthetic"]),
            1
        )

        normalized.append({
            "Column": row["Column"],
            "Real": row["Real"] / maximum,
            "Synthetic": row["Synthetic"] / maximum
        })

    normalized_df = pd.DataFrame(normalized)

    plt.figure(figsize=(14, 7))

    x = range(len(normalized_df))

    width = 0.35

    plt.bar(
        [i - width / 2 for i in x],
        normalized_df["Real"],
        width,
        label="Real"
    )

    plt.bar(
        [i + width / 2 for i in x],
        normalized_df["Synthetic"],
        width,
        label="Synthetic"
    )

    plt.xticks(
        list(x),
        normalized_df["Column"],
        rotation=45,
        ha="right"
    )

    plt.ylabel("Normalized Mean")
    plt.title(f"{domain} - Real vs Synthetic Mean Comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()

    path = os.path.join(
        output_dir,
        "real_vs_synthetic_mean.png"
    )

    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved:", path)

    # =====================================================
    # 2. DISTRIBUTION COMPARISON
    # =====================================================

    for col in columns:

        plt.figure(figsize=(9, 5))

        plt.hist(
            real[col].dropna(),
            bins=30,
            alpha=0.5,
            label="Real"
        )

        plt.hist(
            synthetic[col].dropna(),
            bins=30,
            alpha=0.5,
            label="Synthetic"
        )

        plt.xlabel(col)
        plt.ylabel("Frequency")

        plt.title(
            f"{domain} - {col}\nReal vs Synthetic Distribution"
        )

        plt.legend()
        plt.grid(axis="y", alpha=0.2)
        plt.tight_layout()

        safe_name = (
            col.replace(" ", "_")
               .replace("/", "_")
        )

        path = os.path.join(
            output_dir,
            f"{safe_name}_distribution.png"
        )

        plt.savefig(path, dpi=300)
        plt.close()

    # =====================================================
    # 3. SUMMARY TABLE
    # =====================================================

    comparison["Mean Difference %"] = (
        abs(
            comparison["Real"]
            - comparison["Synthetic"]
        )
        / comparison["Real"].replace(0, 1)
        * 100
    )

    comparison.to_csv(
        os.path.join(
            output_dir,
            "visual_comparison_summary.csv"
        ),
        index=False
    )

    print("Summary saved.")

    print()
    print(f"{domain} charts completed.")


# =========================================================
# RUN ALL DOMAINS
# =========================================================

for domain, config in DOMAINS.items():

    try:

        create_domain_charts(
            domain,
            config
        )

    except Exception as e:

        print()
        print(
            f"ERROR in {domain}: {e}"
        )


print()
print("===================================")
print("ALL DOMAIN VISUALIZATIONS COMPLETE")
print("===================================")

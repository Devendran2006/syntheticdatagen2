import pandas as pd
import os

# =========================================================
# PATHS
# =========================================================

HEALTHCARE_PATH = "data/healthcare/comparison/synthetic_comparison.csv"
BANKING_PATH = "data/banking/comparison/synthetic_comparison.csv"
ENTERPRISE_PATH = "data/enterprise/comparison/synthetic_comparison.csv"

OUTPUT_DIR = "data/comparison"
OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "domain_comparison.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# FUNCTION TO LOAD DOMAIN RESULT
# =========================================================

def load_domain(path, domain_name):

    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return None

    df = pd.read_csv(path)

    if df.empty:
        print(f"WARNING: {path} is empty")
        return None

    row = df.iloc[0]

    return {
        "Domain": domain_name,
        "Distribution Score": float(
            row["Distribution Score"]
        ),
        "Correlation Score": float(
            row["Correlation Score"]
        ),
        "Overall Similarity Score": float(
            row["Overall Similarity Score"]
        ),
        "Real Records": int(
            row["Real Records"]
        ),
        "Synthetic Records": int(
            row["Synthetic Records"]
        )
    }


# =========================================================
# LOAD ALL THREE DOMAINS
# =========================================================

healthcare = load_domain(
    HEALTHCARE_PATH,
    "Healthcare"
)

banking = load_domain(
    BANKING_PATH,
    "Banking"
)

enterprise = load_domain(
    ENTERPRISE_PATH,
    "Enterprise"
)


# =========================================================
# CREATE RESULTS
# =========================================================

results = []

for result in [
    healthcare,
    banking,
    enterprise
]:

    if result is not None:
        results.append(result)


if not results:

    print("No domain comparison results found.")
    exit()


comparison = pd.DataFrame(results)


# =========================================================
# ROUND SCORES
# =========================================================

score_columns = [
    "Distribution Score",
    "Correlation Score",
    "Overall Similarity Score"
]

comparison[score_columns] = comparison[
    score_columns
].round(2)


# =========================================================
# SAVE
# =========================================================

comparison.to_csv(
    OUTPUT_PATH,
    index=False
)


# =========================================================
# DISPLAY
# =========================================================

print()
print("===================================")
print("CROSS-DOMAIN COMPARISON")
print("===================================")

print(
    comparison.to_string(index=False)
)


# =========================================================
# BEST DOMAIN
# =========================================================

best_index = comparison[
    "Overall Similarity Score"
].idxmax()

best_domain = comparison.loc[
    best_index,
    "Domain"
]

best_score = comparison.loc[
    best_index,
    "Overall Similarity Score"
]


# =========================================================
# AVERAGE SCORE
# =========================================================

average_score = comparison[
    "Overall Similarity Score"
].mean()


print()
print("===================================")
print("CROSS-DOMAIN SUMMARY")
print("===================================")

print(
    f"Best Domain      : {best_domain}"
)

print(
    f"Best Score       : {best_score:.2f}%"
)

print(
    f"Average Similarity: {average_score:.2f}%"
)


# =========================================================
# SAVE
# =========================================================

print()
print("Saved to:")
print(OUTPUT_PATH)
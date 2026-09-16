import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================================================
# PATHS
# =========================================================

INPUT_PATH = "data/comparison/domain_comparison.csv"
OUTPUT_DIR = "data/comparison/charts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(INPUT_PATH)

print("Domain Comparison Data:")
print(df.to_string(index=False))

# =========================================================
# 1. OVERALL SIMILARITY
# =========================================================

plt.figure(figsize=(9, 5))

plt.bar(
    df["Domain"],
    df["Overall Similarity Score"]
)

plt.title(
    "Synthetic Data Overall Similarity by Domain",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Domain")
plt.ylabel("Similarity Score (%)")

plt.ylim(0, 100)

for i, value in enumerate(
    df["Overall Similarity Score"]
):
    plt.text(
        i,
        value + 1,
        f"{value:.2f}%",
        ha="center",
        fontweight="bold"
    )

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/overall_similarity.png",
    dpi=300
)

plt.show()

# =========================================================
# 2. DISTRIBUTION SCORE
# =========================================================

plt.figure(figsize=(9, 5))

plt.bar(
    df["Domain"],
    df["Distribution Score"]
)

plt.title(
    "Synthetic Data Distribution Score by Domain",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Domain")
plt.ylabel("Distribution Score (%)")

plt.ylim(0, 100)

for i, value in enumerate(
    df["Distribution Score"]
):
    plt.text(
        i,
        value + 1,
        f"{value:.2f}%",
        ha="center",
        fontweight="bold"
    )

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/distribution_score.png",
    dpi=300
)

plt.show()

# =========================================================
# 3. CORRELATION SCORE
# =========================================================

plt.figure(figsize=(9, 5))

plt.bar(
    df["Domain"],
    df["Correlation Score"]
)

plt.title(
    "Synthetic Data Correlation Score by Domain",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Domain")
plt.ylabel("Correlation Score (%)")

plt.ylim(0, 100)

for i, value in enumerate(
    df["Correlation Score"]
):
    plt.text(
        i,
        value + 1,
        f"{value:.2f}%",
        ha="center",
        fontweight="bold"
    )

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/correlation_score.png",
    dpi=300
)

plt.show()

# =========================================================
# SUMMARY
# =========================================================

best_domain = df.loc[
    df["Overall Similarity Score"].idxmax(),
    "Domain"
]

best_score = df[
    "Overall Similarity Score"
].max()

print()
print("===================================")
print("DOMAIN CHARTS GENERATED")
print("===================================")

print(
    f"Best Performing Domain : {best_domain}"
)

print(
    f"Overall Similarity     : {best_score:.2f}%"
)

print()
print("Charts saved in:")
print(OUTPUT_DIR)
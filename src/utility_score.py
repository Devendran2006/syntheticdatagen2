import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# =========================================================
# PATHS
# =========================================================

REAL_PATH = "data/ctgan_ready.csv"
SYNTHETIC_PATH = "data/synthetic_healthcare.csv"

OUTPUT_DIR = "outputs"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# LOAD DATA
# =========================================================

real = pd.read_csv(REAL_PATH)
synthetic = pd.read_csv(SYNTHETIC_PATH)


# =========================================================
# TARGET
# =========================================================

target = "Medical Condition"


# =========================================================
# PREPARE DATA
# =========================================================

common_columns = [
    col
    for col in real.columns
    if col in synthetic.columns
    and col != target
]


X_real = real[common_columns].copy()
X_synthetic = synthetic[common_columns].copy()

y_real = real[target].copy()
y_synthetic = synthetic[target].copy()


# =========================================================
# ENCODE CATEGORICAL FEATURES
# =========================================================

combined_X = pd.concat(
    [X_real, X_synthetic],
    axis=0
)

combined_X = pd.get_dummies(
    combined_X,
    drop_first=False
)

X_real = combined_X.iloc[
    :len(X_real)
].copy()

X_synthetic = combined_X.iloc[
    len(X_real):
].copy()


# =========================================================
# HANDLE MISSING VALUES
# =========================================================

X_real = X_real.fillna(0)
X_synthetic = X_synthetic.fillna(0)


# =========================================================
# TRAIN MODEL ON REAL DATA
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_real,
    y_real,
    test_size=0.2,
    random_state=42,
    stratify=y_real
)


model_real = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model_real.fit(
    X_train,
    y_train
)

real_predictions = model_real.predict(
    X_test
)

real_accuracy = accuracy_score(
    y_test,
    real_predictions
)


# =========================================================
# TRAIN MODEL ON SYNTHETIC DATA
# =========================================================

synthetic_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

synthetic_model.fit(
    X_synthetic,
    y_synthetic
)

synthetic_predictions = synthetic_model.predict(
    X_test
)

synthetic_accuracy = accuracy_score(
    y_test,
    synthetic_predictions
)


# =========================================================
# UTILITY RETENTION
# =========================================================

if real_accuracy > 0:

    utility_score = (
        synthetic_accuracy
        / real_accuracy
    ) * 100

else:

    utility_score = 0


utility_score = min(
    100,
    max(0, utility_score)
)


# =========================================================
# DISPLAY
# =========================================================

print()
print("===================================")
print("UTILITY RETENTION")
print("===================================")

print(
    "Real Model Accuracy:",
    round(real_accuracy * 100, 2),
    "%"
)

print(
    "Synthetic Model Accuracy:",
    round(synthetic_accuracy * 100, 2),
    "%"
)

print(
    "Utility Retention:",
    round(utility_score, 2),
    "%"
)


# =========================================================
# SAVE
# =========================================================

with open(
    "outputs/utility_score.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        f"Utility Retention: "
        f"{round(utility_score, 2)}%"
    )


print()
print(
    "Utility score saved to "
    "outputs/utility_score.txt"
)
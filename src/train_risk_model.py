import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

df = pd.read_csv("data/ctgan_ready.csv")

features = [
    "Age",
    "Gender",
    "Medical Condition",
    "Admission Type"
]

target = "Length_of_Stay"

X = df[features]
y = df[target]

categorical = [
    "Gender",
    "Medical Condition",
    "Admission Type"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical
        )
    ],
    remainder="passthrough"
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

pipeline.fit(X, y)

joblib.dump(
    pipeline,
    "models/risk_model.pkl"
)

print("Risk Model Saved")
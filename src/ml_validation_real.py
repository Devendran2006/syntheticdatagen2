import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

df = pd.read_csv("data/ctgan_ready.csv")

target = "Medical Condition"

X = df.drop(columns=[target])
y = df[target]

# Encode categorical columns
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Real Data Accuracy:",
      accuracy_score(y_test, pred))
print("Accuracy:",
      accuracy_score(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred))
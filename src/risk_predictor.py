import pandas as pd
import joblib

def predict_risk(age, gender, condition, admission):

    model = joblib.load("models/risk_model.pkl")
    encoder = joblib.load("models/risk_encoder.pkl")

    data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Medical Condition": [condition],
        "Admission Type": [admission]
    })

    data_encoded = encoder.transform(data)

    prediction = model.predict(data_encoded)[0]

    if prediction >= 15:
        risk = "High"

    elif prediction >= 8:
        risk = "Medium"

    else:
        risk = "Low"

    return {
        "Predicted Stay": round(prediction, 1),
        "Risk": risk
    }
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from schemas import CustomerInput

app = FastAPI(title="Churn Prediction API")

model = tf.keras.models.load_model("./best_churn_model.h5")
preprocessor = joblib.load("./preprocessor.pkl")

EXPECTED_COLUMNS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

@app.post("/predict")
def predict_churn(input_data: CustomerInput):
    df = pd.DataFrame([input_data.model_dump()])
    df = df[EXPECTED_COLUMNS]

    X_processed = preprocessor.transform(df)
    X_processed = np.array(X_processed)

    prob = model.predict(X_processed)[0][0]

    print(prob)

    return {
        "churn_probability": float(prob),
        "prediction": "Churn" if prob >= 0.5 else "No Churn"
    }


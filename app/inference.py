# inference.py
import joblib
import pandas as pd

# Load the model
model = joblib.load("../model/diabetes_prediction_model.pkl")

def predict(input_data: dict):
    """
    input_data: dict of features, e.g., {"age": 50, "bmi": 28.5, "gender": 1}
    """
    df = pd.DataFrame([input_data])
    preds = model.predict(df)
    return preds.tolist()

import os

class ModelWrapper:
    def __init__(self):
        model_path = "/app/model/diabetes_prediction_model.pkl"  # full path in container
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
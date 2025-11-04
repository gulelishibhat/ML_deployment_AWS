import pickle
import numpy as np

class ModelWrapper:
    def __init__(self, model_path="diabetes_prediction_model.pkl"):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, features):
        features = np.array(features).reshape(1, -1)
        return self.model.predict(features)[0]
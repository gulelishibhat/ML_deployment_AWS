import pickle
import os

class ModelWrapper:
    def __init__(self):
        model_path = os.path.join(os.environ.get("LAMBDA_TASK_ROOT", "/var/task"), "model", "diabetes_prediction_model.pkl")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, features):
        # Assuming your model expects a 2D array
        return self.model.predict([features])[0]
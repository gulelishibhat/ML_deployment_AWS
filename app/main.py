from inference import ModelWrapper

# Load your model once — Lambda reuses the same container for multiple invocations
model = ModelWrapper()

def lambda_handler(event, context):
    try:
        # Expecting a JSON input: {"features": [...]}
        features = event.get("features")
        if not features:
            return {"error": "Missing 'features' in input"}

        prediction = model.predict(features)
        return {"prediction": int(prediction)}

    except Exception as e:
        return {"error": str(e)}

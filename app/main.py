from flask import Flask, request, jsonify
from inference import ModelWrapper

app = Flask(__name__)
model = ModelWrapper()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    prediction = model.predict(data["features"])
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

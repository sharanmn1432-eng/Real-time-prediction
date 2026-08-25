from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

text_model_path = os.path.join(BASE_DIR, "models", "text_classifier.pkl")

text_model = None

if os.path.exists(text_model_path):
    with open(text_model_path, "rb") as f:
        text_model = pickle.load(f)


@app.route("/")
def home():
    return jsonify({
        "status": "success",
        "message": "Real-Time Prediction API is running",
        "endpoints": {
            "text_prediction": "/predict"
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    if text_model is None:
        return jsonify({
            "error": "Text model not found"
        }), 500

    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({
            "error": "Please provide text"
        }), 400

    text = data["text"]

    prediction = text_model.predict([text])[0]

    return jsonify({
        "input": text,
        "prediction": str(prediction)
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

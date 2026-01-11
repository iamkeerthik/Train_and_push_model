from flask import Flask, request, jsonify
import joblib
from pathlib import Path

app = Flask(__name__)
MODEL_PATH = Path("artifacts/model.pkl")

# Load trained model
if not MODEL_PATH.exists():
    raise FileNotFoundError("Model not found in artifacts. Build Docker after training!")

model = joblib.load(MODEL_PATH)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "send JSON with key 'features'"}), 400
    try:
        features = data["features"]
        pred = model.predict([features])
        return jsonify({"prediction": int(pred[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
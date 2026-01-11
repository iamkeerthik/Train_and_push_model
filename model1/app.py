from flask import Flask, request, jsonify
import joblib
from pathlib import Path

app = Flask(__name__)

MODEL_PATH = Path("artifacts/model.pkl")
LE_PATH = Path("artifacts/label_encoder.pkl")

# Load artifacts
if not MODEL_PATH.exists() or not LE_PATH.exists():
    import train as _train
    _train.main()

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LE_PATH)

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
        pred_int = model.predict([features])[0]
        pred_label = label_encoder.inverse_transform([pred_int])[0]
        return jsonify({"prediction": pred_label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
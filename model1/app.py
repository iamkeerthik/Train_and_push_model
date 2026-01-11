from flask import Flask, request, jsonify
import joblib
from pathlib import Path

app = Flask(__name__)
MODEL_PATH = Path("artifacts/model.pkl")
LE_PATH = Path("artifacts/label_encoder.pkl")

# Train if missing
if not MODEL_PATH.exists():
    import train as _train
    _train.main()

model = joblib.load(MODEL_PATH)

# Load label encoder if exists (for string labels)
le = joblib.load(LE_PATH) if LE_PATH.exists() else None

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
        # Convert back to string label if encoder exists
        if le is not None:
            pred_label = le.inverse_transform(pred)
            return jsonify({"prediction": str(pred_label[0])})
        return jsonify({"prediction": int(pred[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
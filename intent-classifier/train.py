import os
import joblib
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def main():
    # Pull dataset from DVC
    if os.path.exists("data/intent.csv.dvc"):
        os.system("dvc pull data/intent.csv")

    # Load dataset
    df = pd.read_csv("data/intent.csv")
    X = df["text"].tolist()
    y_raw = df["label"].tolist()

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Save label encoder for inference
    os.makedirs("model/artifacts", exist_ok=True)
    joblib.dump(le, "model/artifacts/label_encoder.pkl")

    # Train pipeline
    pipeline = Pipeline([
        ("vect", CountVectorizer()),
        ("clf", MultinomialNB())
    ])
    pipeline.fit(X, y)

    # Save trained model
    joblib.dump(pipeline, "model/artifacts/intent_model.pkl")

    # Save metrics (accuracy on same data as quick check)
    acc = pipeline.score(X, y)
    with open("model/artifacts/metrics.json", "w") as f:
        json.dump({"accuracy": float(acc)}, f)

    print(f"✅ Model saved to model/artifacts/intent_model.pkl")
    print(f"Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
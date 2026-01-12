import os
import joblib
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

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

    # Train pipeline and bundle label encoder inside
    pipeline = Pipeline([
        ("vect", CountVectorizer()),
        ("clf", MultinomialNB()),
        ("label_encoder", le)  # We include encoder as an attribute
    ])
    pipeline.fit(X, y)

    # Save trained pipeline (model + encoder together)
    os.makedirs("model/artifacts", exist_ok=True)
    joblib.dump({
        "model_pipeline": pipeline,
        "label_encoder": le
    }, "model/artifacts/intent_model.pkl")

    # Quick accuracy check
    acc = pipeline.score(X, y)
    with open("model/artifacts/metrics.json", "w") as f:
        json.dump({"accuracy": float(acc)}, f)

    print(f"✅ Model saved to model/artifacts/intent_model.pkl")
    print(f"Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
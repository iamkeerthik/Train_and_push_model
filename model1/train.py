"""
Simple training script:
- pulls data using DVC
- encodes labels as integers
- trains LogisticRegression
- saves model and metrics to artifacts/
"""

import os
import joblib
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

def main():
    # Pull data from DVC
    if os.path.exists("data/iris.csv.dvc"):
        os.system("dvc pull data/iris.csv")

    # Load dataset
    csv_path = "data/iris.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        X = df.iloc[:, :-1].values
        y_raw = df.iloc[:, -1].values
    else:
        iris = load_iris()
        X = iris.data
        y_raw = iris.target  # already integers

    # Encode string labels if needed
    if y_raw.dtype.kind in {'U', 'O'}:  # string/object
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        # save label encoder for inference
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(le, "artifacts/label_encoder.pkl")
    else:
        y = y_raw

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Save model
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/model.pkl")

    # Save metrics
    acc = model.score(X_test, y_test)
    with open("artifacts/metrics.json", "w") as f:
        json.dump({"accuracy": float(acc)}, f)

    print(f"✅ Saved model to artifacts/model.pkl")
    print(f"Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
"""
Simple training script:
- pulls data using DVC
- trains LogisticRegression
- saves model and metrics
"""

import os
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import pandas as pd

# Ensure paths are relative to the script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    # Pull training data with DVC
    dvc_csv = os.path.join(BASE_DIR, "data/iris.csv.dvc")
    if os.path.exists(dvc_csv):
        os.system(f"cd {BASE_DIR} && dvc pull data/iris.csv")  # fetch from S3

    # Load dataset (using CSV if exists, otherwise sklearn iris)
    csv_path = os.path.join(BASE_DIR, "data/iris.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    else:
        iris = load_iris()
        X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Save model
    artifacts_dir = os.path.join(BASE_DIR, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "model.pkl")
    joblib.dump(model, model_path)

    # Save metrics
    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    metrics = {"accuracy": float(model.score(X_test, y_test))}
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    print(f"Saved model to {model_path}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()
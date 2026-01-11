import os
import joblib
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

def main():
    # Pull training data from DVC
    if os.path.exists("data/iris.csv.dvc"):
        os.system("dvc pull data/iris.csv")

    csv_path = "data/iris.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    else:
        from sklearn.datasets import load_iris
        iris = load_iris()
        X, y = iris.data, iris.target

    # Encode string labels to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Save artifacts (model + label encoder + metrics)
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/model.pkl")
    joblib.dump(le, "artifacts/label_encoder.pkl")

    acc = model.score(X_test, y_test)
    with open("artifacts/metrics.json", "w") as f:
        json.dump({"accuracy": float(acc)}, f)

    print(f"Saved model to artifacts/model.pkl and label_encoder.pkl")
    print(f"Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
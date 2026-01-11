# MLOps Model Training & Deployment Pipeline

This repository contains an automated end-to-end MLOps pipeline for training **Model1** and deploying it via Docker to Amazon ECR. It utilizes **DVC** for data version control and **GitHub Actions** for CI/CD.

## 📂 Project Structure

```text
.
├── .github
│   └── workflows
│       ├── release.yaml        # Handles version tagging and releases
│       └── train_push.yaml    # CI/CD: Training, Docker build, and ECR push
├── .gitignore
└── model1
    ├── Dockerfile             # Containerization for model serving
    ├── app.py                 # Application entry point (API/Inference)
    ├── artifacts/             # Local storage for trained models/pickles
    ├── data/
    │   ├── .gitignore         # Prevents raw data from being committed to Git
    │   ├── iris.csv           # Raw data file (tracked by DVC) (.gitignored while pushing to github)
    │   └── iris.csv.dvc       # DVC metadata for the dataset
    ├── requirements.txt       # Python dependencies
    └── train.py               # Model training script
```
## 🛠️ DVC Setup & Data Management

Since we don't store raw data (like `iris.csv`) in Git, we use DVC with an S3 backend.

### 1. Initialize DVC (One-time setup)
If you are starting from scratch:
```bash
dvc init
```
### 2. Configure S3 Remote
Replace my-s3-bucket/ml-data with your actual AWS bucket path:
```bash
dvc remote add -d mys3remote s3://my-s3-bucket/ml-data
dvc remote modify mys3remote region us-east-1
```
### 3. Tracking Data
To track new data or changes:
```bash
cd model1/data
# Add the data file to DVC
dvc add iris.csv

# Commit the .dvc file to Git
git add iris.csv.dvc .gitignore
git commit -m "Update dataset tracking"

# Push the actual data to S3
dvc push
```

## 🚀 CI/CD Pipeline (GitHub Actions)
The workflows in .github/workflows/ automate the following:

1. Training: When code is pushed, the train_push.yaml workflow triggers.

2. DVC Pull: The runner pulls the dataset from S3 using DVC.

3. Build: A Docker image is built using model1/Dockerfile.

4. Push: The resulting image is pushed to Amazon ECR.

## Required GitHub Secrets
To make the workflows function, add these secrets to your repository:

* AWS_ACCESS_KEY_ID 

* AWS_SECRET_ACCESS_KEY

* AWS_REGION

* ECR_REPOSITORY_NAME
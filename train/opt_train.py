import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

import ray
from ray.util.joblib import register_ray

# Constants
TARGET_COLUMN = "future_accidents_6m"
MODEL_NAME = "CrashModel"

@ray.remote
def load_csv(file):
    try:
        df = pd.read_csv(file, parse_dates=True)
        return df
    except Exception as e:
        print(f"[ERROR] Failed to read {file}: {e}")
        return pd.DataFrame()

def preprocess(df):
    #df = df.dropna()
    df["intersection_id"] = df["intersection_id"].astype("category").cat.codes
    #df["accidents_6m"] = df["accidents_6m"].astype(int)
    #df["accidents_1y"] = df["accidents_1y"].astype(int)
    #df["accidents_5y"] = df["accidents_5y"].astype(int)
    #df["YEAR"] = df["YEAR"].astype(int)
    features = ["intersection_id", "accidents_6m", "accidents_1y", "accidents_5y"]
    X = df[features]
    y = df[TARGET_COLUMN]
    return X, y

@ray.remote(num_cpus=1, memory=1 * 1024 * 1024 * 1024)
def train_fold(X_train, X_val, y_train, y_val, config):
    rf = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        random_state=42
    )
    rf.fit(X_train, y_train)
    preds = rf.predict(X_val)
    return {
        "accuracy": accuracy_score(y_val, preds),
        "precision": precision_score(y_val, preds, zero_division=0, average="weighted"),
        "recall": recall_score(y_val, preds, zero_division=0, average="weighted"),
        "f1": f1_score(y_val, preds, zero_division=0, average="weighted")
    }

if __name__ == "__main__":
    ray.init(address="auto")

    mlflow.set_tracking_uri("http://10.56.2.49:8000")
    mlflow.set_experiment("VisionZeroCrashModel")
    mlflow.sklearn.autolog()

    # Load data in parallel
    files = [
        "processed_2018.csv", "processed_2019.csv", "processed_2020.csv",
        "processed_2021.csv", "processed_2022.csv", "processed_2023.csv",
        "processed_2024.csv"
    ]
    futures = [load_csv.remote(file) for file in files]
    dfs = ray.get(futures)
    df = pd.concat(dfs, ignore_index=True)
    
    X, y = preprocess(df)
    tscv = TimeSeriesSplit(n_splits=3)

    config = {"n_estimators": 100, "max_depth": 10}
    results = []

    with mlflow.start_run() as run:
        fold_jobs = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            job = train_fold.remote(X_train, X_val, y_train, y_val, config)
            fold_jobs.append(job)

        metrics = ray.get(fold_jobs)
        avg_metrics = {k: sum([m[k] for m in metrics]) / len(metrics) for k in metrics[0]}

        for key, value in avg_metrics.items():
            mlflow.log_metric(f"avg_{key}", value)

        client = MlflowClient()
        model_uri = f"runs:/{run.info.run_id}/model"
        registered_model = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="development",
            version=registered_model.version
        )

        print("Logged and registered model with metrics:", avg_metrics)

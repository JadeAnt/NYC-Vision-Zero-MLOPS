import os
import numpy
import pandas as pd
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import joblib

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

import ray
from ray import tune
from ray.util.joblib import register_ray
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

# Constants
TARGET_COLUMN = "future_accidents_6m"
MODEL_NAME = "CrashModel"

@ray.remote
def load_data():
    files = [
        "processed_2018.csv", "processed_2019.csv", "processed_2020.csv",
        "processed_2021.csv", "processed_2022.csv", "processed_2023.csv",
        "processed_2024.csv"
    ]
    dfs = []
    for file in files:
        try:
            df = pd.read_csv(file, parse_dates=True)
            dfs.append(df)
        except Exception as e:
            print(f"[ERROR] Failed to read {file}: {e}")
    if not dfs:
        raise ValueError("No CSV files were loaded.")
    return pd.concat(dfs, ignore_index=True)

def preprocess(df):
    df = df.dropna()
    features = ["intersection_id","accidents_6m","accidents_1y","accidents_5y","YEAR"]
    #X = df.drop(columns=[TARGET_COLUMN])
    df["intersection_id"] = df["intersection_id"].astype("category").cat.codes
    df["accidents_6m"] = df["accidents_6m"].astype(int)
    df["accidents_1y"] = df["accidents_1y"].astype(int)
    df["accidents_5y"] = df["accidents_5y"].astype(int)
    df["YEAR"] = df["YEAR"].astype(int)
    X = df[features]
    y = df[TARGET_COLUMN]
    return X, y

def train_and_log_model(X, y):
    mlflow.set_experiment("VisionZeroCrashModel")
    with mlflow.start_run():
        tscv = TimeSeriesSplit(n_splits=5)
        accuracies = []

        # Fixed hyperparameters
        config = {
            "n_estimators": 100,
            "max_depth": 20,
        }

        for train_index, val_index in tscv.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            rf = RandomForestClassifier(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                random_state=42
            )
            rf.fit(X_train, y_train)
            preds = rf.predict(X_val)
            acc = accuracy_score(y_val, preds)
            accuracies.append(acc)

        avg_accuracy = sum(accuracies) / len(accuracies)

        # Log to MLflow
        mlflow.log_params(config)
        mlflow.log_metric("avg_accuracy", avg_accuracy)
        mlflow.sklearn.log_model(rf, artifact_path="model", registered_model_name=MODEL_NAME)

        print(f"Average Accuracy: {avg_accuracy:.4f}")

def main():
    ray.init(address="auto", runtime_env={"env_vars": {"RAY_memory_usage_threshold": "0.9"}})  # Ensure we connect to the cluster

    for node in ray.nodes():
        print(f"Node: {node['NodeManagerAddress']}, Alive: {node['Alive']}, Resources: {node['Resources']}")

    mlflow.set_tracking_uri("http://10.56.2.49:8000")

    df = ray.get(load_data.remote())
    X, y = preprocess(df)
    train_and_log_model(X, y)

if __name__ == "__main__":
    main()

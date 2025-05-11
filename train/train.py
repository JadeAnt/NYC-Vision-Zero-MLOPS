import os
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
DATA_DIR = "/mnt/object"
YEAR_FOLDERS = [f"year_{y}" for y in range(2018, 2025)]
TARGET_COLUMN = "future_accidents_6m"
MODEL_NAME = "CrashModel"

def load_data():
    #print("Files:", os.listdir(DATA_DIR))
    print(os.system('ls -R /mnt/object'))
    print(os.system('df -h'))
    dfs = []
    for year in YEAR_FOLDERS:
        path = os.path.join(DATA_DIR, year, "*.csv")
        files = glob(path)
        if not files:
            print(f"[WARNING] No CSV files found in {path}")
        for file in files:
            try:
                df = pd.read_csv(file, parse_dates=True)
                df["__year"] = year  # For debugging or tracking
                dfs.append(df)
            except Exception as e:
                print(f"[ERROR] Failed to read {file}: {e}")
    if not dfs:
        raise ValueError("No CSV files were loaded. Please check the data directory and file paths.")
    data = pd.concat(dfs, ignore_index=True)
    return data

def preprocess(df):
    df = df.dropna()
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y

def train_model(config, X, y):
    mlflow.set_experiment("VisionZeroCrashModel")
    with mlflow.start_run():
        tscv = TimeSeriesSplit(n_splits=5)
        accuracies = []

        for train_index, val_index in tscv.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            rf = RandomForestClassifier(
                n_estimators=int(config["n_estimators"]),
                max_depth=int(config["max_depth"]),
                min_samples_split=config["min_samples_split"],
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            preds = rf.predict(X_val)
            acc = accuracy_score(y_val, preds)
            accuracies.append(acc)

        avg_accuracy = sum(accuracies) / len(accuracies)
        mlflow.log_params(config)
        mlflow.log_metric("avg_accuracy", avg_accuracy)
        mlflow.sklearn.log_model(rf, artifact_path="model", registered_model_name=MODEL_NAME)
        tune.report(accuracy=avg_accuracy)

def main():
    ray.init()

    register_ray()

    mlflow.set_tracking_uri("http://10.56.2.49:8000")  # Update as needed
    df = load_data()
    X, y = preprocess(df)

    # Hyperparameter search space
    search_space = {
        "n_estimators": tune.randint(50, 200),
        "max_depth": tune.randint(10, 30),
        "min_samples_split": tune.uniform(0.01, 0.3),
    }

    # Scheduler and search algorithm
    scheduler = ASHAScheduler(metric="accuracy", mode="max")
    search_alg = OptunaSearch(metric="accuracy", mode="max")

    # Run tuning
    tuner = tune.Tuner(
        tune.with_parameters(train_model, X=X, y=y),
        tune_config=tune.TuneConfig(
            metric="accuracy",
            mode="max",
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=10,
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    best_result = results.get_best_result(metric="accuracy", mode="max")
    print("Best hyperparameters found were: ", best_result.config)

if __name__ == "__main__":
    main()

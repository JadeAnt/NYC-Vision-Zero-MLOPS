import os
import pandas as pd
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import joblib

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

import ray
from ray import tune
from ray.tune.sklearn import TuneSearchCV
from ray.util.joblib import register_ray


#MODEL_PATH = "crash_model.joblib"
MODEL_NAME = "CrashModel"

DATA_DIR = "/mnt/object"
YEAR_FOLDERS = [f"year_{y}" for y in range(2018, 2025)]
TARGET_COLUMN = "future_accidents_6m"

def load_data():
    dfs = []
    for year in YEAR_FOLDERS:
        path = os.path.join(DATA_DIR, year, "*.csv")
        for file in glob(path):
            df = pd.read_csv(file, parse_dates=True)
            df["__year"] = year  #for debugging or tracking
            dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    return data

def preprocess(df):
    df = df.dropna()
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y

def train_model(X, y):
    mlflow.set_experiment("Timeseries_RF_Tune")

    # Create TimeSeries Split
    tscv = TimeSeriesSplit(n_splits=5)

    # Ray Tune tuning with tune-sklearn
    param_grid = {
        "n_estimators": tune.randint(50, 200),
        "max_depth": tune.randint(10, 30),
        "min_samples_split": tune.uniform(0.01, 0.3),
    }

    rf = RandomForestClassifier(random_state=42)

    # Wrap with TuneSearchCV
    tune_search = TuneSearchCV(
        rf,
        param_distributions=param_grid,
        n_trials=10,
        scoring="accuracy",
        cv=tscv,
        verbose=1,
        local_dir="ray_results",
        loggers=None,
        random_state=42
    )

    # Train and log to MLflow
    with mlflow.start_run() as run:
        tune_search.fit(X, y)

        best_model = tune_search.best_estimator_
        best_params = tune_search.best_params_
        best_score = tune_search.best_score_


        print(f"[INFO] Logged model to MLflow run: {run.info.run_id}")
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", best_score)
        mlflow.sklearn.log_model(
            sk_model = rf,
            artifact_path = "crash_model",
            registered_model_name= MODEL_NAME
        )

        #client = MlflowClient()
        #model_uri = f"runs:/{run.info.run_id}/model"
        #registered_model = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

        #client.set_registered_model_alias(
        #    name=MODEL_NAME,
        #    alias="development",
        #    version=registered_model.version
        #)

        print(f"Best Score: {best_score}")
        print(f"Best Params: {best_params}")

if __name__ == "__main__":
    ray.init()

    register_ray()

    mlflow.set_tracking_uri("http://10.56.2.49:8000") # change later to not hard code like this
    mlflow.sklearn.autolog()
    df = load_data()
    X, y = preprocess(df)
    train_model(X, y)
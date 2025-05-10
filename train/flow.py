import os
import joblib
import mlflow
import ray
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from ray.util.joblib import register_ray
from mlflow.tracking import MlflowClient


MODEL_PATH = "crash_model.joblib"
MODEL_NAME = "CrashModel"

def main():
# Initialize Ray
    ray.init()

    # Register Ray with joblib
    register_ray()

    # Load dataset
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Enable MLflow autologging for scikit-learn
    mlflow.sklearn.autolog()

    print("Testing loading our model")

    model = joblib.load(open(MODEL_PATH, "rb"))
    print(model.__getstate__)

    print("Training other model")

    # Start an MLflow run
    with mlflow.start_run() as run:
        # Train model with Ray's joblib backend
        with joblib.parallel_backend('ray'):
            model = RandomForestClassifier(n_estimators=100)
            model.fit(X_train, y_train)

        # Evaluate model
        accuracy = model.score(X_test, y_test)
        loss = 1 - accuracy  # Simplified loss

        # Log additional metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("loss", loss)

        print(f"[INFO] Logged model to MLflow run: {run.info.run_id}")

        # Register model if accuracy meets threshold
        if accuracy >= 0.80:
            client = MlflowClient()
            model_uri = f"runs:/{run.info.run_id}/model"
            registered_model = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

            client.set_registered_model_alias(
                name=MODEL_NAME,
                alias="development",
                version=registered_model.version
            )
            print(f"[INFO] Registered model as version {registered_model.version} with alias 'development'")
        else:
            print("[INFO] Model accuracy did not meet the threshold for registration.")

if __name__ == "__main__":
    main()
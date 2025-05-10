import os
import time
import joblib
import mlflow
import ray
from ray.train.sklearn import SklearnTrainer
from ray.train import ScalingConfig, Checkpoint
from mlflow.tracking import MlflowClient

MODEL_PATH = "crash_model.joblib"
MODEL_NAME = "CrashModel"

def train_loop(config):
    time.sleep(5)

    model = joblib.load(MODEL_PATH)

    print(model.__getstate__())

    # Dummy metrics
    accuracy = 0.85
    loss = 0.35

    # Save the model to checkpoint
    checkpoint = Checkpoint.from_dict({"model": model})
    ray.train.report({"accuracy": accuracy, "loss": loss}, checkpoint=checkpoint)

def main():
    ray.init()

    trainer = SklearnTrainer(
        train_loop_per_worker=train_loop,
        scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
    )

    result = trainer.fit()

    model = result.checkpoint.to_dict()["model"]
    metrics = result.metrics
    accuracy = metrics.get("accuracy", 0.0)
    loss = metrics.get("loss", 1.0)

    with mlflow.start_run() as run:
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("loss", loss)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"[INFO] Logged model to MLflow run: {run.info.run_id}")

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

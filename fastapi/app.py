from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import numpy as np

app = FastAPI(
    title="Crash Severity Prediction API",
    description="API for predicting crash severity based on input features",
    version="1.0.0"
)

class CrashRequest(BaseModel):
    features: list[float] = Field(
        ..., 
        description="List of feature values in the order used during model training"
    )

class PredictionResponse(BaseModel):
    severity: str = Field(
        ...,
        description="Predicted severity class"
    )
    probability: float = Field(
        ...,
        ge=0, 
        le=1,
        description="Prediction probability for the predicted class"
    )

# Load the crash severity model
MODEL_PATH = "crash_model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# If the model has class labels, retrieve them
class_labels = getattr(model, "classes_", None)

@app.post("/predict", response_model=PredictionResponse)
def predict_crash(request: CrashRequest):
    # Convert the list of features into the required shape
    data = np.array(request.features).reshape(1, -1)

    # Perform prediction
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(data)
        pred_idx = int(np.argmax(proba, axis=1)[0])
        pred_label = class_labels[pred_idx] if class_labels is not None else pred_idx
        confidence = float(proba[0, pred_idx])
    else:
        # If predict_proba is not available, fallback to predict
        pred = model.predict(data)[0]
        pred_label = pred
        confidence = None

    return PredictionResponse(
        severity=str(pred_label),
        probability=confidence if confidence is not None else 0.0
    )

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Initialize FastAPI app
app = FastAPI(
    title="Intersection Crash Forecast API",
    description="Predicts number of accidents at an intersection in the next 6 months",
    version="1.0.0"
)

# Load the trained model
model = joblib.load("crash_model.joblib")  # Update path if needed

# Define input data structure
class IntersectionFeatures(BaseModel):
    intersection_id: int
    accidents_6m: int
    accidents_1y: int
    accidents_5y: int

# Health check
@app.get("/test")
async def root():
    return {"message": "FastAPI is running"}

# Prediction endpoint
@app.post("/predict")
def predict(data: IntersectionFeatures):
    input_data = np.array([[
        data.intersection_id,
        data.accidents_6m,
        data.accidents_1y,
        data.accidents_5y
    ]])

    prediction = model.predict(input_data)
    return {"predicted_future_accidents_6m": float(prediction[0])}

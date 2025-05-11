from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Initialize FastAPI app
app = FastAPI(
    title="Crash Severity Prediction API",
    description="API for predicting crash severity based on input features",
    version="1.0.0"
)

# Load the trained model
model = joblib.load("crash_model.joblib")  # Make sure the filename matches

# Define the input data structure
class AccidentDetails(BaseModel):
    weather_condition: int
    road_surface: int
    light_condition: int
    vehicle_count: int
    pedestrian_involved: bool
    speed_limit: float

# Health check/test endpoint
@app.get("/test")
async def root():
    return {"message": "Hello World"}

# Prediction endpoint
@app.post("/predict")
def predict(details: AccidentDetails):
    input_data = np.array([[
        details.weather_condition,
        details.road_surface,
        details.light_condition,
        details.vehicle_count,
        int(details.pedestrian_involved),  # Convert bool to int
        details.speed_limit
    ]])

    prediction = model.predict(input_data)
    return {"predicted_severity": int(prediction[0])}

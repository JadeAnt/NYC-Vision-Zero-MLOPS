from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import hashlib
import os

# Initialize FastAPI app
app = FastAPI(
    title="Intersection Crash Forecast API",
    description="Predicts number of accidents at an intersection in the next 6 months",
    version="1.0.0"
)

MOUNT_DIR = os.getenv("MOUNT_DIR", "/mnt/chi_data")

# Load the trained model
model = joblib.load("crash_model.joblib")  # Update path if needed

# Define input data structure
class IntersectionFeatures(BaseModel):
    intersection_id: str
    accidents_6m: int
    accidents_1y: int
    accidents_5y: int

@app.get("/test")
async def root():
    return {"message": "FastAPI is running"}

# Prediction endpoint
@app.post("/predict")
def predict(data: IntersectionFeatures):
    input_data = np.array([[
        int(hashlib.sha256(data.intersection_id.encode('utf-8')).hexdigest(), 16),
        data.accidents_6m,
        data.accidents_1y,
        data.accidents_5y
    ]])
    
    prediction = model.predict(input_data)
    label = ""
    if(float(prediction[0]) <= 1):
        label = "Safe"
    elif(float(prediction[0]) <= 3):
        label = "Caution"
    elif(float(prediction[0]) > 3):
        label = "Dangerous"

    prediction_result = {
        "predicted_future_accidents_6m": float(prediction[0]),
        "label_classification" : label
    }

    dir_listing = {}
    if not os.path.exists(MOUNT_DIR):
        dir_listing = {"error": f"Mount directory {MOUNT_DIR} does not exist."}
    else:
        try:
            entries = os.listdir(MOUNT_DIR)
            directories = [entry for entry in entries if os.path.isdir(os.path.join(MOUNT_DIR, entry))]
            dir_listing = {
                "mounted_at": MOUNT_DIR,
                "directories": directories,
                "all_entries_in_mount": entries
            }
        except Exception as e:
            dir_listing = {"error": f"Failed to list directories in {MOUNT_DIR}: {str(e)}"}

    combined_response = {**prediction_result, "directory_info": dir_listing}
    
    return combined_response

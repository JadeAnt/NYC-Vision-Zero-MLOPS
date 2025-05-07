from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import numpy as np
import pandas as pd
import joblib

app = FastAPI(
    title="Crash Severity Prediction API",
    description="API for predicting crash severity based on input features",
    version="1.0.0"
)

@app.get("/test")
async def root():
    return {"message": "Hello World"}

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
import requests

# URL to download the model file from Google Drive
MODEL_URL = "https://drive.google.com/uc?export=download&id=1icVcey1pfpf0wc2XT3EG5Z90LUdpt0iG"
MODEL_PATH = "model.pkl"

# Download the model file if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("Model download complete.")

# Load the trained model
model = joblib.load(MODEL_PATH)

# Initialize FastAPI app
app = FastAPI()

# Define the expected input format using Pydantic
class Features(BaseModel):
    data: list

# Root endpoint for testing
@app.get("/")
def root():
    return {"message": "House Price Predictor API"}

# Prediction endpoint
@app.post("/predict")
def predict(features: Features):
    X = np.array([features.data])
    prediction = model.predict(X)
    return {"predicted_price": float(prediction[0])}

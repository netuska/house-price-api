### File: main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

app = FastAPI()

class Features(BaseModel):
    data: list

@app.get("/")
def root():
    return {"message": "House Price Predictor API"}

@app.post("/predict")
def predict(features: Features):
    X = np.array([features.data])
    prediction = model.predict(X)
    return {"predicted_price": float(prediction[0])}

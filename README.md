# House Price Prediction API

This is a simple FastAPI application that predicts house prices using the California Housing dataset.

## Run Locally

```bash
pip install -r requirements.txt
python train_model.py
uvicorn main:app --reload
```

## Deploy to Render
1. Create a new Web Service
2. Connect your GitHub repo
3. Set the start command:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
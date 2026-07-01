from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
import os

app = FastAPI(title="Churn Prediction API")
MODEL_NAME = os.getenv("MODEL_NAME", "churn-logistic")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
model = None

@app.on_event("startup")
def load_model():
    global model
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(customer: dict):
    df = pd.DataFrame([customer])
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])
    return {"churn_prediction": pred, "churn_probability": round(prob, 4)}

# src/serve.py
"""
Churn Prediction Serving API — Kubernetes ready.
Proper readiness and liveness endpoints for K8s probes.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn probability",
    version="2.0.0"
)

MODEL_PATH  = os.getenv("MODEL_PATH", "models/model.pkl")
MODEL_NAME  = os.getenv("MODEL_NAME", "churn-logistic")
model       = None
model_ready = False


@app.on_event("startup")
def load_model():
    global model, model_ready
    logger.info(f"Loading model from: {MODEL_PATH}")
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        model_ready = True
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model_ready = False


# ── Health endpoints ─────────────────────────────────────
@app.get("/health")
def health():
    """Liveness probe — is the server alive?"""
    return {"status": "alive"}


@app.get("/ready")
def ready():
    """
    Readiness probe — is the model loaded and ready for traffic?
    Returns 503 if model not loaded — K8s won't send traffic yet.
    """
    if not model_ready:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet"
        )
    return {"status": "ready", "model": MODEL_NAME}


@app.get("/")
def root():
    return {
        "service": "Churn Prediction API",
        "status":  "running",
        "model":   MODEL_NAME,
        "docs":    "/docs"
    }


# ── Request / Response schemas ───────────────────────────
class CustomerFeatures(BaseModel):
    tenure_months:             float
    monthly_charges:           float
    total_charges:             float
    support_calls:             float
    contract_month_to_month:   float
    has_tech_support:          float
    senior_citizen:            float
    paperless_billing:         float

    class Config:
        json_schema_extra = {
            "example": {
                "tenure_months": 5,
                "monthly_charges": 95.0,
                "total_charges": 475.0,
                "support_calls": 6,
                "contract_month_to_month": 1,
                "has_tech_support": 0,
                "senior_citizen": 0,
                "paperless_billing": 1
            }
        }


class ChurnResponse(BaseModel):
    churn_prediction:  int
    churn_label:       str
    churn_probability: float
    stay_probability:  float
    model:             str
    risk_level:        str


# ── Prediction endpoint ──────────────────────────────────
@app.post("/predict", response_model=ChurnResponse)
def predict(customer: CustomerFeatures):
    if not model_ready:
        raise HTTPException(503, "Model not ready")

    df   = pd.DataFrame([customer.dict()])
    pred = int(model.predict(df)[0])
    prob = model.predict_proba(df)[0]

    churn_prob = round(float(prob[1]), 4)
    stay_prob  = round(float(prob[0]), 4)

    # business-friendly risk level
    if churn_prob >= 0.7:
        risk = "HIGH"
    elif churn_prob >= 0.4:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return ChurnResponse(
        churn_prediction  = pred,
        churn_label       = "Will Churn" if pred == 1 else "Will Stay",
        churn_probability = churn_prob,
        stay_probability  = stay_prob,
        model             = MODEL_NAME,
        risk_level        = risk
    )


@app.get("/model-info")
def model_info():
    return {
        "model_name": MODEL_NAME,
        "model_path": MODEL_PATH,
        "ready":      model_ready
    }
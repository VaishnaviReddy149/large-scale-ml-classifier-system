import pickle
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="ML-powered fraud detection on 284K transactions using Random Forest",
    version="1.0.0"
)

# ── Load Model ─────────────────────────────────────────────────────────────────
MODEL = None

def load_model():
    global MODEL
    model_path = "models/random_forest.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            MODEL = pickle.load(f)
        print("✅ Model loaded from models/random_forest.pkl")
    else:
        raise FileNotFoundError("Model not found. Run train.py first and save the model.")

# ── Schemas ────────────────────────────────────────────────────────────────────
class Transaction(BaseModel):
    features: List[float]  # 30 scaled features

class PredictionResponse(BaseModel):
    transaction_id: int
    prediction: str
    fraud_probability: float
    is_fraud: bool

class BatchRequest(BaseModel):
    transactions: List[Transaction]

# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
def root():
    return {
        "message": "Credit Card Fraud Detection API",
        "status": "running",
        "endpoints": ["/predict", "/predict/batch", "/health", "/docs"]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "model_type": "RandomForestClassifier"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(transaction.features) != 30:
        raise HTTPException(status_code=400, detail=f"Expected 30 features, got {len(transaction.features)}")

    X = np.array(transaction.features).reshape(1, -1)
    prediction = MODEL.predict(X)[0]
    probability = MODEL.predict_proba(X)[0][1]

    return PredictionResponse(
        transaction_id=1,
        prediction="FRAUD" if prediction == 1 else "LEGITIMATE",
        fraud_probability=round(float(probability), 4),
        is_fraud=bool(prediction)
    )

@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    X = np.array([t.features for t in request.transactions])

    predictions = MODEL.predict(X)
    probabilities = MODEL.predict_proba(X)[:, 1]

    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        results.append({
            "transaction_id": i + 1,
            "prediction": "FRAUD" if pred == 1 else "LEGITIMATE",
            "fraud_probability": round(float(prob), 4),
            "is_fraud": bool(pred)
        })

    fraud_count = sum(1 for r in results if r["is_fraud"])
    return {
        "total_transactions": len(results),
        "fraud_detected": fraud_count,
        "legitimate": len(results) - fraud_count,
        "results": results
    }
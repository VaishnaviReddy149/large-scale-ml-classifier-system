import pytest
from fastapi.testclient import TestClient
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the model so tests don't need the actual .pkl file
import unittest.mock as mock
import numpy as np

with mock.patch("src.api.load_model"):
    from src.api import app, MODEL
    import src.api as api_module

    # Create a mock model
    mock_model = mock.MagicMock()
    mock_model.predict.side_effect = lambda X: np.zeros(len(X), dtype=int)
    mock_model.predict_proba.side_effect = lambda X: np.array([[0.98, 0.02]] * len(X))
    api_module.MODEL = mock_model

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_valid():
    payload = {"features": [0.1] * 30}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "fraud_probability" in data
    assert data["prediction"] in ["FRAUD", "LEGITIMATE"]

def test_predict_wrong_features():
    payload = {"features": [0.1] * 10}  # wrong length
    response = client.post("/predict", json=payload)
    assert response.status_code == 400

def test_predict_batch():
    payload = {"transactions": [{"features": [0.1] * 30}, {"features": [-0.5] * 30}]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_transactions"] == 2
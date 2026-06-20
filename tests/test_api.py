import pytest
import numpy as np
import unittest.mock as mock
from fastapi.testclient import TestClient

# Mock the model loading before importing app
with mock.patch("builtins.open", mock.mock_open(read_data=b"")):
    with mock.patch("pickle.load") as mock_load:
        mock_model = mock.MagicMock()
        mock_model.predict.side_effect = lambda X: np.zeros(len(X), dtype=int)
        mock_model.predict_proba.side_effect = lambda X: np.array([[0.98, 0.02]] * len(X))
        mock_load.return_value = mock_model

        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

        with mock.patch("os.path.exists", return_value=True):
            from src.api import app
            import src.api as api_module
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
    assert data["prediction"] in ["FRAUD", "LEGITIMATE"]

def test_predict_wrong_features():
    payload = {"features": [0.1] * 10}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400

def test_predict_batch():
    payload = {"transactions": [{"features": [0.1] * 30}, {"features": [-0.5] * 30}]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_transactions"] == 2
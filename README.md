# Large-Scale ML Classifier System

A production-grade machine learning pipeline for real-time credit card fraud detection, built with PySpark, PyTorch, scikit-learn, MLflow, FastAPI, and Docker.

## 🏗️ Architecture
Raw Data (284,807 transactions)

↓

PySpark Preprocessing (Feature Scaling)

↓

Model Training + MLflow Tracking

├── Logistic Regression (baseline)

├── Random Forest (best: 97.9% ROC-AUC)

└── PyTorch Neural Network

↓

FastAPI REST Endpoint (/predict, /predict/batch)

↓

Docker Container → AWS EC2

## 📊 Results

| Model | Accuracy | F1 Score | ROC-AUC | Recall |
|---|---|---|---|---|
| Logistic Regression | 97.4% | 0.119 | 0.977 | 0.943 |
| **Random Forest** | **99.9%** | **0.831** | **0.979** | **0.867** |
| Neural Network | 99.4% | 0.377 | 0.978 | 0.914 |

**Dataset:** 284,807 transactions, 492 fraud cases (0.17% — highly imbalanced)

## 🛠️ Tech Stack

- **Data Processing:** PySpark 4.1, pandas
- **ML Models:** scikit-learn, PyTorch
- **Experiment Tracking:** MLflow
- **API:** FastAPI + Uvicorn
- **Containerization:** Docker
- **CI/CD:** GitHub Actions
- **Cloud:** AWS EC2
- **Testing:** pytest (5/5 tests passing)

## 🚀 Quick Start

### Local Setup

```bash
git clone https://github.com/VaishnaviReddy149/large-scale-ml-classifier-system.git
cd large-scale-ml-classifier-system

python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Run Preprocessing
```bash
python src/preprocessing.py
```

### Train Models
```bash
python src/train.py
```

### Start API
```bash
python src/save_model.py
uvicorn src.api:app --reload --port 8000
```

### Run Tests
```bash
pytest tests/ -v
```

## 🐳 Docker

```bash
docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/predict` | POST | Single transaction prediction |
| `/predict/batch` | POST | Batch predictions |
| `/docs` | GET | Interactive Swagger UI |

### Example Request

```bash
POST /predict
{
  "features": [0.1, -1.2, 0.3, ...]  # 30 scaled features
}
```

### Example Response

```json
{
  "transaction_id": 1,
  "prediction": "LEGITIMATE",
  "fraud_probability": 0.0158,
  "is_fraud": false
}
```

## 📁 Project Structure
large-scale-ml-classifier-system/

├── .github/workflows/ci.yml    # GitHub Actions CI/CD

├── src/

│   ├── preprocessing.py        # PySpark data pipeline

│   ├── train.py                # Model training + MLflow

│   ├── api.py                  # FastAPI endpoints

│   └── save_model.py           # Model serialization

├── tests/

│   └── test_api.py             # pytest test suite

├── Dockerfile

├── requirements.txt

└── README.md

## ⚙️ CI/CD Pipeline

GitHub Actions automatically:
1. Runs pytest on every push
2. Builds Docker image on merge to main
3. Pushes to Docker Hub
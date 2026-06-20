import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, roc_auc_score)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore")

# ── Load Data ──────────────────────────────────────────────────────────────────
def load_data():
    print("Loading preprocessed data...")
    train_df = pd.read_csv("data/processed/train.csv")
    test_df  = pd.read_csv("data/processed/test.csv")

    X_train = train_df.drop("label", axis=1).values
    y_train = train_df["label"].values
    X_test  = test_df.drop("label", axis=1).values
    y_test  = test_df["label"].values

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Fraud in train: {y_train.sum()}, Fraud in test: {y_test.sum()}")
    return X_train, X_test, y_train, y_test

# ── Metrics Helper ─────────────────────────────────────────────────────────────
def get_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_prob),
    }

def print_metrics(name, metrics):
    print(f"\n{'='*40}")
    print(f"  {name} Results")
    print(f"{'='*40}")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")

# ── Model 1: Logistic Regression ───────────────────────────────────────────────
def train_logistic_regression(X_train, X_test, y_train, y_test):
    print("\n[1/3] Training Logistic Regression...")

    with mlflow.start_run(run_name="LogisticRegression"):
        params = {"C": 0.1, "max_iter": 1000, "class_weight": "balanced", "random_state": 42}
        mlflow.log_params(params)

        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = get_metrics(y_test, y_pred, y_prob)

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "logistic_regression_model")

        print_metrics("Logistic Regression", metrics)
        return metrics

# ── Model 2: Random Forest ─────────────────────────────────────────────────────
def train_random_forest(X_train, X_test, y_train, y_test):
    print("\n[2/3] Training Random Forest...")

    with mlflow.start_run(run_name="RandomForest"):
        params = {"n_estimators": 100, "max_depth": 10, 
                  "class_weight": "balanced", "random_state": 42, "n_jobs": -1}
        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = get_metrics(y_test, y_pred, y_prob)

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "random_forest_model")

        print_metrics("Random Forest", metrics)
        return metrics

# ── Model 3: PyTorch Neural Network ───────────────────────────────────────────
class FraudNet(nn.Module):
    def __init__(self, input_dim):
        super(FraudNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

def train_neural_network(X_train, X_test, y_train, y_test):
    print("\n[3/3] Training Neural Network...")

    with mlflow.start_run(run_name="NeuralNetwork"):
        # Hyperparams
        params = {"epochs": 20, "batch_size": 512, "lr": 0.001, "hidden1": 64, "hidden2": 32}
        mlflow.log_params(params)

        # Tensors
        X_tr = torch.FloatTensor(X_train)
        y_tr = torch.FloatTensor(y_train)
        X_te = torch.FloatTensor(X_test)
        y_te = torch.FloatTensor(y_test)

        # Handle class imbalance
        pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        dataset = TensorDataset(X_tr, y_tr)
        loader  = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)

        model     = FraudNet(X_train.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

        # Training loop
        model.train()
        for epoch in range(params["epochs"]):
            total_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                out  = model(X_batch).squeeze()
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{params['epochs']} - Loss: {total_loss/len(loader):.4f}")
                mlflow.log_metric("train_loss", total_loss / len(loader), step=epoch)

        # Evaluation
        model.eval()
        with torch.no_grad():
            y_prob = model(X_te).squeeze().numpy()
            y_pred = (y_prob >= 0.5).astype(int)

        metrics = get_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)
        input_example = X_te[:1]  # one sample as example input
        mlflow.pytorch.log_model(model, "neural_network_model", input_example=input_example, serialization_format="pickle")
        print_metrics("Neural Network", metrics)
        return metrics

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mlflow.set_experiment("CreditCard_Fraud_Detection")

    X_train, X_test, y_train, y_test = load_data()

    lr_metrics  = train_logistic_regression(X_train, X_test, y_train, y_test)
    rf_metrics  = train_random_forest(X_train, X_test, y_train, y_test)
    nn_metrics  = train_neural_network(X_train, X_test, y_train, y_test)

    # Summary
    print("\n" + "="*40)
    print("  FINAL COMPARISON")
    print("="*40)
    print(f"  {'Model':<22} {'F1':>6} {'ROC-AUC':>8} {'Recall':>8}")
    print(f"  {'-'*44}")
    print(f"  {'Logistic Regression':<22} {lr_metrics['f1']:>6.4f} {lr_metrics['roc_auc']:>8.4f} {lr_metrics['recall']:>8.4f}")
    print(f"  {'Random Forest':<22} {rf_metrics['f1']:>6.4f} {rf_metrics['roc_auc']:>8.4f} {rf_metrics['recall']:>8.4f}")
    print(f"  {'Neural Network':<22} {nn_metrics['f1']:>6.4f} {nn_metrics['roc_auc']:>8.4f} {nn_metrics['recall']:>8.4f}")
    print("\n✅ All models trained and logged to MLflow!")
    print("   Run: mlflow ui   to view results in browser")
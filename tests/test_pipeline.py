import os
import pandas as pd
from fastapi.testclient import TestClient
from src.train import train_pipeline
from src.app import app


def test_dataset_schema():
    df = pd.read_csv("data/credit_scoring.csv")
    expected_cols = {
        "age", "annual_income", "loan_amount", "credit_history_years",
        "debt_to_income_ratio", "num_open_accounts", "late_payments", "default_risk"
    }
    assert expected_cols.issubset(set(df.columns)), "Colonnes manquantes dans le dataset"
    assert df.isnull().sum().sum() == 0, "Le dataset contient des valeurs nulles"


def test_training_and_metrics():
    metrics = train_pipeline()
    assert os.path.exists("models/model_pipeline.joblib"), "Le fichier modèle n'a pas été généré"
    assert metrics["roc_auc"] >= 0.70, f"ROC-AUC insuffisant : {metrics['roc_auc']}"


def test_api_endpoints():
    client = TestClient(app)
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["model_loaded"] is True

    payload = {
        "age": 42,
        "annual_income": 65000.0,
        "loan_amount": 10000.0,
        "credit_history_years": 12,
        "debt_to_income_ratio": 0.18,
        "num_open_accounts": 5,
        "late_payments": 0
    }
    pred_resp = client.post("/predict", json=payload)
    assert pred_resp.status_code == 200
    body = pred_resp.json()
    assert "default_prediction" in body
    assert 0.0 <= body["default_probability"] <= 1.0

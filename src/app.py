import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "models/model_pipeline.joblib")

app = FastAPI(
    title="Credit Risk Scoring API",
    description="API de prédiction du risque de défaut de crédit — TP MLOps",
    version="1.0.0"
)


class CreditApplication(BaseModel):
    age: int = Field(..., ge=18, le=100, example=35)
    annual_income: float = Field(..., gt=0, example=45000.0)
    loan_amount: float = Field(..., gt=0, example=12000.0)
    credit_history_years: int = Field(..., ge=0, le=60, example=8)
    debt_to_income_ratio: float = Field(..., ge=0.0, le=5.0, example=0.28)
    num_open_accounts: int = Field(..., ge=0, le=50, example=4)
    late_payments: int = Field(..., ge=0, le=50, example=0)


class PredictionResponse(BaseModel):
    default_prediction: int
    default_probability: float
    risk_level: str


def get_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modèle introuvable à l'emplacement : {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health_check():
    model_ready = os.path.exists(MODEL_PATH)
    return {"status": "healthy" if model_ready else "degraded", "model_loaded": model_ready}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: CreditApplication):
    try:
        model = get_model()
        input_df = pd.DataFrame([application.model_dump()])
        pred = int(model.predict(input_df)[0])
        proba = float(model.predict_proba(input_df)[0][1])

        if proba < 0.30:
            risk = "FAIBLE"
        elif proba < 0.60:
            risk = "MODÉRÉ"
        else:
            risk = "ÉLEVÉ"

        return PredictionResponse(
            default_prediction=pred,
            default_probability=round(proba, 4),
            risk_level=risk
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'inférence : {str(e)}")

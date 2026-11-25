import os
import yaml
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score


def load_params(config_path: str = "params.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_pipeline():
    params = load_params()
    data_cfg = params["data"]
    model_cfg = params["model"]
    mlflow_cfg = params["mlflow"]

    # 1. Chargement des données
    df = pd.read_csv(data_cfg["raw_path"])
    X = df.drop(columns=["default_risk"])
    y = df["default_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=data_cfg["test_size"], random_state=data_cfg["random_state"], stratify=y
    )

    # 2. Configuration MLflow
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run(run_name="RF_Baseline_Pipeline") as run:
        # Log des hyperparamètres
        mlflow.log_params({
            "model_type": model_cfg["type"],
            "n_estimators": model_cfg["n_estimators"],
            "max_depth": model_cfg["max_depth"],
            "min_samples_split": model_cfg["min_samples_split"],
            "class_weight": model_cfg["class_weight"],
            "test_size": data_cfg["test_size"]
        })

        # 3. Construction et entraînement du pipeline
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=model_cfg["n_estimators"],
                max_depth=model_cfg["max_depth"],
                min_samples_split=model_cfg["min_samples_split"],
                class_weight=model_cfg["class_weight"],
                random_state=data_cfg["random_state"]
            ))
        ])
        pipeline.fit(X_train, y_train)

        # 4. Évaluation
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred)), 4),
            "recall": round(float(recall_score(y_test, y_pred)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        }
        mlflow.log_metrics(metrics)

        # 5. Sauvegarde locale et dans le Model Registry MLflow
        os.makedirs("models", exist_ok=True)
        local_model_path = "models/model_pipeline.joblib"
        joblib.dump(pipeline, local_model_path)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=mlflow_cfg["registered_model_name"]
        )

        print(f"Run ID : {run.info.run_id}")
        print(f"Métriques enregistrées : {metrics}")
        return metrics


if __name__ == "__main__":
    train_pipeline()

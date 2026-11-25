# Ressources du TP MLOps — Cycle de vie complet d'un modèle de Credit Scoring

## Arborescence du projet
- `data/credit_scoring.csv` : Jeu de données d'entraînement (1 500 clients).
- `data/production_requests.csv` : Échantillon de requêtes en production présentant une dérive (Data Drift).
- `params.yaml` : Fichier centralisé des hyperparamètres et configurations.
- `src/train.py` : Pipeline d'entraînement reproductible avec tracking MLflow et Model Registry.
- `src/app.py` : API REST FastAPI avec validation Pydantic pour l'inférence temps réel.
- `src/monitor.py` : Script d'audit de Data Drift avec Evidently AI.
- `tests/test_pipeline.py` : Suite de tests automatisés (schéma, qualité modèle, API).
- `Dockerfile` : Conteneurisation de l'API de prédiction.
- `.github/workflows/mlops_ci.yml` : Pipeline d'intégration continue GitHub Actions.

## Démarrage rapide
```bash
python -m venv venv
source venv/bin/activate  # Sous Windows : venv\Scripts\activate
pip install -r requirements.txt
python -m src.train
mlflow ui --port 5000
```

ml-serving-platform

Production ML inference service with FastAPI, Docker, MLflow model registry, and automated data drift monitoring — built for real-time scoring at scale.

Show Image Show Image Show Image Show Image Show Image

Overview

This project implements a production-grade ML inference platform that goes beyond model training — it covers the full serving lifecycle:

REST inference API with /predict, /health, and /metrics endpoints
MLflow model registry for versioned model storage, staging, and promotion
Automated data drift monitoring using Population Stability Index (PSI) — compares incoming request distributions against the training baseline and triggers alerts when drift exceeds threshold
Automated retraining trigger — when drift is detected, a retraining pipeline is kicked off and the new model is registered back into MLflow
Request logging — every prediction request and response is logged to SQLite for audit and monitoring
Containerized deployment — full Docker Compose setup, deployable to AWS EC2
Architecture
┌─────────────────────────────────────────────────────────────┐
│                        Client Request                        │
└───────────────────────────┬─────────────────────────────────┘
                            │  POST /predict
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Inference API                    │
│   /predict   /health   /metrics   /drift-report             │
└──────┬──────────────────────────────────────┬───────────────┘
       │  load model                          │  log request
       ▼                                      ▼
┌──────────────────┐                ┌─────────────────────────┐
│  MLflow Registry │                │   Request Logger (SQLite)│
│  (model versions)│                │   - features            │
│  staging → prod  │                │   - predictions         │
└──────────────────┘                │   - timestamps          │
                                    └────────────┬────────────┘
                                                 │  scheduled check
                                                 ▼
                                    ┌─────────────────────────┐
                                    │   Drift Monitor          │
                                    │   PSI per feature        │
                                    │   threshold: PSI > 0.2   │
                                    └────────────┬────────────┘
                                                 │  drift detected
                                                 ▼
                                    ┌─────────────────────────┐
                                    │   Retraining Trigger     │
                                    │   retrain → evaluate     │
                                    │   → register in MLflow   │
                                    └─────────────────────────┘
Project Structure
ml-serving-platform/
├── model/
│   ├── train.py              # Training script with MLflow logging
│   ├── evaluate.py           # Model evaluation utilities
│   └── feature_config.py     # Feature definitions and baseline stats
│
├── api/
│   ├── main.py               # FastAPI app — /predict, /health, /metrics
│   ├── schemas.py            # Pydantic request/response models
│   ├── model_loader.py       # MLflow model loading + caching
│   └── request_logger.py     # SQLite request/response logger
│
├── monitoring/
│   ├── drift_detector.py     # PSI-based feature drift detection
│   ├── retraining_trigger.py # Automated retraining on drift alert
│   └── monitor_job.py        # Scheduled monitoring job (APScheduler)
│
├── tests/
│   ├── test_api.py           # FastAPI endpoint tests
│   └── test_drift.py         # Drift detection unit tests
│
├── docker-compose.yml        # Full stack: API + MLflow server
├── Dockerfile                # API container
├── requirements.txt
└── README.md
Quickstart
1. Clone and install
bash
git clone https://github.com/ManyaEleti/ml-serving-platform.git
cd ml-serving-platform
pip install -r requirements.txt
2. Train and register the model
bash
python model/train.py
# Logs metrics, params, and registers model to MLflow
# MLflow UI: http://localhost:5000
3. Run with Docker Compose
bash
docker-compose up --build
# API:    http://localhost:8000
# MLflow: http://localhost:5000
4. Make a prediction
bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, 1.2, -0.3, 0.8, 1.1, -0.5, 0.2]}'
5. Check drift report
bash
curl http://localhost:8000/drift-report
Key Features
/predict endpoint
Loads the latest production model from MLflow registry
Returns prediction + confidence score
Logs every request (features + prediction) to SQLite
Drift Monitoring (PSI)
Runs on a configurable schedule (default: every 100 requests)
Computes Population Stability Index per feature against training baseline
PSI < 0.1: no action | PSI 0.1–0.2: warning | PSI > 0.2: trigger retraining
Drift report available at /drift-report
Automated Retraining
Triggered automatically when any feature PSI exceeds threshold
Retrains on latest logged request data + original training data
Evaluates new model vs. current production model on held-out set
Registers and promotes to production in MLflow only if performance improves
MLflow Integration
Full experiment tracking: params, metrics, artifacts per run
Model versioning: registered models with staging → production promotion
Model loading with caching — no cold-start latency on repeated calls
Tech Stack
Component	Technology
Inference API	FastAPI
Model Registry	MLflow
Containerization	Docker, Docker Compose
Drift Detection	PSI (custom Python)
Scheduling	APScheduler
Request Logging	SQLite
ML Models	Scikit-learn, XGBoost
Deployment	AWS EC2
Testing	pytest
Monitoring in Action
[2026-11-11 01:00:00] Drift check triggered after 100 requests
Feature 'tenure':       PSI = 0.07  ✓ stable
Feature 'monthly_charges': PSI = 0.14  ⚠ warning
Feature 'num_products': PSI = 0.23  ✗ DRIFT DETECTED
→ Retraining triggered...
→ New model ROC-AUC: 0.891 vs current: 0.880
→ Promoting new model to production in MLflow
[2026-11-11 01:03:42] Retraining complete. Model v3 promoted to production.
Author

Lakshmi Manya Eleti github.com/ManyaEleti | linkedin.com/in/lakshmi-manya-eleti-116142241

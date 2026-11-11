ml-serving-platform

A production ML inference platform built phase by phase. Trains a model, serves it through a typed REST API, logs every prediction, monitors feature drift using Population Stability Index, and automatically retrains and promotes a new model when drift exceeds threshold — all wired together with MLflow, FastAPI, Docker, and AWS EC2.

The project is built phase by phase, with each phase committed and documented before the next begins.

Status
Phase	Scope	State
0	Repository scaffold, tooling, folder structure	Done
1	Training pipeline with MLflow experiment tracking and model registry	Done
2	Baseline stats generation for drift monitoring	Done
3	FastAPI inference API — /predict, /health, /metrics, /drift-report	Done
4	SQLite request logger — features, predictions, latency per request	Done
5	PSI-based drift detector — per-feature Population Stability Index	Done
6	Automated retraining trigger — retrain, compare AUC, promote if better	Done
7	Docker Compose — API + MLflow server containerized	Done
8	AWS EC2 deployment	In Progress
9	APScheduler — scheduled drift checks independent of request count	Planned
10	Prometheus + Grafana metrics dashboard	Planned
11	Test suite — API endpoint tests, PSI unit tests, retraining integration tests	Planned
12	GitHub Actions CI — lint, test, build on every push	Planned
Stack
Area	Choice
Inference API	FastAPI, Uvicorn
Model Registry	MLflow
ML Models	XGBoost, Scikit-learn
Drift Detection	PSI (custom Python implementation)
Request Logging	SQLite
Scheduling	APScheduler
Containerization	Docker, Docker Compose
Deployment	AWS EC2
Validation	Pydantic v2
Testing	pytest, httpx
Language	Python 3.10+
What is implemented
Training pipeline (model/train.py)

Trains an XGBoost classifier with MLflow experiment tracking. Every run logs hyperparameters, metrics (ROC-AUC, F1, Precision, Recall), and artifacts. After training, the model is registered in the MLflow model registry and promoted through Staging → Production.

A baseline stats file (baseline_stats.json) is computed from the training distribution — bin edges and frequencies per feature — and stored as an MLflow artifact. The drift monitor loads this at startup to compare incoming request distributions against.

Inference API (api/main.py)

FastAPI app with four endpoints:

Endpoint	Method	Description
/predict	POST	Score a feature vector; returns prediction, probability, model version, latency
/health	GET	Model load status, uptime, model version
/metrics	GET	Request counts, avg latency, avg score, drift status
/drift-report	GET	Full PSI report across all features, drift flags, action taken

The model is loaded from the MLflow registry at startup and cached in memory. Every prediction is logged to SQLite with its feature vector, prediction, probability, latency, and model version.

Drift is checked every N requests (configurable via DRIFT_CHECK_EVERY_N, default 100). When drift is detected, retraining is triggered inline.

Request logging (api/main.py → SQLite)

Every call to /predict writes a row to request_log:

id | timestamp | features (JSON) | prediction | probability | latency_ms | model_version

The accumulated log feeds the drift detector and the retraining pipeline — no separate data store needed.

Drift monitoring (monitoring/drift_detector.py)

Population Stability Index measures how much each feature's distribution has shifted from training:

PSI = Σ (current% − baseline%) × ln(current% / baseline%)
PSI Range	Status	Action
< 0.1	✓ Stable	None
0.1 – 0.2	⚠ Warning	Log, monitor
> 0.2	✗ Drift detected	Trigger retraining

PSI is computed per feature against the training baseline bin edges. The /drift-report endpoint returns per-feature scores and which features crossed each threshold.

Automated retraining (monitoring/retraining_trigger.py)

When any feature's PSI exceeds 0.2, the retraining pipeline runs:

Evaluates the current production model's ROC-AUC on the most recent 500 logged requests
Retrains on the full dataset with MLflow tracking
Compares new model AUC against current production
Promotes to Production in MLflow only if the new model outperforms — otherwise keeps the existing model
Reloads the drift detector baseline
[2026-11-11 01:00:00] Drift check triggered after 100 requests
  tenure              PSI = 0.07   ✓ stable
  monthly_charges     PSI = 0.14   ⚠ warning
  num_products        PSI = 0.23   ✗ DRIFT DETECTED
→ Retraining triggered...
  Current production ROC-AUC: 0.880
  New model ROC-AUC:          0.891
→ Model v3 promoted to Production
Project structure
ml-serving-platform/
├── model/
│   ├── train.py               # Training + MLflow logging + baseline stats
│   ├── evaluate.py            # Evaluation utilities
│   └── feature_config.py      # Feature names and schema
│
├── api/
│   ├── main.py                # FastAPI app — /predict /health /metrics /drift-report
│   ├── schemas.py             # Pydantic request/response models
│   ├── model_loader.py        # MLflow model loading with caching
│   └── request_logger.py      # SQLite request/response logger
│
├── monitoring/
│   ├── drift_detector.py      # PSI computation per feature
│   ├── retraining_trigger.py  # Retrain → compare → promote pipeline
│   └── monitor_job.py         # APScheduler scheduled drift job
│
├── tests/
│   ├── test_api.py            # FastAPI endpoint tests
│   └── test_drift.py          # PSI unit tests
│
├── Dockerfile
├── docker-compose.yml         # API + MLflow server
├── requirements.txt
└── README.md
Quickstart
Prerequisites
Python 3.10+
Docker and Docker Compose
An MLflow tracking server (or run locally via Docker Compose)
1. Clone and install
bash
git clone https://github.com/ManyaEleti/ml-serving-platform.git
cd ml-serving-platform
pip install -r requirements.txt
2. Train and register the model
bash
python model/train.py
# Logs to MLflow, registers model, saves baseline_stats.json
# MLflow UI: http://localhost:5000
3. Run with Docker Compose
bash
docker-compose up --build
# API:    http://localhost:8000
# MLflow: http://localhost:5000
# Docs:   http://localhost:8000/docs
4. Make a prediction
bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.2, -0.5, 0.8, 1.1, -0.3, 0.6, -0.2]}'
json
{
  "prediction": 1,
  "probability": 0.7843,
  "model_version": "2",
  "request_id": "a3f2c1d0-...",
  "latency_ms": 4.21
}
5. Check drift
bash
curl http://localhost:8000/drift-report
6. Run tests
bash
pytest tests/ -v
Configuration

The API reads configuration from environment variables. All have sensible defaults for local development.

Variable	Default	Purpose
MLFLOW_TRACKING_URI	http://localhost:5000	MLflow server address
MODEL_NAME	churn-classifier	Registered model name in MLflow
MODEL_STAGE	Production	MLflow model stage to load
DB_PATH	request_log.db	SQLite request log path
DRIFT_CHECK_EVERY_N	100	Check drift every N predictions
PSI_WARNING_THRESHOLD	0.1	PSI threshold for warning
PSI_DRIFT_THRESHOLD	0.2	PSI threshold to trigger retraining
BASELINE_STATS_PATH	baseline_stats.json	Training baseline distribution
Design decisions

Why PSI for drift detection and not KS-test or MMD? PSI is interpretable, fast to compute, and produces a per-feature score that maps directly to an action threshold. KS-test and MMD are better for detecting subtle shifts, but PSI's binned approach is robust to small sample sizes and gives an immediately actionable signal. PSI > 0.2 is a widely used production threshold (insurance, finance).

Why SQLite for request logging and not Kafka or a database? This platform is designed to run on a single EC2 instance with zero external dependencies. SQLite handles thousands of writes per second for single-writer workloads, needs no server, and the full log is available as a local file. Replacing it with Postgres or a Kafka consumer is a one-file change to request_logger.py.

Why promote only if AUC improves? Automatic promotion without a performance gate would let a degraded model replace a good one when the retraining data is itself drifted or noisy. The comparison step uses the most recent 500 logged requests as a held-out evaluation set — the same data the new model was not trained on — so the comparison is fair.

Why MLflow for the model registry and not a file on disk? MLflow gives versioning, stage transitions, artifact storage, and a UI for free. The same training script works locally and in a CI pipeline without changing the promotion logic.

Known limitations
The retraining trigger is synchronous — it blocks the drift check thread while retraining. An async job queue (Celery or APScheduler) is planned for Phase 9.
The training data is synthetic. Replace model/train.py's load_data() with real data loading for a production use case.
SQLite is single-writer. Under high concurrency, switch request_logger.py to PostgreSQL.
The drift check runs every N requests, not on a wall-clock schedule. APScheduler integration is planned for Phase 9.
Author

Lakshmi Manya Eleti github.com/ManyaEleti · linkedin.com/in/lakshmi-manya-eleti-116142241

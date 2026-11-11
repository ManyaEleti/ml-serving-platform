from fastapi import FastAPI
app = FastAPI(title="ML Serving Platform")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: dict):
    # TODO: load model from MLflow and score
    return {"prediction": None, "status": "in_progress"}

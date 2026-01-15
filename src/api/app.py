import time
import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from src.config import settings
from src.utils.logger import get_logger
from src.api.schemas import Transaction, PredictionResponse
import uuid

# NEW: Import Prometheus Tools
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, REGISTRY

logger = get_logger(__name__)

artifacts = {}

# --- CUSTOM METRICS ---
metric_name_counter = "fraud_detection_predictions_total"
metric_name_histogram = "fraud_detection_probability_dist"

# Safe Registration (Idempotent)
if metric_name_counter in REGISTRY._names_to_collectors:
    FRAUD_COUNTER = REGISTRY._names_to_collectors[metric_name_counter]
else:
    FRAUD_COUNTER = Counter(
        metric_name_counter,
        "Total number of predictions by class",
        [
            "prediction_class"
        ],  # <--- RENAMED from 'class' to avoid Python keyword conflict
    )

if metric_name_histogram in REGISTRY._names_to_collectors:
    PREDICTION_PROBABILITY = REGISTRY._names_to_collectors[metric_name_histogram]
else:
    PREDICTION_PROBABILITY = Histogram(
        metric_name_histogram,
        "Distribution of fraud probabilities",
        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("⚡ Loading Model and Preprocessor...")
        artifacts["model"] = joblib.load(settings.MODEL_DIR / "challenger.pkl")
        artifacts["preprocessor"] = joblib.load(settings.MODEL_DIR / "preprocessor.pkl")
        logger.info("✅ Artifacts loaded successfully.")
        yield
        artifacts.clear()
        logger.info("🛑 Shutting down.")
    except Exception as e:
        logger.critical(f"❌ Failed to load artifacts: {e}")
        raise e


app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION, lifespan=lifespan)

# --- ACTIVATE MONITORING ---
instrumentator = Instrumentator().instrument(app).expose(app)


@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: Transaction):
    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())

    try:
        # 1. Prepare Data
        raw_df = pd.DataFrame(
            [{"Time": transaction.Time, "Amount": transaction.Amount}]
        )
        scaled_features = artifacts["preprocessor"].transform(raw_df)
        final_input = [
            scaled_features[0][0],
            scaled_features[0][1],
        ] + transaction.V_features
        model_features = artifacts["model"].feature_names_in_
        input_data = pd.DataFrame([final_input], columns=model_features)

        # 2. Predict
        probability = artifacts["model"].predict_proba(input_data)[0][1]
        is_fraud = probability > 0.5

        # --- UPDATE METRICS ---
        PREDICTION_PROBABILITY.observe(probability)

        label = "fraud" if is_fraud else "normal"
        # UPDATED USAGE: Use the new label name
        FRAUD_COUNTER.labels(prediction_class=label).inc()

        end_time = time.perf_counter()
        latency = (end_time - start_time) * 1000

        logger.info(
            f"🔮 Req: {request_id} | Prob: {probability:.4f} | Latency: {latency:.2f}ms"
        )

        return PredictionResponse(
            transaction_id=request_id,
            is_fraud=bool(is_fraud),
            probability=float(probability),
            latency_ms=round(latency, 2),
        )

    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal Prediction Error: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)

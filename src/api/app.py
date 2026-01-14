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

logger = get_logger(__name__)

# Global variables to hold artifacts
artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load artifacts on startup. Clean up on shutdown.
    """
    try:
        logger.info("⚡ Loading Model and Preprocessor...")

        # Load artifacts
        artifacts["model"] = joblib.load(settings.MODEL_DIR / "challenger.pkl")
        artifacts["preprocessor"] = joblib.load(settings.MODEL_DIR / "preprocessor.pkl")

        logger.info("✅ Artifacts loaded successfully.")
        yield

        # Cleanup
        artifacts.clear()
        logger.info("🛑 Shutting down.")

    except Exception as e:
        logger.critical(f"❌ Failed to load artifacts: {e}")
        raise e


app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION, lifespan=lifespan)


@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: Transaction):
    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())

    try:
        # 1. Convert Input to DataFrame (Expected format for Scaler)
        raw_df = pd.DataFrame(
            [{"Time": transaction.Time, "Amount": transaction.Amount}]
        )

        # 2. Apply Preprocessing (RobustScaler)
        scaled_features = artifacts["preprocessor"].transform(raw_df)

        # 3. Combine with V features
        # Order matters! Our scaler returns [Amount, Time] or [Time, Amount]
        # based on how features.py was written.
        # To be safe, we access by index: 0=Amount, 1=Time (based on features.py logic)

        # The model expects 30 features.
        # We need to construct the list in the exact order the model was trained on.
        # Assuming Training order was: [Scaled_Amount, Scaled_Time, V1, V2, ... V28]
        final_input = [
            scaled_features[0][0],
            scaled_features[0][1],
        ] + transaction.V_features

        # 4. Create DataFrame for Prediction
        # We use feature_names_in_ to ensure columns align perfectly
        model_features = artifacts["model"].feature_names_in_
        input_data = pd.DataFrame([final_input], columns=model_features)

        # 5. Predict
        probability = artifacts["model"].predict_proba(input_data)[0][1]
        is_fraud = probability > 0.5

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
        # We print the specific error to help debugging
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Internal Prediction Error: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)

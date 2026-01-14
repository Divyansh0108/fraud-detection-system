import pandas as pd
import xgboost as xgb
import numpy as np
import mlflow
import mlflow.xgboost
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.config import settings
from src.utils.logger import get_logger
import joblib

logger = get_logger(__name__)


def load_processed_data():
    """Loads processed data."""
    train_path = settings.PROCESSED_DATA_DIR / "train.csv"
    test_path = settings.PROCESSED_DATA_DIR / "test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(settings.TARGET_COLUMN, axis=1)
    y_train = train_df[settings.TARGET_COLUMN]

    X_test = test_df.drop(settings.TARGET_COLUMN, axis=1)
    y_test = test_df[settings.TARGET_COLUMN]

    return X_train, y_train, X_test, y_test


def train_challenger():
    mlflow.set_experiment(f"Fraud-Detection-{settings.ENVIRONMENT}")

    with mlflow.start_run(run_name="Challenger_XGBoost"):
        logger.info("🥊 Starting Challenger Training (XGBoost)...")

        X_train, y_train, X_test, y_test = load_processed_data()

        # Calculate scale_pos_weight for imbalance
        # Formula: (Count of Negatives) / (Count of Positives)
        # This tells XGBoost to pay attention to the minority class
        ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

        params = {
            "objective": "binary:logistic",
            "max_depth": 6,  # Depth of trees
            "learning_rate": 0.1,  # Step size
            "scale_pos_weight": ratio,  # Handle Imbalance
            "n_estimators": 100,
            "seed": settings.RANDOM_STATE,
        }

        logger.info(f"🔧 XGBoost Params: {params}")

        # Train
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "model")

        logger.info(f"✅ Challenger Results:")
        logger.info(f"   📊 Precision: {metrics['precision']:.4f} (Goal: > 0.50)")
        logger.info(f"   🔍 Recall:    {metrics['recall']:.4f} (Goal: > 0.80)")
        logger.info(f"   🏆 F1 Score:  {metrics['f1']:.4f}")

        # Save Artifact
        settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = settings.MODEL_DIR / "challenger.pkl"
        joblib.dump(model, model_path)
        logger.info(f"📦 Model saved to {model_path}")

if __name__ == "__main__":
    train_challenger()

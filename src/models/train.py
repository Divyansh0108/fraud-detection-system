import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.config import settings
from src.utils.logger import get_logger
import joblib

logger = get_logger(__name__)

def load_processed_data():
    """Loads processed data from the Warehouse."""
    train_path = settings.PROCESSED_DATA_DIR / "train.csv"
    test_path = settings.PROCESSED_DATA_DIR / "test.csv"
    
    logger.info(f"Loading data from {settings.PROCESSED_DATA_DIR}")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(settings.TARGET_COLUMN, axis=1)
    y_train = train_df[settings.TARGET_COLUMN]
    
    X_test = test_df.drop(settings.TARGET_COLUMN, axis=1)
    y_test = test_df[settings.TARGET_COLUMN]
    
    return X_train, y_train, X_test, y_test

def train_baseline():
    """
    Trains the Baseline Logistic Regression.
    """
    mlflow.set_experiment(f"Fraud-detection-{settings.ENVIRONMENT}")

    with mlflow.start_run(run_name="Baseline_logistic_balanced"):
        logger.info("Starting Baseline training...")

        X_train, y_train, X_test, y_test = load_processed_data()

        params = {
            "C": 1.0,
            "random_state": settings.RANDOM_STATE,
            "solver": "liblinear",
            "class_weight": "balanced",
        }

        logger.info(f"Model params: {params}")

        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        logger.info(f"✅ Training Complete. F1: {metrics['f1']:.4f}")
        logger.warning(f"📊 Precision: {metrics['precision']:.4f} (Expect Low)")
        logger.warning(f"🔍 Recall:    {metrics['recall']:.4f} (Expect High)")

if __name__ == "__main__":
    try:
        train_baseline()
    except Exception as e:
        logger.critical(f"❌ Training failed: {str(e)}")
        raise
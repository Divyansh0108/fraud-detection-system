import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Project Info
    PROJECT_NAME: str = "Fraud Detection System"
    VERSION: str = "1.0.0"

    # Business Constraints
    LATENCY_THRESHOLD_MS: int = 50

    # Paths (The Warehouse)
    DATA_DIR: str = os.path.join(os.getcwd(), "data")
    MODEL_DIR: str = os.path.join(os.getcwd(), "models")

    # Model Parameters (The Scientist)
    TARGET_COLUMN: str = "Class"
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42


settings = Settings()

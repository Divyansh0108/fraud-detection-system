from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    PROJECT_NAME: str = "Fraud Detection System"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"  # development, production

    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"

    RAW_DATA_PATH: Path = DATA_DIR / "raw" / "creditcard.csv"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    MODEL_DIR: Path = BASE_DIR / "models"

    LOG_LEVEL: str = "INFO"
    LOG_DIR: Path = BASE_DIR / "logs"
    JSON_LOGS: bool = False

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()

    MAX_LATENCY_MS: int = 50

    TARGET_COLUMN: str = "Class"
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42


settings = Settings()

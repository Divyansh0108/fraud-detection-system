import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from src.config import settings

# Ensure logs directory exists at path defined in config
settings.LOG_DIR.mkdir(exist_ok=True)


import colorlog
from pythonjsonlogger import jsonlogger


def configure_logger():
    """
    Configures the root logger with rotation and console output.
    """
    handlers = []

    # 1. Handler A: Console (With Colors)
    # Why: Much easier to read logs during development.
    console_handler = logging.StreamHandler(sys.stdout)
    color_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
    console_handler.setFormatter(color_formatter)
    handlers.append(console_handler)

    # 2. Handler B: File (With Rotation and optional JSON)
    # Why: Production systems (ELK/GCP) love JSON for easy parsing.
    file_handler = RotatingFileHandler(
        settings.LOG_DIR / "app.log", maxBytes=10 * 1024 * 1024, backupCount=5
    )

    if settings.JSON_LOGS:
        file_formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        )
    else:
        file_formatter = logging.Formatter(
            "%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
        )

    file_handler.setFormatter(file_formatter)
    handlers.append(file_handler)

    # 3. Set Basic Config
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        handlers=handlers,
        force=True,  # Overwrite previous configuration
    )


# Run configuration immediately
configure_logger()


def get_logger(name: str):
    """
    Returns a configured logger instance.
    Example: logger = get_logger(__name__)
    """
    return logging.getLogger(name)

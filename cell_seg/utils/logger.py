"""Logging + per-stage timing / try-except wrapper."""
import logging
import time
from contextlib import contextmanager
from typing import Iterator

from config import OUTPUTS_DIR

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_FILE = OUTPUTS_DIR / "app.log"


def get_logger(name: str) -> logging.Logger:
    """Logger writing to console and outputs/app.log; safe to call repeatedly."""
    if not name:
        raise ValueError("Logger name must be a non-empty string.")
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False  # avoid duplicate lines via the root logger
    formatter = logging.Formatter(LOG_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


@contextmanager
def stage_timer(logger: logging.Logger, stage: str) -> Iterator[None]:
    """Log stage duration on success; log traceback and re-raise on failure."""
    start = time.perf_counter()
    try:
        yield
    except Exception:
        logger.exception("%s failed after %.2f s", stage, time.perf_counter() - start)
        raise
    logger.info("%s done in %.2f s", stage, time.perf_counter() - start)

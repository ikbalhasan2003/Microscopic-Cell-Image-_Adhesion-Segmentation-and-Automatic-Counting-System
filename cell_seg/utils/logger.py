"""Logging + per-stage timing / try-except wrapper."""
import logging
from contextlib import contextmanager
from typing import Iterator


def get_logger(name: str) -> logging.Logger:
    raise NotImplementedError


@contextmanager
def stage_timer(logger: logging.Logger, stage: str) -> Iterator[None]:
    """Log stage duration on success; log traceback and re-raise on failure."""
    raise NotImplementedError
    yield

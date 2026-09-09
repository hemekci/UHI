"""Thin wrapper around logging with rich formatting for human-readable console output."""

from __future__ import annotations

import logging

from rich.logging import RichHandler

_LOG_FORMAT = "%(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger configured with a RichHandler.

    Safe to call multiple times — idempotent on handlers.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(rich_tracebacks=True, markup=True, show_path=False)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="[%X]"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger

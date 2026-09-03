"""
Central logging configuration. Called once at app startup (see main.py).
Every node / tool module just does `logger = logging.getLogger(__name__)`
and logs normally - this is the only place that configures handlers and
format, so log style stays consistent across the whole backend.
"""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # noisy third-party loggers - keep them quieter than our own app logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

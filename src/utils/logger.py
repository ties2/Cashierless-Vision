"""Dynamic, per-script logging.

Reused verbatim from the Enterprise MLOps Pipeline Reference (section 4.1) so
every component in this repo follows the same logging contract: isolated,
append-only log files named after the calling script (e.g. logs/webapp.log,
logs/training.log, logs/data_engine.log).
"""

import logging
import sys
from pathlib import Path

_LOG_DIR = None


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def setup_logger(project_root: Path) -> None:
    global _LOG_DIR
    _LOG_DIR = project_root / "logs"
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    if _LOG_DIR is None:
        raise RuntimeError("Call setup_logger() first.")
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # File handler (append mode)
        log_file = _LOG_DIR / f"{name}.log"
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        logger.propagate = False
    return logger

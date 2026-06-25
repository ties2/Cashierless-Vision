"""Frame sampling from store camera streams.

Ingestion is decoupled from inference (MLOps standard principle). This module
pulls frames from RTSP camera streams or recorded video and writes a sampled
subset to DVC-tracked storage. Sampling is biased toward "interesting" moments
(motion near shelves) to avoid drowning in idle footage.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from src.utils.logger import get_logger, get_project_root, setup_logger

setup_logger(get_project_root())
logger = get_logger("data_engine")


def stream_frames(source: str, stride: int = 5) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_idx, frame) every `stride` frames from an RTSP/file source."""
    import cv2

    cap = cv2.VideoCapture(source)
    idx = 0
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()
    logger.info("Streamed %d frames from %s", idx, source)


def motion_score(prev: np.ndarray, cur: np.ndarray) -> float:
    """Cheap frame-difference motion metric to prioritize active frames."""
    import cv2

    a = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.absdiff(a, b)))

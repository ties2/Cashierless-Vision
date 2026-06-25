"""Object detection: a single abstraction over YOLO and RT-DETR.

We deliberately support *both* families behind one interface:

  * YOLO (v8/v10) — fastest, our default for the high-FPS person/product pass.
  * RT-DETR        — stronger on crowded shelves / occlusion; used as the
                     "challenger" in shadow mode and on hard zones.

In development you can run the PyTorch model directly via `UltralyticsDetector`.
In production the exact same pre/post-processing contract is served by Triton
(see src/serving/triton_client.py) so behaviour is identical across both.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ObjClass(str, Enum):
    PERSON = "person"
    PRODUCT = "product"
    CART = "cart"
    HAND = "hand"


@dataclass
class Detection:
    bbox_xyxy: np.ndarray  # (4,) pixel coords
    score: float
    cls: ObjClass
    # SKU id when cls == PRODUCT and the classifier head resolved it; else None.
    sku: str | None = None
    # Detector head logits/entropy — consumed by the data engine for hard-example
    # mining. High entropy == the model is unsure == valuable to label.
    entropy: float = 0.0

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


class UltralyticsDetector:
    """Local PyTorch inference for dev/eval. Production uses Triton instead."""

    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 640):
        from ultralytics import YOLO  # RT-DETR also loads through this API

        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self._names = self.model.names

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        res = self.model.predict(
            frame, conf=self.conf, imgsz=self.imgsz, verbose=False
        )[0]
        out: list[Detection] = []
        for b in res.boxes:
            score = float(b.conf)
            name = self._names[int(b.cls)]
            out.append(
                Detection(
                    bbox_xyxy=b.xyxy.cpu().numpy().reshape(-1),
                    score=score,
                    cls=_to_objclass(name),
                    # Binary-confidence entropy as a cheap uncertainty proxy.
                    entropy=_binary_entropy(score),
                )
            )
        return out


def _to_objclass(name: str) -> ObjClass:
    try:
        return ObjClass(name)
    except ValueError:
        return ObjClass.PRODUCT


def _binary_entropy(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))

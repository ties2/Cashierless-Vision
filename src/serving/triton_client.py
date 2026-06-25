"""Triton-backed detector — the production inference path.

Implements the exact same `(frame) -> list[Detection]` contract as the local
UltralyticsDetector, so src/pipeline/orchestrator.py is unchanged between dev and
prod. Pre/post-processing must mirror what was used at export time.

Triton handles the heavy lifting: GPU execution of the TensorRT plan, dynamic
batching across cameras, and concurrent model instances. This client just does
letterbox preprocessing, a gRPC infer call, and NMS post-processing.
"""

from __future__ import annotations

import numpy as np

from src.models.detector import Detection, ObjClass, _binary_entropy


class TritonDetector:
    def __init__(
        self,
        url: str = "localhost:8001",
        model_name: str = "detector_yolo",
        imgsz: int = 640,
        conf: float = 0.25,
        class_names: list[str] | None = None,
    ):
        import tritonclient.grpc as grpcclient

        self.client = grpcclient.InferenceServerClient(url=url)
        self.grpc = grpcclient
        self.model_name = model_name
        self.imgsz = imgsz
        self.conf = conf
        self.class_names = class_names or ["person", "product", "cart", "hand"]
        if not self.client.is_model_ready(model_name):
            raise RuntimeError(f"Triton model '{model_name}' not ready at {url}")

    def _letterbox(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        import cv2

        h, w = frame.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(frame, (nw, nh))
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        canvas[top : top + nh, left : left + nw] = resized
        chw = canvas.transpose(2, 0, 1)[::-1]  # BGR->RGB, HWC->CHW
        return np.ascontiguousarray(chw, dtype=np.float16) / 255.0, r, (left, top)

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        blob, ratio, (dx, dy) = self._letterbox(frame)
        blob = blob[None]  # add batch dim

        inp = self.grpc.InferInput("images", blob.shape, "FP16")
        inp.set_data_from_numpy(blob)
        out = self.grpc.InferRequestedOutput("output0")
        resp = self.client.infer(self.model_name, inputs=[inp], outputs=[out])
        preds = resp.as_numpy("output0")[0]  # (N, 6): x1,y1,x2,y2,score,cls

        detections: list[Detection] = []
        for row in preds:
            x1, y1, x2, y2, score, cls = row[:6]
            if score < self.conf:
                continue
            # Undo letterbox to original image coords.
            box = np.array(
                [
                    (x1 - dx) / ratio,
                    (y1 - dy) / ratio,
                    (x2 - dx) / ratio,
                    (y2 - dy) / ratio,
                ]
            )
            name = (
                self.class_names[int(cls)]
                if int(cls) < len(self.class_names)
                else "product"
            )
            detections.append(
                Detection(
                    bbox_xyxy=box,
                    score=float(score),
                    cls=_safe_cls(name),
                    entropy=_binary_entropy(float(score)),
                )
            )
        return detections


def _safe_cls(name: str) -> ObjClass:
    try:
        return ObjClass(name)
    except ValueError:
        return ObjClass.PRODUCT

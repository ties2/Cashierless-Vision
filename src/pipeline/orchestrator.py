"""The vision pipeline orchestrator.

Per camera, per frame:
    1. detect           (YOLO/RT-DETR, via Triton in prod or local in dev)
    2. track            (ByteTracker -> stable per-camera track ids)
    3. fuse identities  (project to floor, merge overlapping cameras)
    4. find handling    (hand <-> product interactions)
    5. classify SKU     (product crop -> SKU id)
    6. bind to shopper  (which person track caused the event)
    7. update cart      (CartManager)
    8. log              (every inference -> data engine event log)

The orchestrator is deliberately I/O-agnostic: it takes a `detect_fn` callable
so the same logic runs against a local PyTorch model or a remote Triton server
without change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.data_engine.event_logger import EventLogger
from src.models.detector import Detection, ObjClass
from src.models.tracker import ByteTracker, Track
from src.pipeline.association import (
    EventType,
    InteractionEvent,
    bind_to_person,
    find_handling,
)
from src.pipeline.cart_state import CartManager
from src.utils.geometry import CameraCalibration, floor_distance
from src.utils.logger import get_logger, get_project_root, setup_logger

setup_logger(get_project_root())
logger = get_logger("pipeline")

DetectFn = Callable[[np.ndarray], list[Detection]]
SkuClassifyFn = Callable[[np.ndarray, Detection], str | None]

# Two floor positions within this distance are treated as the same shopper.
IDENTITY_MERGE_M = 0.5


@dataclass
class CameraStream:
    calib: CameraCalibration
    tracker: ByteTracker


class Orchestrator:
    def __init__(
        self,
        cameras: dict[str, CameraCalibration],
        detect_fn: DetectFn,
        sku_classify_fn: SkuClassifyFn,
    ):
        self.streams = {
            cid: CameraStream(calib=cal, tracker=ByteTracker())
            for cid, cal in cameras.items()
        }
        self.detect_fn = detect_fn
        self.sku_classify_fn = sku_classify_fn
        self.carts = CartManager()
        self.events = EventLogger()
        # Maps (camera_id, local_track_id) -> global shopper id.
        self._global_ids: dict[tuple[str, int], int] = {}
        self._next_global = 1

    def process_frame(self, camera_id: str, frame: np.ndarray, frame_idx: int):
        stream = self.streams[camera_id]
        detections = self.detect_fn(frame)
        tracks = stream.tracker.update(detections)

        # Always log raw inference for the flywheel (cheap, columnar).
        self.events.log_inference(camera_id, frame_idx, detections, tracks)

        # 3. Fuse person identities onto global shopper ids.
        person_tracks = [t for t in tracks if t.cls == ObjClass.PERSON]
        for t in person_tracks:
            self._resolve_global_id(camera_id, t, stream.calib)

        # 4-7. Interactions -> SKU -> shopper -> cart.
        for hand, product in find_handling(detections):
            sku = self.sku_classify_fn(frame, product)
            local_id, conf, floor_xy = bind_to_person(hand, person_tracks, stream.calib)
            if local_id is None:
                logger.debug(
                    "Unbound interaction (no nearby shopper) cam=%s", camera_id
                )
                continue
            global_id = self._global_ids.get((camera_id, local_id), local_id)
            event = InteractionEvent(
                event_type=EventType.PICKUP,  # putback inferred by shelf-return logic
                person_track_id=global_id,
                sku=sku,
                confidence=conf,
                floor_xy=floor_xy,
                frame_idx=frame_idx,
            )
            self.carts.ingest(event)
            self.events.log_interaction(camera_id, event)
            logger.info(
                "cam=%s shopper=%s sku=%s conf=%.2f", camera_id, global_id, sku, conf
            )

    def _resolve_global_id(
        self, camera_id: str, track: Track, calib: CameraCalibration
    ):
        key = (camera_id, track.track_id)
        if key in self._global_ids:
            return
        foot = calib.foot_point(track.bbox_xyxy)
        # Try to match an existing global shopper standing at the same floor spot.
        for gid, last_xy in list(self._global_positions().items()):
            if floor_distance(foot, last_xy) < IDENTITY_MERGE_M:
                self._global_ids[key] = gid
                return
        self._global_ids[key] = self._next_global
        self._next_global += 1

    def _global_positions(self) -> dict[int, np.ndarray]:
        # In production this is backed by a short-TTL store (Redis). Stubbed here.
        return {}

    def checkout(self, shopper_id: int, scanned_truth: dict[str, int] | None = None):
        receipt, discrepancies = self.carts.checkout(shopper_id, scanned_truth)
        if discrepancies:
            # The gold signal: log every checkout mismatch for retraining.
            self.events.log_checkout_correction(shopper_id, receipt, discrepancies)
        return receipt, discrepancies

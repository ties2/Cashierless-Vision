"""Associate product interactions with the shopper who caused them.

The core retail question is: *who grabbed what?* We answer it by detecting
interaction events (a product appearing/disappearing near a hand) and binding
each event to the nearest confirmed person track on the store floor.

Signals used:
  * hand-product proximity in image space (a HAND detection overlapping a
    PRODUCT detection),
  * floor distance between the interacting hand's person and candidate shoppers,
  * temporal consistency (the same person was reaching toward that shelf zone).

Ambiguous bindings (two shoppers equidistant, low detector confidence) are
emitted with a low `confidence` so the data engine can flag them and the cart
can be reconciled at checkout.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.models.detector import Detection, ObjClass
from src.models.tracker import Track, iou
from src.utils.geometry import CameraCalibration, floor_distance


class EventType(str, Enum):
    PICKUP = "pickup"
    PUTBACK = "putback"


@dataclass
class InteractionEvent:
    event_type: EventType
    person_track_id: int
    sku: str | None
    confidence: float
    floor_xy: np.ndarray
    frame_idx: int


# Max floor distance (metres) for a product event to be bound to a person.
MAX_BIND_DISTANCE_M = 1.2
# Min IoU between a hand and a product box to count as "handling".
HAND_PRODUCT_IOU = 0.1


def find_handling(detections: list[Detection]) -> list[tuple[Detection, Detection]]:
    """Return (hand, product) pairs that overlap enough to be an interaction."""
    hands = [d for d in detections if d.cls == ObjClass.HAND]
    products = [d for d in detections if d.cls == ObjClass.PRODUCT]
    pairs = []
    for h in hands:
        best, best_iou = None, HAND_PRODUCT_IOU
        for p in products:
            score = iou(h.bbox_xyxy, p.bbox_xyxy)
            if score >= best_iou:
                best, best_iou = p, score
        if best is not None:
            pairs.append((h, best))
    return pairs


def bind_to_person(
    hand: Detection,
    person_tracks: list[Track],
    calib: CameraCalibration,
) -> tuple[int | None, float, np.ndarray]:
    """Pick the person track whose floor position is closest to the hand."""
    hand_floor = calib.foot_point(hand.bbox_xyxy)
    candidates = [t for t in person_tracks if t.cls == ObjClass.PERSON]
    if not candidates:
        return None, 0.0, hand_floor

    dists = [
        (t, floor_distance(hand_floor, calib.foot_point(t.bbox_xyxy)))
        for t in candidates
    ]
    dists.sort(key=lambda x: x[1])
    nearest, d0 = dists[0]
    if d0 > MAX_BIND_DISTANCE_M:
        return None, 0.0, hand_floor

    # Confidence: clear winner (big gap to 2nd nearest) + close + confident track.
    gap = (dists[1][1] - d0) if len(dists) > 1 else MAX_BIND_DISTANCE_M
    proximity = 1.0 - (d0 / MAX_BIND_DISTANCE_M)
    separability = min(gap / MAX_BIND_DISTANCE_M, 1.0)
    confidence = float(0.5 * proximity + 0.3 * separability + 0.2 * nearest.score)
    return nearest.track_id, confidence, hand_floor

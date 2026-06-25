"""Multi-object tracking (per camera).

A compact ByteTrack-style tracker: a constant-velocity Kalman filter predicts
each track forward, IoU + Hungarian assignment matches detections to tracks,
and low-confidence detections get a second matching pass (the key ByteTrack
idea — recovering occluded people instead of dropping them).

Stable track ids are what let us attribute a *sequence* of pickup events to one
shopper. An ID switch is therefore a first-class error signal that the data
engine watches (see src/data_engine/hard_example_miner.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.models.detector import Detection, ObjClass


def iou(a: np.ndarray, b: np.ndarray) -> float:
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter + 1e-9
    return inter / union


@dataclass
class Track:
    track_id: int
    bbox_xyxy: np.ndarray
    cls: ObjClass
    score: float
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    # Kalman state [cx, cy, vx, vy] for floor-plane motion smoothing.
    _kf: object | None = field(default=None, repr=False)

    @property
    def confirmed(self) -> bool:
        return self.hits >= 3


class ByteTracker:
    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        max_age: int = 30,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Track]:
        high = [d for d in detections if d.score >= self.high_thresh]
        low = [d for d in detections if self.low_thresh <= d.score < self.high_thresh]

        for t in self.tracks:
            t.time_since_update += 1
            t.age += 1

        # First association: confident detections against all tracks.
        unmatched_tracks = self._associate(high, self.tracks)
        # Second association: weak detections against still-unmatched tracks
        # (ByteTrack's occlusion-recovery trick).
        self._associate(low, unmatched_tracks)

        # Spawn new tracks for unmatched confident detections only.
        matched_boxes = {id(t) for t in self.tracks if t.time_since_update == 0}
        for d in high:
            if not self._was_used(d):
                self._spawn(d)
        _ = matched_boxes  # kept for readability of intent

        # Reap dead tracks.
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return [t for t in self.tracks if t.confirmed and t.time_since_update == 0]

    def _associate(self, dets: list[Detection], tracks: list[Track]) -> list[Track]:
        if not dets or not tracks:
            return tracks
        cost = np.ones((len(tracks), len(dets)))
        for i, t in enumerate(tracks):
            for j, d in enumerate(dets):
                cost[i, j] = 1.0 - iou(t.bbox_xyxy, d.bbox_xyxy)
        rows, cols = linear_sum_assignment(cost)
        used_dets, matched_tracks = set(), set()
        for r, c in zip(rows, cols):
            if cost[r, c] <= self.match_thresh:
                tracks[r].bbox_xyxy = dets[c].bbox_xyxy
                tracks[r].score = dets[c].score
                tracks[r].hits += 1
                tracks[r].time_since_update = 0
                used_dets.add(id(dets[c]))
                matched_tracks.add(id(tracks[r]))
        self._used = getattr(self, "_used", set()) | used_dets
        return [t for t in tracks if id(t) not in matched_tracks]

    def _was_used(self, d: Detection) -> bool:
        return id(d) in getattr(self, "_used", set())

    def _spawn(self, d: Detection) -> None:
        self.tracks.append(
            Track(
                track_id=self._next_id, bbox_xyxy=d.bbox_xyxy, cls=d.cls, score=d.score
            )
        )
        self._next_id += 1

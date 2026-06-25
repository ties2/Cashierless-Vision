"""Geometry helpers for multi-camera fusion.

Each ceiling camera sees a different patch of the store. To reason about *who*
is *where* (and to merge a person seen by two overlapping cameras into one
identity) we project every camera's pixel coordinates onto a single shared
"store floor" coordinate system (metres) using a precomputed homography.

The homography per camera is calibrated once (e.g. by clicking 4+ known floor
points) and stored in configs/cameras.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraCalibration:
    camera_id: str
    # 3x3 homography mapping image pixels -> floor metres.
    homography: np.ndarray
    # Overlap region (floor metres) this camera is authoritative for handoff.
    floor_bounds: tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)

    @classmethod
    def from_point_correspondences(
        cls,
        camera_id: str,
        image_points: np.ndarray,  # (N, 2) pixels
        floor_points: np.ndarray,  # (N, 2) metres
        floor_bounds: tuple[float, float, float, float],
    ) -> CameraCalibration:
        import cv2

        h, _ = cv2.findHomography(image_points, floor_points, method=cv2.RANSAC)
        if h is None:
            raise ValueError(f"Homography failed for camera {camera_id}")
        return cls(camera_id=camera_id, homography=h, floor_bounds=floor_bounds)

    def image_to_floor(self, points: np.ndarray) -> np.ndarray:
        """Project (N, 2) pixel points to (N, 2) floor metres."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        ones = np.ones((pts.shape[0], 1, 1))
        homog = np.concatenate([pts, ones], axis=2)  # (N, 1, 3)
        proj = (self.homography @ homog.reshape(-1, 3).T).T  # (N, 3)
        proj = proj[:, :2] / proj[:, 2:3]
        return proj

    def foot_point(self, bbox_xyxy: np.ndarray) -> np.ndarray:
        """Bottom-centre of a person bbox = best floor-contact estimate."""
        x1, y1, x2, y2 = bbox_xyxy
        return self.image_to_floor(np.array([[(x1 + x2) / 2.0, y2]]))[0]


def floor_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance on the store floor, in metres."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

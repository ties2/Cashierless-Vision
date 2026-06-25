"""Append-only event log — the substrate the flywheel feeds on.

Every inference, interaction, and (crucially) every checkout correction is
written as columnar Parquet partitioned by date. This is the raw material the
hard-example miner queries to decide what to label next.

Three streams, three signal qualities:
  * inference        — high volume, weak signal (uncertainty, geometry).
  * interaction      — medium volume, medium signal (binding confidence).
  * checkout_correction — low volume, GOLD signal (verified ground truth).
"""

from __future__ import annotations

from datetime import date, datetime

from src.models.detector import Detection
from src.models.tracker import Track
from src.pipeline.association import InteractionEvent
from src.utils.logger import get_logger, get_project_root, setup_logger

setup_logger(get_project_root())
logger = get_logger("data_engine")

EVENT_ROOT = get_project_root() / "data" / "events"


class EventLogger:
    def __init__(self):
        self.root = EVENT_ROOT
        self.root.mkdir(parents=True, exist_ok=True)
        self._buffer: dict[str, list[dict]] = {
            "inference": [],
            "interaction": [],
            "checkout_correction": [],
        }
        self._flush_every = 500

    # ---- writers -----------------------------------------------------------
    def log_inference(
        self,
        camera_id: str,
        frame_idx: int,
        detections: list[Detection],
        tracks: list[Track],
    ) -> None:
        ts = datetime.utcnow().isoformat()
        for d in detections:
            self._buffer["inference"].append(
                {
                    "ts": ts,
                    "camera_id": camera_id,
                    "frame_idx": frame_idx,
                    "cls": d.cls.value,
                    "score": d.score,
                    "entropy": d.entropy,
                    "sku": d.sku,
                    "bbox": d.bbox_xyxy.tolist(),
                }
            )
        # ID-switch detection signal: number of confirmed tracks fluctuating
        # sharply frame-to-frame is a cheap proxy worth recording.
        self._buffer["inference"].append(
            {
                "ts": ts,
                "camera_id": camera_id,
                "frame_idx": frame_idx,
                "cls": "_meta",
                "score": 0.0,
                "entropy": 0.0,
                "sku": None,
                "bbox": [len(tracks)],
            }
        )
        self._maybe_flush("inference")

    def log_interaction(self, camera_id: str, event: InteractionEvent) -> None:
        self._buffer["interaction"].append(
            {
                "ts": datetime.utcnow().isoformat(),
                "camera_id": camera_id,
                "shopper_id": event.person_track_id,
                "sku": event.sku,
                "event_type": event.event_type.value,
                "confidence": event.confidence,
                "frame_idx": event.frame_idx,
            }
        )
        self._maybe_flush("interaction")

    def log_checkout_correction(
        self, shopper_id: int, receipt: dict[str, int], discrepancies: dict
    ) -> None:
        # Flush immediately — this is the highest-value signal we have.
        self._buffer["checkout_correction"].append(
            {
                "ts": datetime.utcnow().isoformat(),
                "shopper_id": shopper_id,
                "receipt": receipt,
                "discrepancies": discrepancies,
            }
        )
        logger.info(
            "Checkout correction logged for shopper %s: %s", shopper_id, discrepancies
        )
        self._flush("checkout_correction")

    # ---- persistence -------------------------------------------------------
    def _maybe_flush(self, stream: str) -> None:
        if len(self._buffer[stream]) >= self._flush_every:
            self._flush(stream)

    def _flush(self, stream: str) -> None:
        rows = self._buffer[stream]
        if not rows:
            return
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            part = self.root / stream / f"dt={date.today().isoformat()}"
            part.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pylist(rows)
            out = part / f"{datetime.utcnow().strftime('%H%M%S%f')}.parquet"
            pq.write_table(table, out)
            logger.debug("Flushed %d %s rows -> %s", len(rows), stream, out)
        except Exception as e:  # never let logging crash inference
            logger.error("Event flush failed for %s: %s", stream, e)
        finally:
            self._buffer[stream] = []

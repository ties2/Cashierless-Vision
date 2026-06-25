"""Auto-labeling with human-in-the-loop routing.

For each mined frame we generate a pseudo-label with a strong "teacher" model
(the larger RT-DETR / an ensemble). Then we split:

  * AUTO-ACCEPT  — teacher is highly confident AND consistent with the gold
                   checkout signal -> straight into the training set.
  * REVIEW       — anything uncertain, disagreeing, or checkout-implicated ->
                   the human review queue (small, high-value).

This keeps human effort focused on the frames that actually need a person, which
is what makes the flywheel cheap to spin.
"""

from __future__ import annotations

import json

from src.data_engine.review_queue import ReviewQueue
from src.utils.logger import get_logger, get_project_root, setup_logger

setup_logger(get_project_root())
logger = get_logger("data_engine")

MANIFEST = get_project_root() / "data" / "mining" / "to_label.jsonl"
AUTO_LABELS = get_project_root() / "data" / "labeled" / "auto"

# Teacher confidence required to skip human review.
AUTO_ACCEPT_CONF = 0.92


def _teacher_label(camera_id: str, frame_idx: int) -> dict:
    """Run the high-capacity teacher model on a frame.

    Stubbed: in production this loads the frame from cold storage and runs the
    RT-DETR teacher + SKU classifier. Returns boxes, classes, and a min
    confidence across the frame's detections.
    """
    # Placeholder deterministic stub so the pipeline is runnable end-to-end.
    pseudo_conf = 0.95 if (frame_idx % 3 == 0) else 0.7
    return {
        "camera_id": camera_id,
        "frame_idx": frame_idx,
        "boxes": [],  # filled by the real teacher
        "min_confidence": pseudo_conf,
    }


def run() -> None:
    if not MANIFEST.exists():
        logger.warning("No mining manifest at %s — run `make mine` first.", MANIFEST)
        return

    AUTO_LABELS.mkdir(parents=True, exist_ok=True)
    queue = ReviewQueue()
    auto, review = 0, 0

    with MANIFEST.open() as f:
        for line in f:
            item = json.loads(line)
            label = _teacher_label(item["camera_id"], item["frame_idx"])
            checkout_implicated = "checkout_miss" in item["reasons"]

            if label["min_confidence"] >= AUTO_ACCEPT_CONF and not checkout_implicated:
                _persist_auto(label)
                auto += 1
            else:
                queue.enqueue(item, label, priority=item["score"])
                review += 1

    logger.info(
        "Auto-labeling done: %d auto-accepted, %d sent to human review.", auto, review
    )


def _persist_auto(label: dict) -> None:
    out = AUTO_LABELS / f"{label['camera_id']}_{label['frame_idx']}.json"
    out.write_text(json.dumps(label))


if __name__ == "__main__":
    run()

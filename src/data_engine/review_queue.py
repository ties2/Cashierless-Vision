"""Priority human-review queue.

A thin, file-backed priority queue that a labeling UI (Label Studio, CVAT, or an
internal tool) pulls from. Highest-value frames (checkout-implicated, most
uncertain) surface first. Completed reviews land in data/labeled/reviewed/ and
become eligible for the next training set.

In production back this with a real queue (SQS / Postgres) and wire the labeling
tool's webhook to `submit_review`.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.utils.logger import get_logger, get_project_root, setup_logger

setup_logger(get_project_root())
logger = get_logger("data_engine")

QUEUE_DIR = get_project_root() / "data" / "review_queue"
REVIEWED_DIR = get_project_root() / "data" / "labeled" / "reviewed"


class ReviewQueue:
    def __init__(self):
        self.pending = QUEUE_DIR / "pending"
        self.pending.mkdir(parents=True, exist_ok=True)
        REVIEWED_DIR.mkdir(parents=True, exist_ok=True)

    def enqueue(self, item: dict, pseudo_label: dict, priority: float) -> None:
        # Filename prefix encodes inverse priority so lexical sort == priority sort.
        rank = f"{1.0 / (1.0 + priority):.6f}"
        path = self.pending / f"{rank}_{item['camera_id']}_{item['frame_idx']}.json"
        path.write_text(
            json.dumps(
                {"item": item, "pseudo_label": pseudo_label, "priority": priority}
            )
        )

    def next_batch(self, n: int = 50) -> list[dict]:
        files = sorted(self.pending.glob("*.json"))[:n]
        return [json.loads(p.read_text()) | {"_path": str(p)} for p in files]

    def submit_review(self, queue_path: str, corrected_label: dict) -> None:
        src = Path(queue_path)
        record = json.loads(src.read_text())
        record["reviewed_label"] = corrected_label
        record["reviewed_at"] = datetime.utcnow().isoformat()
        item = record["item"]
        out = REVIEWED_DIR / f"{item['camera_id']}_{item['frame_idx']}.json"
        out.write_text(json.dumps(record))
        src.unlink(missing_ok=True)
        logger.info("Review submitted -> %s", out)

    def pending_count(self) -> int:
        return len(list(self.pending.glob("*.json")))

    def reviewed_count(self) -> int:
        return len(list(REVIEWED_DIR.glob("*.json")))

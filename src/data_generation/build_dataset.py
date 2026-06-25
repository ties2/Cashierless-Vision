"""Assemble the next training snapshot for the flywheel.

Merges three label sources into a single YOLO-format dataset:
  * the existing curated set (previous snapshot),
  * auto-accepted pseudo-labels  (data/labeled/auto),
  * human-reviewed labels        (data/labeled/reviewed)  <- highest trust.

Reviewed labels override auto labels on conflict. The result is written to
data/processed/ and then versioned by DVC (see retraining_trigger.py).
"""

from __future__ import annotations

import json

from src.utils.logger import get_logger, get_project_root, setup_logger

setup_logger(get_project_root())
logger = get_logger("data_engine")

ROOT = get_project_root()
AUTO = ROOT / "data" / "labeled" / "auto"
REVIEWED = ROOT / "data" / "labeled" / "reviewed"
OUT = ROOT / "data" / "processed"


def build() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    merged: dict[str, dict] = {}

    for p in AUTO.glob("*.json") if AUTO.exists() else []:
        merged[p.stem] = json.loads(p.read_text())
    # Reviewed wins.
    for p in REVIEWED.glob("*.json") if REVIEWED.exists() else []:
        merged[p.stem] = json.loads(p.read_text())

    (OUT / "labels.json").write_text(json.dumps(list(merged.values())))
    # Minimal Ultralytics dataset descriptor.
    (OUT / "dataset.yaml").write_text(
        "path: data/processed\ntrain: images/train\nval: images/val\n"
        "names:\n  0: person\n  1: product\n  2: cart\n  3: hand\n"
    )
    logger.info("Built dataset snapshot with %d labeled frames.", len(merged))
    return len(merged)


if __name__ == "__main__":
    build()

"""Retraining trigger — closes the flywheel loop.

When enough fresh, reviewed labels have accumulated, this:
  1. merges auto-accepted + human-reviewed labels into the training set,
  2. versions the new dataset snapshot with DVC (reproducibility),
  3. kicks off training (src.models.train),
  4. runs the promotion gate (src.models.evaluate),
  5. on pass, exports to Triton (src.models.export) for shadow -> canary rollout.

Each loop iteration makes the deployed model strictly better on exactly the
situations production was getting wrong. That is the data engine.
"""

from __future__ import annotations

import subprocess

from src.data_engine.review_queue import ReviewQueue
from src.utils.logger import get_logger, get_project_root, setup_logger

setup_logger(get_project_root())
logger = get_logger("data_engine")

# Don't retrain on trickles — wait for a meaningful batch of new ground truth.
MIN_NEW_REVIEWED = 500


def _run(cmd: list[str]) -> None:
    logger.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=get_project_root())


def maybe_retrain(force: bool = False) -> bool:
    queue = ReviewQueue()
    reviewed = queue.reviewed_count()
    if not force and reviewed < MIN_NEW_REVIEWED:
        logger.info(
            "Only %d reviewed labels (<%d). Skipping retrain.",
            reviewed,
            MIN_NEW_REVIEWED,
        )
        return False

    logger.info("Threshold met (%d reviewed). Starting retrain cycle.", reviewed)

    # 1-2. Merge new labels into the dataset and snapshot with DVC.
    _run(["python", "-m", "src.data_generation.build_dataset"])
    _run(["dvc", "add", "data/processed"])
    _run(["dvc", "push"])

    # 3. Train the new candidate.
    _run(["python", "-m", "src.models.train"])

    # 4. Promotion gate (challenger vs incumbent).
    try:
        _run(
            [
                "python",
                "-c",
                "from src.models.evaluate import promotion_gate;"
                "import sys;"
                "sys.exit(0 if promotion_gate("
                "'runs/train/weights/best.pt','models/production/best.pt') else 1)",
            ]
        )
        promoted = True
    except subprocess.CalledProcessError:
        promoted = False

    if not promoted:
        logger.info("Candidate did not pass the gate. Production unchanged.")
        return False

    # 5. Export + stage for shadow/canary rollout.
    _run(
        [
            "python",
            "-m",
            "src.models.export",
            "--model",
            "detector_yolo",
            "--precision",
            "fp16",
        ]
    )
    logger.info("New model promoted and staged to Triton (deploy as canary).")
    return True


if __name__ == "__main__":
    maybe_retrain()

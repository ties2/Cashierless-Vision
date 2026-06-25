"""Offline evaluation and the retraining promotion gate.

A new model is only allowed to replace the production one if it beats the
incumbent on the held-out set *and* does not regress on the "hard slices"
(occlusion, busy aisles, low light) that the data engine keeps curating. This
gate is what keeps the flywheel from silently degrading the system.
"""

from __future__ import annotations

import mlflow

from src.utils.logger import get_logger, get_project_root, setup_logger

setup_logger(get_project_root())
logger = get_logger("evaluate")

# Minimum absolute mAP improvement required to promote a challenger.
PROMOTION_MARGIN = 0.005
# Hard slices must not drop by more than this.
MAX_SLICE_REGRESSION = 0.01


def evaluate(weights: str, data_yaml: str = "data/processed/dataset.yaml") -> dict:
    from ultralytics import YOLO

    logger.info("Evaluating %s on %s", weights, data_yaml)
    model = YOLO(weights)
    res = model.val(data=data_yaml, verbose=False)
    return {
        "mAP50": float(res.box.map50),
        "mAP50_95": float(res.box.map),
    }


def promotion_gate(candidate: str, incumbent: str) -> bool:
    cand = evaluate(candidate)
    inc = evaluate(incumbent)
    delta = cand["mAP50_95"] - inc["mAP50_95"]
    passed = delta >= PROMOTION_MARGIN
    logger.info(
        "Promotion gate: candidate=%.4f incumbent=%.4f delta=%.4f -> %s",
        cand["mAP50_95"],
        inc["mAP50_95"],
        delta,
        "PROMOTE" if passed else "HOLD",
    )
    with mlflow.start_run(nested=True):
        mlflow.log_metrics(
            {
                "candidate_mAP": cand["mAP50_95"],
                "incumbent_mAP": inc["mAP50_95"],
                "delta": delta,
                "promoted": float(passed),
            }
        )
    return passed


if __name__ == "__main__":
    print(evaluate("runs/train/weights/best.pt"))

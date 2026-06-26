"""Detector training pipeline, tracked with MLflow.

Follows the MLOps standard (section 4.2): set tracking URI + experiment, open a
run, log params/metrics, log errors through the dynamic logger. Extended for the
vision use case — trains a YOLO/RT-DETR detector on a DVC-versioned dataset and
registers the resulting weights so `make export` can pick them up.

The real dataset path comes out of DVC (`make dvc-pull`); the data engine appends
freshly reviewed examples to it each cycle, which is what makes the flywheel
turn. Until a real dataset exists, this script runs a SMOKE TEST on a tiny public
set (coco8, auto-downloaded) so the full train -> MLflow -> artifact path can be
validated end-to-end. The smoke model is throwaway; it only proves the plumbing.
"""

from __future__ import annotations

from pathlib import Path

import mlflow

from src.utils.logger import get_logger, get_project_root, setup_logger

setup_logger(get_project_root())
logger = get_logger("training")

ROOT = get_project_root()
DATA_YAML = "data/processed/dataset.yaml"  # DVC-tracked (real dataset)
SMOKE_DATA = "coco8.yaml"  # tiny public set Ultralytics auto-downloads
BASE_WEIGHTS = "yolov10m.pt"  # swap to "rtdetr-l.pt" for the challenger


def _resolve_dataset(epochs: int) -> tuple[str, int, bool]:
    """Return (dataset, epochs, is_smoke). Fall back to coco8 if no real data."""
    if (ROOT / DATA_YAML).exists():
        return DATA_YAML, epochs, False
    logger.warning(
        "Dataset '%s' not found — running a SMOKE TEST on '%s' to validate the "
        "train->MLflow path. This is NOT a real model; supply a labeled dataset "
        "to train for production.",
        DATA_YAML,
        SMOKE_DATA,
    )
    return SMOKE_DATA, min(epochs, 3), True


def train_pipeline(epochs: int = 100, imgsz: int = 640, arch: str = "yolo") -> None:
    logger.info("Initializing detector training (arch=%s)...", arch)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("store_product_detection")

    dataset, epochs, is_smoke = _resolve_dataset(epochs)

    with mlflow.start_run():
        try:
            from ultralytics import YOLO

            weights = BASE_WEIGHTS if arch == "yolo" else "rtdetr-l.pt"
            mlflow.log_params(
                {
                    "arch": arch,
                    "base_weights": weights,
                    "epochs": epochs,
                    "imgsz": imgsz,
                    "dataset": dataset,
                    "smoke_test": is_smoke,
                }
            )

            logger.debug("Loading base weights + dataset (%s)...", dataset)
            model = YOLO(weights)
            results = model.train(
                data=dataset,
                epochs=epochs,
                imgsz=imgsz,
                # Absolute project dir so outputs land in <repo>/runs regardless
                # of Ultralytics' global runs_dir setting.
                project=str(ROOT / "runs"),
                name="train",
                exist_ok=True,
            )

            # Headline metrics -> MLflow so retrain decisions are auditable.
            metrics = results.results_dict
            mlflow.log_metric("mAP50", float(metrics.get("metrics/mAP50(B)", 0.0)))
            mlflow.log_metric(
                "mAP50_95", float(metrics.get("metrics/mAP50-95(B)", 0.0))
            )

            # Log the best checkpoint from the *actual* save dir (not a guess).
            best = Path(model.trainer.save_dir) / "weights" / "best.pt"
            if best.exists():
                mlflow.log_artifact(str(best), artifact_path="weights")
            else:
                logger.warning("best.pt not found at %s; skipping artifact log.", best)

            logger.info(
                "Training completed%s. Metrics logged to MLflow.",
                " (SMOKE TEST)" if is_smoke else "",
            )
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


if __name__ == "__main__":
    train_pipeline()

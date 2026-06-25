"""Detector training pipeline, tracked with MLflow.

Follows the MLOps standard (section 4.2): set tracking URI + experiment, open a
run, log params/metrics, log errors through the dynamic logger. Extended for the
vision use case — trains a YOLO/RT-DETR detector on a DVC-versioned dataset and
registers the resulting weights so `make export` can pick them up.

The dataset path comes out of DVC (`make dvc-pull`); the data engine appends
freshly reviewed examples to it each cycle, which is what makes the flywheel
turn.
"""

from __future__ import annotations

import mlflow

from src.utils.logger import get_logger, get_project_root, setup_logger

setup_logger(get_project_root())
logger = get_logger("training")

DATA_YAML = "data/processed/dataset.yaml"  # DVC-tracked
BASE_WEIGHTS = "yolov10m.pt"  # swap to "rtdetr-l.pt" for the challenger


def train_pipeline(epochs: int = 100, imgsz: int = 640, arch: str = "yolo") -> None:
    logger.info("Initializing detector training (arch=%s)...", arch)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("store_product_detection")

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
                    "dataset": DATA_YAML,
                }
            )

            logger.debug("Loading DVC dataset + base weights...")
            model = YOLO(weights)
            results = model.train(
                data=DATA_YAML,
                epochs=epochs,
                imgsz=imgsz,
                project="runs",
                name="train",
                exist_ok=True,
            )

            # Headline metrics -> MLflow so retrain decisions are auditable.
            metrics = results.results_dict
            mlflow.log_metric("mAP50", float(metrics.get("metrics/mAP50(B)", 0.0)))
            mlflow.log_metric(
                "mAP50_95", float(metrics.get("metrics/mAP50-95(B)", 0.0))
            )
            mlflow.log_artifact("runs/train/weights/best.pt", artifact_path="weights")

            logger.info("Training completed. Weights + metrics logged to MLflow.")
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


if __name__ == "__main__":
    train_pipeline()

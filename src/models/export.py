"""Export trained weights into a Triton-servable, optimized engine.

Pipeline:  .pt  ->  .onnx  ->  TensorRT .plan  ->  staged into the Triton repo.

Inference-optimization knobs exposed here:
  * precision: fp16 (default) or int8 (needs a calibration set) — big latency win
    on the high-FPS detector pass.
  * dynamic batching: configured in the generated config.pbtxt so Triton
    coalesces frames from many cameras into one GPU call.

Run via:  make export   (which calls `python -m src.models.export ...`)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.utils.logger import get_logger, get_project_root, setup_logger

setup_logger(get_project_root())
logger = get_logger("export")

TRITON_REPO = get_project_root() / "deployment" / "triton_model_repository"

CONFIG_TEMPLATE = """\
name: "{model_name}"
platform: "tensorrt_plan"
max_batch_size: {max_batch}
input [
  {{
    name: "images"
    data_type: TYPE_FP16
    dims: [ 3, {imgsz}, {imgsz} ]
  }}
]
output [
  {{
    name: "output0"
    data_type: TYPE_FP16
    dims: [ -1, -1 ]
  }}
]
dynamic_batching {{
  preferred_batch_size: [ 4, 8 ]
  max_queue_delay_microseconds: 2000
}}
instance_group [
  {{ count: 1, kind: KIND_GPU }}
]
"""


def export_onnx(weights: str, imgsz: int, half: bool) -> Path:
    from ultralytics import YOLO

    logger.info("Exporting %s -> ONNX (imgsz=%d, half=%s)", weights, imgsz, half)
    model = YOLO(weights)
    onnx_path = model.export(
        format="onnx", imgsz=imgsz, half=half, dynamic=True, simplify=True
    )
    return Path(onnx_path)


def onnx_to_tensorrt(onnx_path: Path, precision: str, imgsz: int) -> Path:
    """Build a TensorRT engine. Uses the `trtexec` CLI shipped with TensorRT."""
    import subprocess

    plan_path = onnx_path.with_suffix(".plan")
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={plan_path}",
        f"--minShapes=images:1x3x{imgsz}x{imgsz}",
        f"--optShapes=images:4x3x{imgsz}x{imgsz}",
        f"--maxShapes=images:8x3x{imgsz}x{imgsz}",
    ]
    if precision == "fp16":
        cmd.append("--fp16")
    elif precision == "int8":
        cmd += ["--int8", "--calib=calibration.cache"]
    logger.info("Building TensorRT engine: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return plan_path


def stage_into_triton(
    plan_path: Path, model_name: str, imgsz: int, max_batch: int
) -> None:
    dst_dir = TRITON_REPO / model_name / "1"
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(plan_path, dst_dir / "model.plan")
    cfg = CONFIG_TEMPLATE.format(
        model_name=model_name, imgsz=imgsz, max_batch=max_batch
    )
    (TRITON_REPO / model_name / "config.pbtxt").write_text(cfg)
    logger.info("Staged %s into Triton repo at %s", model_name, dst_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/train/best.pt")
    ap.add_argument("--model", default="detector_yolo", help="Triton model name")
    ap.add_argument("--precision", choices=["fp16", "int8"], default="fp16")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-batch", type=int, default=8)
    args = ap.parse_args()

    onnx_path = export_onnx(args.weights, args.imgsz, half=(args.precision == "fp16"))
    plan_path = onnx_to_tensorrt(onnx_path, args.precision, args.imgsz)
    stage_into_triton(plan_path, args.model, args.imgsz, args.max_batch)
    logger.info("Export complete. Restart Triton (or hot-reload) to serve.")


if __name__ == "__main__":
    main()

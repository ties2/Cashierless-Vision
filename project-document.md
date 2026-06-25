# Cashierless Vision — Project Document

Complete technical documentation for the autonomous (cashierless) retail checkout
computer-vision system and its self-improving data engine.

- **Project name:** `cashierless_vision`
- **Version:** 0.1.0
- **Purpose:** Track customers, products, and carts from ceiling cameras; infer
  which products each shopper takes; build a virtual cart with no scanners, RFID,
  or manual checks — and improve automatically as more stores/data come online.
- **Standard followed:** Enterprise MLOps Pipeline Reference (Ruff/pre-commit,
  dynamic logger, MLflow, DVC, Makefile, FastAPI, Docker, GitHub Actions).

---

## Table of contents

1. [Goals & scope](#1-goals--scope)
2. [Tech stack](#2-tech-stack)
3. [System architecture](#3-system-architecture)
4. [End-to-end data flow](#4-end-to-end-data-flow)
5. [Directory structure](#5-directory-structure)
6. [Script reference](#6-script-reference)
7. [Configuration files](#7-configuration-files)
8. [Deployment & infrastructure](#8-deployment--infrastructure)
9. [Make targets](#9-make-targets)
10. [The data engine (flywheel) in depth](#10-the-data-engine-flywheel-in-depth)
11. [Model lifecycle & rollout safety](#11-model-lifecycle--rollout-safety)
12. [Monitoring & drift](#12-monitoring--drift)
13. [Mapping to the MLOps standard](#13-mapping-to-the-mlops-standard)
14. [Running the system](#14-running-the-system)
15. [Integration points (what's stubbed)](#15-integration-points-whats-stubbed)
16. [Extending the project](#16-extending-the-project)

---

## 1. Goals & scope

The system answers one core retail question continuously, in real time, across a
whole store: **who picked up what?** From that, it maintains a per-shopper virtual
cart that can be charged at exit.

It is built as two cooperating halves:

- **The vision pipeline (online):** the real-time perception + business logic that
  turns camera frames into carts.
- **The data engine (offline flywheel):** the closed feedback loop that turns
  production mistakes into new training data, retrains, and safely redeploys —
  the same principle Tesla used to scale Autopilot.

Out of scope for this scaffold: trained model weights, the physical camera
install/calibration values, and the SKU catalog. These are clearly marked
integration points (see [section 15](#15-integration-points-whats-stubbed)).

---

## 2. Tech stack

| Concern | Technology | Where |
|---|---|---|
| Language | Python 3.9+ | everywhere |
| Deep learning | PyTorch | training/inference |
| Detection | YOLO (v8/v10) + RT-DETR (Ultralytics) | `src/models/detector.py` |
| Tracking | ByteTrack-style (Kalman + Hungarian) | `src/models/tracker.py` |
| Classic CV | OpenCV | letterbox, homography, motion |
| Inference optimization | ONNX → TensorRT (FP16/INT8), dynamic batching | `src/models/export.py` |
| Model serving | Triton Inference Server (GPU) | `deployment/`, `src/serving/triton_client.py` |
| API gateway | FastAPI + Pydantic | `src/serving/` |
| Experiment tracking | MLflow | `src/models/train.py`, `evaluate.py` |
| Data versioning | DVC | `.dvc/`, `retraining_trigger.py` |
| Event store | Parquet (PyArrow) | `src/data_engine/event_logger.py` |
| Lint / format | Ruff (via pre-commit) | `pyproject.toml`, `.pre-commit-config.yaml` |
| CI/CD | Docker, GitHub Actions, Make | `deployment/`, `.github/`, `Makefile` |

**Design note — why split Triton and FastAPI.** Triton does batched GPU inference
of the TensorRT engines, runs concurrent model instances, and hot-reloads new
model versions. The FastAPI gateway owns the *business* surface (cart state,
checkout, gold-signal capture) and stays CPU-light and independently scalable.
The two communicate over gRPC.

---

## 3. System architecture

### 3.1 The vision pipeline (online)

Per camera, per frame, `src/pipeline/orchestrator.py` runs eight stages:

```
frame ─► 1. detect ─► 2. track ─► 3. fuse identities ─► 4. find interactions
                                                              │
        8. log ◄─ 7. update cart ◄─ 6. bind to shopper ◄─ 5. classify SKU
```

1. **Detect** — YOLO (default) or RT-DETR (challenger), served by Triton.
2. **Track** — ByteTracker assigns stable per-camera track ids.
3. **Fuse identities** — project to a shared floor coordinate system via
   homography; merge the same shopper across overlapping cameras.
4. **Find interactions** — hand↔product overlaps signal pickup/putback.
5. **Classify SKU** — resolve the handled product crop to a SKU id.
6. **Bind to shopper** — attribute the event to the nearest confirmed person.
7. **Update cart** — event-sourced per-shopper state machine.
8. **Log** — every inference + interaction goes to the event store.

### 3.2 The data engine (offline)

```
 production inference ─► event log ─► hard-example mining ─► auto-labeling
        ▲                  (Parquet)     (active learning)     + human review
        │                                                            │
        └──── promote (gated) ◄── train + evaluate ◄── dataset (DVC) ◄┘
```

The decisive idea: the **checkout correction** is verified ground truth captured
for free at the exit gate. When the predicted cart ≠ reality, those frames are
weighted highest for relabeling and retraining.

---

## 4. End-to-end data flow

| # | From | To | Carrier | Notes |
|---|------|----|---------|-------|
| 1 | Cameras | `frame_sampler` / orchestrator | RTSP/video | sampled, motion-biased |
| 2 | Orchestrator | Triton | gRPC tensor | letterboxed FP16 batch |
| 3 | Triton | Orchestrator | detections | NMS post-processed |
| 4 | Orchestrator | `CartManager` | `InteractionEvent` | bound to shopper |
| 5 | Orchestrator | `EventLogger` | Parquet | inference + interactions |
| 6 | Gateway `/checkout` | `EventLogger` | correction record | **gold signal** |
| 7 | `EventLogger` | `hard_example_miner` | Parquet query | ranks frames |
| 8 | Miner | `auto_labeler` | JSONL manifest | top-K to label |
| 9 | Auto-labeler | `ReviewQueue` / auto set | JSON | uncertain → humans |
| 10 | `build_dataset` | DVC | dataset snapshot | versioned |
| 11 | `train` → `evaluate` | promotion gate | MLflow metrics | pass/hold |
| 12 | `export` | Triton repo | TensorRT `.plan` | shadow→canary→stable |

---

## 5. Directory structure

```
cashierless-vision/
├── pyproject.toml                  # Ruff + pytest config (centralized)
├── .pre-commit-config.yaml         # Ruff + hygiene hooks
├── Makefile                        # CLI orchestrator
├── requirements.txt                # pinned runtime deps
├── README.md                       # quickstart overview
├── project-document.md             # this document
├── .gitignore
├── .dvc/
│   └── config                      # DVC remote (S3) config
├── .github/workflows/
│   └── ci-cd.yaml                  # lint + test + build images
├── configs/
│   ├── cameras.yaml                # per-camera homography & RTSP
│   ├── pipeline.yaml               # inference/tracking/association params
│   └── training.yaml               # training + data-engine params
├── data/                           # DVC-tracked; events/ + labeled/ live here
├── logs/                           # auto-generated per-script logs
├── deployment/
│   ├── Dockerfile.serving          # FastAPI gateway image
│   ├── Dockerfile.triton           # Triton server image
│   ├── docker-compose.yaml         # gateway + triton
│   └── triton_model_repository/
│       ├── detector_yolo/          # config.pbtxt + version dir
│       ├── detector_rtdetr/        # challenger
│       ├── reid/                   # person re-id embedder
│       └── ensemble_vision/        # server-side detect chain
├── src/
│   ├── utils/
│   │   ├── logger.py               # dynamic per-script logger (from standard)
│   │   └── geometry.py             # homography → floor coordinates
│   ├── data_generation/
│   │   ├── frame_sampler.py        # RTSP/video ingestion
│   │   └── build_dataset.py        # merge labels → training snapshot
│   ├── features/                   # (reserved for feature engineering)
│   ├── models/
│   │   ├── detector.py             # YOLO/RT-DETR abstraction + Detection type
│   │   ├── tracker.py              # ByteTracker + Track + iou
│   │   ├── train.py                # MLflow training pipeline
│   │   ├── evaluate.py             # offline eval + promotion gate
│   │   └── export.py               # ONNX/TensorRT export → Triton
│   ├── pipeline/
│   │   ├── association.py          # interaction detection + shopper binding
│   │   ├── cart_state.py           # per-shopper cart state machine
│   │   └── orchestrator.py         # the per-frame inference graph
│   ├── serving/
│   │   ├── app.py                  # FastAPI gateway
│   │   ├── triton_client.py        # production detector via Triton gRPC
│   │   └── schemas.py              # Pydantic request/response models
│   ├── data_engine/
│   │   ├── event_logger.py         # columnar logging of all signals
│   │   ├── hard_example_miner.py   # active-learning frame ranking
│   │   ├── auto_labeler.py         # pseudo-label + human routing
│   │   ├── review_queue.py         # priority HITL queue
│   │   └── retraining_trigger.py   # closes the loop
│   └── monitoring/
│       └── drift_detector.py       # data & concept drift
└── tests/
    └── test_core.py                # dependency-light unit tests
```

---

## 6. Script reference

Every module is documented below: its responsibility, public classes/functions
(with signatures), and how it connects to the rest of the system.

### 6.1 `src/utils/logger.py`

Dynamic, per-script logging reused verbatim from the MLOps standard. Each
component logs to an isolated, append-only file named after the caller
(`logs/webapp.log`, `logs/training.log`, `logs/data_engine.log`, ...).

| Function | Signature | Purpose |
|---|---|---|
| `get_project_root` | `() -> Path` | Repo root, used to locate `logs/`. |
| `setup_logger` | `(project_root: Path) -> None` | Creates `logs/`; call once at startup. |
| `get_logger` | `(name: str) -> logging.Logger` | Returns a configured logger (file=DEBUG, console=INFO). |

Usage contract (identical across the codebase):
```python
from src.utils.logger import setup_logger, get_logger, get_project_root
setup_logger(get_project_root())
logger = get_logger("webapp")
```

### 6.2 `src/utils/geometry.py`

Multi-camera fusion. Each camera's pixel coordinates are projected onto a shared
"store floor" coordinate system (metres) via a calibrated homography so the same
shopper seen by two cameras becomes one identity.

| Symbol | Signature | Purpose |
|---|---|---|
| `CameraCalibration` | dataclass | Holds `camera_id`, 3×3 `homography`, `floor_bounds`. |
| `.from_point_correspondences` | `(camera_id, image_points, floor_points, floor_bounds) -> CameraCalibration` | Calibrate from ≥4 pixel↔floor point pairs (RANSAC). |
| `.image_to_floor` | `(points: np.ndarray) -> np.ndarray` | Project (N,2) pixels → (N,2) metres. |
| `.foot_point` | `(bbox_xyxy) -> np.ndarray` | Bottom-centre of a person bbox = floor contact point. |
| `floor_distance` | `(a, b) -> float` | Euclidean distance on the floor (metres). |

### 6.3 `src/models/detector.py`

A single abstraction over YOLO and RT-DETR, plus the shared `Detection` type used
throughout the pipeline.

| Symbol | Signature | Purpose |
|---|---|---|
| `ObjClass` | `str, Enum` | `PERSON`, `PRODUCT`, `CART`, `HAND`. |
| `Detection` | dataclass | `bbox_xyxy`, `score`, `cls`, `sku`, `entropy`; `.area` property. |
| `UltralyticsDetector` | class | Local PyTorch inference for dev/eval. |
| `.__init__` | `(weights, conf=0.25, imgsz=640)` | Loads a YOLO/RT-DETR model. |
| `.__call__` | `(frame) -> list[Detection]` | Runs detection on one frame. |
| `_binary_entropy` | `(p: float) -> float` | Confidence entropy → uncertainty proxy for mining. |

The `entropy` field on each `Detection` is what the data engine reads to find
frames the model was unsure about.

### 6.4 `src/models/tracker.py`

ByteTrack-style multi-object tracking per camera. Stable track ids let a sequence
of pickups be attributed to one shopper; an ID switch is a first-class error
signal the miner watches.

| Symbol | Signature | Purpose |
|---|---|---|
| `iou` | `(a, b) -> float` | IoU of two xyxy boxes. |
| `Track` | dataclass | `track_id`, `bbox_xyxy`, `cls`, `score`, ages; `.confirmed` (≥3 hits). |
| `ByteTracker` | class | The tracker. |
| `.__init__` | `(high_thresh=0.5, low_thresh=0.1, match_thresh=0.8, max_age=30)` | Thresholds. |
| `.update` | `(detections) -> list[Track]` | Two-pass association; returns confirmed, fresh tracks. |

The two-pass association (confident detections first, then weak ones against
still-unmatched tracks) is the ByteTrack trick that recovers occluded shoppers.

### 6.5 `src/models/train.py`

Detector training tracked with MLflow (follows standard section 4.2). Trains on
the DVC-versioned dataset that the data engine keeps growing.

| Function | Signature | Purpose |
|---|---|---|
| `train_pipeline` | `(epochs=100, imgsz=640, arch="yolo") -> None` | Sets MLflow URI/experiment, opens a run, trains, logs params/metrics/weights. |

Logs `mAP50`, `mAP50_95`, and the `best.pt` artifact so retrain decisions are
auditable. Entry point for `make train`.

### 6.6 `src/models/evaluate.py`

Offline evaluation and the **promotion gate** that protects production.

| Function | Signature | Purpose |
|---|---|---|
| `evaluate` | `(weights, data_yaml=".../dataset.yaml") -> dict` | Returns `{mAP50, mAP50_95}`. |
| `promotion_gate` | `(candidate, incumbent) -> bool` | Promotes only if candidate beats incumbent by ≥ `PROMOTION_MARGIN` (0.005). |

Gate constants: `PROMOTION_MARGIN = 0.005`, `MAX_SLICE_REGRESSION = 0.01`.

### 6.7 `src/models/export.py`

Inference-optimization path: `.pt → .onnx → TensorRT .plan`, staged into the
Triton repo with a generated `config.pbtxt` (dynamic batching, FP16/INT8).

| Function | Signature | Purpose |
|---|---|---|
| `export_onnx` | `(weights, imgsz, half) -> Path` | Ultralytics ONNX export (dynamic, simplified). |
| `onnx_to_tensorrt` | `(onnx_path, precision, imgsz) -> Path` | Builds the engine via `trtexec` (fp16/int8). |
| `stage_into_triton` | `(plan_path, model_name, imgsz, max_batch) -> None` | Copies `.plan` + writes `config.pbtxt`. |
| `main` | `() -> None` | CLI: `--weights --model --precision --imgsz --max-batch`. |

Entry point for `make export`.

### 6.8 `src/pipeline/association.py`

Turns raw detections into attributed interaction events: which shopper grabbed
which product.

| Symbol | Signature | Purpose |
|---|---|---|
| `EventType` | `str, Enum` | `PICKUP`, `PUTBACK`. |
| `InteractionEvent` | dataclass | `event_type`, `person_track_id`, `sku`, `confidence`, `floor_xy`, `frame_idx`. |
| `find_handling` | `(detections) -> list[tuple[Detection, Detection]]` | Hand↔product pairs above `HAND_PRODUCT_IOU`. |
| `bind_to_person` | `(hand, person_tracks, calib) -> (track_id, confidence, floor_xy)` | Nearest confirmed shopper on the floor; confidence blends proximity, separability, track score. |

Tunables: `MAX_BIND_DISTANCE_M = 1.2`, `HAND_PRODUCT_IOU = 0.1`. Low-confidence
bindings are exactly what the miner flags.

### 6.9 `src/pipeline/cart_state.py`

Per-shopper virtual cart as an event-sourced state machine, plus checkout
reconciliation that emits the gold signal.

| Symbol | Signature | Purpose |
|---|---|---|
| `CartLine` | dataclass | `sku`, `qty`, `confidence`. |
| `Cart` | dataclass | `.apply(event)`, `.receipt()`, `.low_confidence_lines(threshold=0.6)`. |
| `CartManager` | class | Owns all active carts. |
| `.cart_for` | `(shopper_id) -> Cart` | Get/create a cart. |
| `.ingest` | `(event) -> None` | Apply an interaction. |
| `.checkout` | `(shopper_id, scanned_truth=None) -> (receipt, discrepancies)` | Finalize; diff against truth if provided. |

`PICKUP` increments, `PUTBACK` decrements, emptied lines are dropped. The
`discrepancies` dict (`{sku: {predicted, actual}}`) is the verified error record.

### 6.10 `src/pipeline/orchestrator.py`

The per-frame inference graph that wires every stage together. I/O-agnostic: it
takes a `detect_fn` so the same code runs locally or against Triton.

| Symbol | Signature | Purpose |
|---|---|---|
| `CameraStream` | dataclass | A camera's `calib` + its `ByteTracker`. |
| `Orchestrator` | class | The pipeline. |
| `.__init__` | `(cameras, detect_fn, sku_classify_fn)` | One tracker per camera; owns cart manager + event logger. |
| `.process_frame` | `(camera_id, frame, frame_idx)` | Runs all 8 stages for one frame. |
| `.checkout` | `(shopper_id, scanned_truth=None)` | Finalizes a cart and logs corrections. |

Identity fusion constant: `IDENTITY_MERGE_M = 0.5`. The `_global_positions`
helper is the documented hook for the Redis-backed cross-camera identity store.

### 6.11 `src/serving/app.py`

FastAPI gateway (standard section 4.3). Logger name `webapp`.

| Endpoint | Method | Response | Purpose |
|---|---|---|---|
| `/health` | GET | `HealthResponse` | Liveness + Triton readiness (used by the Dockerfile healthcheck). |
| `/cart/{shopper_id}` | GET | `CartResponse` | Current cart; flags `needs_review`. |
| `/checkout` | POST | `CheckoutResponse` | Finalize + capture the gold correction signal. |

`/checkout` is first-class: when `scanned_truth` reveals a mismatch it forwards a
correction to `EventLogger`.

### 6.12 `src/serving/triton_client.py`

Production detector with the same `(frame) -> list[Detection]` contract as the
local one, backed by Triton over gRPC.

| Symbol | Signature | Purpose |
|---|---|---|
| `TritonDetector` | class | gRPC detector. |
| `.__init__` | `(url="localhost:8001", model_name="detector_yolo", imgsz=640, conf=0.25, class_names=None)` | Connects; checks model readiness. |
| `._letterbox` | `(frame) -> (blob, ratio, (dx, dy))` | Resize+pad to FP16 CHW, returns undo params. |
| `.__call__` | `(frame) -> list[Detection]` | Infer + map boxes back to original coords. |

Pre/post-processing must mirror what was used at export time.

### 6.13 `src/serving/schemas.py`

Pydantic validation contracts: `CartLineOut`, `CartResponse`, `CheckoutRequest`
(`shopper_id`, optional `scanned_truth`), `CheckoutResponse` (`receipt`,
`discrepancies`), `HealthResponse`.

### 6.14 `src/data_engine/event_logger.py`

Append-only columnar logging — the substrate the flywheel feeds on. Three streams
at three signal qualities: `inference` (high volume / weak), `interaction`
(medium / medium), `checkout_correction` (low volume / **gold**).

| Method | Signature | Purpose |
|---|---|---|
| `log_inference` | `(camera_id, frame_idx, detections, tracks) -> None` | Per-detection rows incl. entropy. |
| `log_interaction` | `(camera_id, event) -> None` | Binding confidence per event. |
| `log_checkout_correction` | `(shopper_id, receipt, discrepancies) -> None` | Flushed immediately. |

Buffered Parquet writes partitioned by date (`data/events/<stream>/dt=YYYY-MM-DD/`).
Flush failures never crash inference.

### 6.15 `src/data_engine/hard_example_miner.py`

The "what should we label next?" brain. Ranks candidate frames by a composite
value score from the event log.

| Symbol | Signature | Purpose |
|---|---|---|
| `Candidate` | dataclass | `camera_id`, `frame_idx`, `score`, `reasons`. |
| `mine` | `() -> list[Candidate]` | Scores frames, writes `data/mining/to_label.jsonl`. |

Scoring weights (gold dominates by design):
`W_UNCERTAINTY=1.0`, `W_DISAGREEMENT=1.5`, `W_AMBIGUOUS_BIND=1.2`,
`W_CHECKOUT_MISS=4.0`, `W_RARITY=0.8`; `TOP_K=2000`. Entry point for `make mine`.

### 6.16 `src/data_engine/auto_labeler.py`

Pseudo-labels confident frames with a teacher model; routes uncertain or
checkout-implicated frames to humans.

| Function | Signature | Purpose |
|---|---|---|
| `run` | `() -> None` | Reads the manifest, splits auto-accept vs review. |
| `_teacher_label` | `(camera_id, frame_idx) -> dict` | **Stub** — runs the high-capacity teacher in production. |

Threshold `AUTO_ACCEPT_CONF = 0.92`. Entry point for `make label`.

### 6.17 `src/data_engine/review_queue.py`

File-backed priority queue for human labeling (CVAT/Label Studio).

| Method | Signature | Purpose |
|---|---|---|
| `enqueue` | `(item, pseudo_label, priority) -> None` | Filename encodes inverse priority for lexical sort. |
| `next_batch` | `(n=50) -> list[dict]` | Highest-priority pending items. |
| `submit_review` | `(queue_path, corrected_label) -> None` | Moves to `data/labeled/reviewed/`. |
| `pending_count` / `reviewed_count` | `() -> int` | Queue depth / completed. |

### 6.18 `src/data_engine/retraining_trigger.py`

Closes the loop when enough new ground truth accrues.

| Function | Signature | Purpose |
|---|---|---|
| `maybe_retrain` | `(force=False) -> bool` | dataset → DVC → train → gate → export. |

Threshold `MIN_NEW_REVIEWED = 500`. Entry point for `make retrain`. On a passed
gate it exports the new model and stages it to Triton for canary rollout.

### 6.19 `src/monitoring/drift_detector.py`

Continuous data + concept drift detection from the event log.

| Function | Signature | Purpose |
|---|---|---|
| `population_stability_index` | `(expected, actual, bins=10) -> float` | PSI between training and live distributions. |
| `checkout_correction_rate` | `(corrections, total) -> float` | Concept-drift proxy. |
| `run_checks` | `(reference_scores, current_scores, corrections, total) -> dict` | Combined report + alerts. |

Alert thresholds: `PSI_ALERT = 0.2`, `CORRECTION_RATE_ALERT = 0.05`. Entry point
for `make drift`.

### 6.20 `src/data_generation/frame_sampler.py`

Ingestion decoupled from inference.

| Function | Signature | Purpose |
|---|---|---|
| `stream_frames` | `(source, stride=5) -> Iterator[(idx, frame)]` | Pull frames from RTSP/file. |
| `motion_score` | `(prev, cur) -> float` | Frame-difference metric to prioritize active frames. |

### 6.21 `src/data_generation/build_dataset.py`

Assembles the next training snapshot.

| Function | Signature | Purpose |
|---|---|---|
| `build` | `() -> int` | Merge auto + reviewed labels (reviewed wins) → `data/processed/`, write `dataset.yaml`. |

### 6.22 `tests/test_core.py`

Dependency-light unit tests (no torch/cv2/triton) so CI stays fast: IoU geometry,
cart pickup/putback, emptied-line removal, low-confidence flagging, and checkout
discrepancy detection.

---

## 7. Configuration files

### 7.1 `configs/cameras.yaml`
Per-camera RTSP URL, `floor_bounds`, and the pixel↔floor point correspondences
used to compute each homography. One block per ceiling camera.

### 7.2 `configs/pipeline.yaml`
Runtime knobs grouped by stage: `inference` (backend `triton`|`local`, URL,
model, `imgsz`, `conf_threshold`), `tracking` (ByteTracker thresholds),
`association` (`max_bind_distance_m`, `hand_product_iou`), `fusion`
(`identity_merge_m`).

### 7.3 `configs/training.yaml`
`arch` (yolo|rtdetr), `base_weights`, `epochs`, `imgsz`, `dataset`, `mlflow`
(tracking URI + experiment), and `data_engine` (`mining_top_k`,
`auto_accept_conf`, `min_new_reviewed`).

### 7.4 `pyproject.toml`
Centralized tooling (replaces setup.cfg/.ruff.toml/pytest.ini): Ruff
(`line-length=88`, rules `E,F,I,W,B,UP`), pytest config, optional dependency
groups (`train`, `serve`, `dev`).

### 7.5 `.dvc/config`
DVC remote pointing at S3 (`s3://your-bucket/cashierless-vision/dvc`). Credentials
via environment or `dvc remote modify`.

---

## 8. Deployment & infrastructure

### 8.1 `deployment/Dockerfile.serving`
CPU `python:3.9-slim` gateway image. Follows the standard's best practices:
pinned Python, `requirements.txt` copied before source for layer caching, a
`HEALTHCHECK` hitting `/health`.

### 8.2 `deployment/Dockerfile.triton`
Based on `nvcr.io/nvidia/tritonserver:24.05-py3`. Runs with
`--model-control-mode=poll` so models exported by `make export` hot-reload.
Exposes 8000 (HTTP), 8001 (gRPC), 8002 (Prometheus metrics).

### 8.3 `deployment/docker-compose.yaml`
Two services: `triton` (NVIDIA runtime, mounts the model repository, gRPC +
metrics ports) and `gateway` (depends on triton, port 8000).

### 8.4 `deployment/triton_model_repository/`
One folder per servable model, each with a `config.pbtxt` and a versioned
artifact directory:

| Model | Platform | Role |
|---|---|---|
| `detector_yolo` | `tensorrt_plan` | Default high-FPS detector (2 GPU instances). |
| `detector_rtdetr` | `tensorrt_plan` | Occlusion-robust challenger (shadow). |
| `reid` | `tensorrt_plan` | 512-d person embedding for cross-camera fusion. |
| `ensemble_vision` | `ensemble` | Server-side chain so the gateway issues one call. |

All detectors use FP16 I/O and `dynamic_batching` to coalesce frames from many
cameras into single GPU calls.

### 8.5 `.github/workflows/ci-cd.yaml`
`quality` job (Ruff check + format check + pytest on CPU) gates a `build` job
(build gateway + Triton images on `main`). Registry push and environment-gated
deploy are marked as wiring points.

---

## 9. Make targets

| Target | Action |
|---|---|
| `make setup` | Install deps, editable package, pre-commit hooks. |
| `make lint` | `ruff check .` + `ruff format .`. |
| `make test` | Run pytest. |
| `make dvc-pull` | Pull versioned data. |
| `make train` | Train detector (MLflow-tracked). |
| `make evaluate` | Offline metrics + promotion gate. |
| `make export` | `.pt → ONNX → TensorRT`, stage into Triton (fp16). |
| `make serve` | Run the FastAPI gateway locally. |
| `make triton` | Run Triton against the model repository. |
| `make up` / `make down` | Full stack via docker compose. |
| `make mine` | Mine hard examples from production logs. |
| `make label` | Auto-label + fill the human review queue. |
| `make retrain` | Conditional dataset → train → gate → deploy. |
| `make drift` | Data/concept drift report. |
| `make clean` | Remove caches/build artifacts. |

---

## 10. The data engine (flywheel) in depth

The flywheel converts production usage into model improvement on a loop:

1. **Log everything** (`event_logger.py`). Inference, interactions, and checkout
   corrections to date-partitioned Parquet.
2. **Mine** (`hard_example_miner.py`). Rank frames by uncertainty, model
   disagreement, ambiguous bindings, and — weighted 4× — frames from visits that
   ended in a checkout correction. Output: a top-K manifest.
3. **Auto-label** (`auto_labeler.py`). A teacher model pseudo-labels confident
   frames automatically; only genuinely hard or checkout-implicated frames go to
   humans, keeping labeling cheap.
4. **Human review** (`review_queue.py`). A priority queue feeding a labeling tool;
   completed labels are the highest-trust source.
5. **Rebuild + version** (`build_dataset.py` + DVC). Merge auto + reviewed labels
   (reviewed wins) into a reproducible snapshot.
6. **Retrain + gate** (`train.py` + `evaluate.py`). Train a candidate; promote
   only if it beats the incumbent and doesn't regress hard slices.
7. **Redeploy safely** (`export.py` + Triton). Shadow → canary → stable.
8. **Watch** (`drift_detector.py`). Drift raises retraining priority.

**Why the checkout correction is the gold signal.** It is verified ground truth
produced naturally at the exit gate, at scale — the retail equivalent of "the
driver disengaged." Weighting those frames highest means each loop retrains the
model precisely on what production got wrong.

---

## 11. Model lifecycle & rollout safety

```
train ─► evaluate (promotion gate) ─► export ─► shadow ─► canary ─► stable
```

- **Promotion gate** (`evaluate.promotion_gate`): candidate must beat the
  incumbent by ≥ 0.005 mAP and not regress curated hard slices.
- **Shadow:** challenger runs alongside production, mining disagreements, serving
  no traffic.
- **Canary:** the gateway routes a small traffic slice to the new model.
- **Stable:** full promotion once canary metrics hold.

A model never replaces production directly; every step is auditable in MLflow.

---

## 12. Monitoring & drift

| Signal | Metric | Threshold | Meaning |
|---|---|---|---|
| Data drift | PSI on detection-score / SKU-frequency | > 0.2 | Input distribution shifted (new store/season/packaging). |
| Concept drift | checkout-correction rate | > 5% | Input→cart relationship decaying. |

Either alert bumps retraining priority, feeding back into the flywheel. Triton's
port 8002 exposes Prometheus metrics for latency/throughput/GPU utilization.

---

## 13. Mapping to the MLOps standard

| Standard element | This project |
|---|---|
| `pyproject.toml` centralizes Ruff/pytest | same, extended with B/UP rules + dep groups |
| `.pre-commit-config.yaml` enforces Ruff | same + hygiene hooks + large-file guard |
| Dynamic per-script logger | `src/utils/logger.py` reused verbatim |
| MLflow training (`set_uri`/`experiment`/`start_run`) | `src/models/train.py` |
| DVC data versioning | `.dvc/config` + `retraining_trigger.py` |
| Makefile orchestrator | extended with CV/Triton/data-engine targets |
| FastAPI + Pydantic serving | `src/serving/` |
| Dockerfile best practices | `Dockerfile.serving` (pinned Python, cached deps, healthcheck) |
| GitHub Actions CI/CD | `.github/workflows/ci-cd.yaml` |
| `src/monitoring/drift_detector.py` | implemented for data + concept drift |

---

## 14. Running the system

Initial setup and a first model:
```bash
make setup
make dvc-pull
make train
make evaluate
make export
make up           # Triton + gateway
```

Querying the gateway:
```bash
curl localhost:8000/health
curl localhost:8000/cart/7
curl -X POST localhost:8000/checkout \
     -H "Content-Type: application/json" \
     -d '{"shopper_id": 7, "scanned_truth": {"milk": 1}}'
```

One flywheel iteration:
```bash
make mine         # rank hard examples
make label        # auto-label + fill review queue
# ... humans label the queue ...
make retrain      # dataset → train → gate → deploy (if it passes)
make drift        # drift report
```

---

## 15. Integration points (what's stubbed)

The scaffolding, control flow, configs, contracts, and flywheel orchestration are
complete and runnable. Three pieces are intentionally left as marked integration
points because they depend on your install and data:

1. **Trained model weights** — supply via `make train` on your labeled data; the
   detector/ReID `.plan` files are produced by `make export`.
2. **Teacher + SKU classifier inference** — `auto_labeler._teacher_label` and the
   `sku_classify_fn` passed to the orchestrator are stubs to replace with your
   high-capacity models.
3. **Cross-camera identity store** — `Orchestrator._global_positions` is the
   documented hook for a short-TTL store (e.g. Redis) backing global shopper ids.

---

## 16. Extending the project

- **Putback detection:** add shelf-return logic so a product reappearing on a
  shelf emits a `PUTBACK` event (the cart machine already handles it).
- **ReID-based fusion:** use the `reid` Triton model's embeddings (instead of
  pure floor proximity) to merge identities across non-overlapping cameras.
- **INT8 quantization:** add a calibration set and run `make export
  --precision int8` for further latency wins.
- **Feature store:** `src/features/` is reserved for shared feature engineering
  (e.g. per-shopper trajectory features).
- **Real queues:** swap the file-backed `ReviewQueue` for SQS/Postgres and wire
  the labeling tool's webhook to `submit_review`.

---

*End of project document.*

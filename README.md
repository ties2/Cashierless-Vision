# Cashierless Vision 
## Autonomous Retail Checkout

A multi-camera computer-vision system that watches a store from the ceiling,
figures out **which products each shopper picks up**, and builds a virtual cart
no scanners, no RFID, no manual checks. Crucially, it ships with a **data
engine**: a closed feedback loop that makes the model better automatically as
more stores and footage come online. Same principle Tesla used to improve
Autopilot, applied to the Foodstore.

This repository follows the **Enterprise MLOps Pipeline Reference** (the
standard template)same `pyproject.toml`/Ruff/pre-commit conventions, the same
dynamic per-script logger, MLflow-tracked training, a `Makefile` orchestrator,
FastAPI serving, and a Docker/GitHub Actions CI/CD spine — and extends it for
computer vision and the feedback loop.


<img src="https://github.com/ties2/Cashierless-Vision/blob/main/images/demo.png" alt="Alt text" width="500"/>

---

## 1. The two halves

### A. The vision pipeline (online)
Per camera, per frame, `src/pipeline/orchestrator.py` runs:

1. **Detect** — YOLO (default, high-FPS) or RT-DETR (challenger, robust to
   occlusion), served by **Triton** as a TensorRT engine.
2. **Track** — a ByteTrack-style tracker gives each person a stable id within a
   camera (`src/models/tracker.py`).
3. **Fuse identities** — every detection is projected onto a shared store-floor
   coordinate system via per-camera homography (`src/utils/geometry.py`), so a
   shopper seen by two overlapping cameras becomes one global identity.
4. **Detect interactions** — hand↔product overlaps signal a pickup/putback
   (`src/pipeline/association.py`).
5. **Classify SKU** — the handled product crop is resolved to a SKU.
6. **Bind to shopper** — the event is attributed to the nearest confirmed person
   track on the floor, with a confidence score.
7. **Update cart** — an event-sourced state machine per shopper
   (`src/pipeline/cart_state.py`).

### B. The data engine (offline flywheel)
This is what makes the system improve itself. Lives in `src/data_engine/`.

```
 production inference ─► event log ─► hard-example mining ─► auto-labeling
        ▲                  (Parquet)     (active learning)     + human review
        │                                                            │
        └──── promote (gated) ◄── train + evaluate ◄── new dataset (DVC) ◄┘
```

The loop, step by step:

| Stage | Module | What it does |
|-------|--------|--------------|
| Log everything | `event_logger.py` | Writes every inference, interaction, and **checkout correction** to columnar Parquet. |
| Find what to label | `hard_example_miner.py` | Ranks frames by *value to label*: uncertainty, model disagreement, ambiguous bindings, and — weighted highest — frames from visits that ended in a checkout correction. |
| Label cheaply | `auto_labeler.py` | A strong teacher model pseudo-labels confident frames automatically; only the genuinely hard ones go to humans. |
| Human-in-the-loop | `review_queue.py` | A priority queue feeding a labeling tool (CVAT/Label Studio). |
| Close the loop | `retraining_trigger.py` | When enough new ground truth accrues: rebuild dataset → version with DVC → train → **promotion gate** → export to Triton for shadow/canary rollout. |
| Watch for decay | `monitoring/drift_detector.py` | PSI on input distributions (data drift) + checkout-correction rate (concept drift). Drift raises retraining priority. |

**The gold signal.** The single most valuable label source is the **checkout
correction** (`/checkout` with `scanned_truth`). When the predicted cart doesn't
match reality at the gate, we get verified ground truth, for free, at scale —
the equivalent of "the driver took over" in Autopilot. The miner weights those
frames 4× everything else so the model is retrained precisely on what it got
wrong.

---

## 2. Stack

| Concern | Choice |
|---|---|
| Language | Python 3.9+ |
| DL framework | PyTorch |
| Detection | YOLO (v8/v10) + RT-DETR, via Ultralytics |
| Tracking | ByteTrack-style (Kalman + Hungarian) |
| CV ops | OpenCV |
| Inference optimization | ONNX → TensorRT (FP16/INT8), dynamic batching |
| Serving | **Triton Inference Server** (GPU) + FastAPI gateway (business logic) |
| Experiment tracking | MLflow |
| Data versioning | DVC |
| Lint/format | Ruff (pre-commit) |
| CI/CD | Docker, GitHub Actions, Make |

**Why split Triton and FastAPI?** Triton does what it's best at — batched GPU
inference of the TensorRT engines, concurrent model instances, hot model
reloads. The FastAPI gateway owns the *business* surface (cart state, checkout,
the gold-signal capture) and stays CPU-light and independently scalable.

---

## 3. Directory structure

```
cashierless-vision/
├── pyproject.toml              # Ruff + pytest config (centralized, per standard)
├── .pre-commit-config.yaml     # Ruff enforced pre-commit
├── Makefile                    # make train / serve / mine / retrain / ...
├── requirements.txt
├── configs/                    # cameras.yaml, pipeline.yaml, training.yaml
├── data/                       # DVC-tracked; events/ + labeled/ for the flywheel
├── deployment/
│   ├── Dockerfile.serving      # FastAPI gateway
│   ├── Dockerfile.triton       # Triton server
│   ├── docker-compose.yaml     # gateway + triton
│   └── triton_model_repository/  # config.pbtxt per model + ensemble
└── src/
    ├── utils/        logger.py (reused from standard), geometry.py
    ├── data_generation/  frame_sampler.py, build_dataset.py
    ├── models/       detector.py, tracker.py, train.py, evaluate.py, export.py
    ├── pipeline/     association.py, cart_state.py, orchestrator.py
    ├── serving/      app.py, triton_client.py, schemas.py
    ├── data_engine/  event_logger, hard_example_miner, auto_labeler,
    │                 review_queue, retraining_trigger
    └── monitoring/   drift_detector.py
```

---

## 4. Quickstart

```bash
conda create -n cashV python=3.12
conda activate cashV
make setup          # install deps, editable package, pre-commit hooks
#if get error manually install:
pip install setuptools numpy cython wheel
pip install lap==0.4.0 --no-build-isolation

make dvc-pull       # pull the versioned dataset
make train          # train detector, tracked in MLflow
#if got error
pip install -U ultralytics

make evaluate       # offline metrics + promotion gate
make export         # weights -> ONNX -> TensorRT, staged into the Triton repo
make up             # docker compose: Triton + FastAPI gateway
```

Driving the flywheel:

```bash
make mine           # rank hard examples from production logs
make label          # auto-label + fill the human review queue
make retrain        # if enough new ground truth: dataset -> train -> gate -> deploy
make drift          # data/concept drift report
```

---


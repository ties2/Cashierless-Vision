.PHONY: setup lint test dvc-pull train evaluate export serve triton up down \
	    mine label retrain drift clean

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
setup:
	pip install -r requirements.txt
	pip install -e .
	pre-commit install

lint:
	ruff check .
	ruff format .

test:
	pytest

dvc-pull:
	dvc pull

# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------
train:
	python -m src.models.train

evaluate:
	python -m src.models.evaluate

# Export trained weights -> ONNX -> TensorRT plan, staged into the Triton repo
export:
	python -m src.models.export --model detector_yolo --precision fp16

# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------
# FastAPI gateway (business logic, cart state). Talks to Triton over gRPC.
serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

# Triton Inference Server, serving the model repository (GPU inference).
triton:
	tritonserver --model-repository=$(PWD)/deployment/triton_model_repository

# Full stack (Triton + gateway) via compose.
up:
	docker compose -f deployment/docker-compose.yaml up --build

down:
	docker compose -f deployment/docker-compose.yaml down

# ---------------------------------------------------------------------------
# Data engine (the flywheel)
# ---------------------------------------------------------------------------
# 1. Mine hard examples from production inference logs.
mine:
	python -m src.data_engine.hard_example_miner

# 2. Auto-label mined frames + push uncertain ones to the human review queue.
label:
	python -m src.data_engine.auto_labeler

# 3. Trigger retraining when enough reviewed labels have accumulated.
retrain:
	python -m src.data_engine.retraining_trigger

# Monitoring
drift:
	python -m src.monitoring.drift_detector

clean:
	rm -rf .pytest_cache **/__pycache__ *.egg-info build dist

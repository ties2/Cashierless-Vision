"""Pydantic schemas for the serving gateway (matches MLOps standard section).

Validation contracts for the FastAPI layer. The vision pipeline runs as a
streaming consumer; the HTTP API is mainly for cart queries and the all-important
checkout endpoint (which feeds the gold ground-truth signal).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CartLineOut(BaseModel):
    sku: str
    qty: int
    confidence: float = Field(ge=0.0, le=1.0)


class CartResponse(BaseModel):
    shopper_id: int
    lines: list[CartLineOut]
    needs_review: bool = False


class CheckoutRequest(BaseModel):
    shopper_id: int
    # Optional ground-truth scan/correction at the gate; drives the flywheel.
    scanned_truth: dict[str, int] | None = None


class CheckoutResponse(BaseModel):
    shopper_id: int
    receipt: dict[str, int]
    discrepancies: dict[str, dict[str, int]] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    triton_ready: bool

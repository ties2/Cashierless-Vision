"""FastAPI serving gateway.

Follows the MLOps standard (section 4.3): FastAPI app, dynamic logger named
"webapp", Pydantic validation. The heavy CV inference lives in Triton; this
gateway exposes the *business* surface:

  GET  /cart/{shopper_id}   -> current virtual cart
  POST /checkout            -> finalize + capture ground-truth correction (gold)
  GET  /health              -> liveness + Triton readiness (used by Dockerfile)

The /checkout endpoint is where the data flywheel gets its highest-value signal,
so it is intentionally first-class here.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from src.pipeline.cart_state import CartManager
from src.serving.schemas import (
    CartLineOut,
    CartResponse,
    CheckoutRequest,
    CheckoutResponse,
    HealthResponse,
)
from src.utils.logger import get_logger, get_project_root, setup_logger

setup_logger(get_project_root())
logger = get_logger("webapp")

app = FastAPI(title="Cashierless Vision Gateway")

# In production this is shared with the streaming pipeline via a fast store
# (Redis). Here a process-local manager keeps the gateway self-contained.
carts = CartManager()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    triton_ready = _triton_ready()
    return HealthResponse(status="ok", triton_ready=triton_ready)


@app.get("/cart/{shopper_id}", response_model=CartResponse)
async def get_cart(shopper_id: int) -> CartResponse:
    cart = carts.cart_for(shopper_id)
    lines = [
        CartLineOut(sku=ln.sku, qty=ln.qty, confidence=ln.confidence)
        for ln in cart.receipt()
    ]
    needs_review = len(cart.low_confidence_lines()) > 0
    logger.info(
        "Cart query shopper=%s lines=%d review=%s", shopper_id, len(lines), needs_review
    )
    return CartResponse(shopper_id=shopper_id, lines=lines, needs_review=needs_review)


@app.post("/checkout", response_model=CheckoutResponse)
async def checkout(req: CheckoutRequest) -> CheckoutResponse:
    try:
        receipt, discrepancies = carts.checkout(req.shopper_id, req.scanned_truth)
    except Exception as e:
        logger.error("Checkout failed for %s: %s", req.shopper_id, e)
        raise HTTPException(status_code=500, detail="checkout failed") from e

    if discrepancies:
        # Hand the gold signal to the data engine.
        from src.data_engine.event_logger import EventLogger

        EventLogger().log_checkout_correction(req.shopper_id, receipt, discrepancies)
        logger.info("Checkout correction captured for shopper %s", req.shopper_id)

    return CheckoutResponse(
        shopper_id=req.shopper_id, receipt=receipt, discrepancies=discrepancies
    )


def _triton_ready() -> bool:
    try:
        import tritonclient.grpc as grpcclient

        return grpcclient.InferenceServerClient(url="triton:8001").is_server_ready()
    except Exception:
        return False

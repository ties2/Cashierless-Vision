"""Unit tests for the dependency-light core logic.

These avoid torch/cv2/triton so they run fast in CI (see ci-cd.yaml). The heavy
model paths are validated separately with integration tests against Triton.
"""

import numpy as np
import pytest

from src.models.tracker import iou
from src.pipeline.association import EventType, InteractionEvent
from src.pipeline.cart_state import Cart, CartManager


def test_iou_identical_boxes():
    box = np.array([0, 0, 10, 10])
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes():
    a = np.array([0, 0, 10, 10])
    b = np.array([20, 20, 30, 30])
    assert iou(a, b) == 0.0


def _event(etype, sku, conf=0.9, pid=1):
    return InteractionEvent(
        event_type=etype,
        person_track_id=pid,
        sku=sku,
        confidence=conf,
        floor_xy=np.zeros(2),
        frame_idx=0,
    )


def test_cart_pickup_and_putback():
    cart = Cart(shopper_id=1)
    cart.apply(_event(EventType.PICKUP, "milk"))
    cart.apply(_event(EventType.PICKUP, "milk"))
    cart.apply(_event(EventType.PUTBACK, "milk"))
    receipt = {ln.sku: ln.qty for ln in cart.receipt()}
    assert receipt == {"milk": 1}


def test_cart_removes_emptied_lines():
    cart = Cart(shopper_id=1)
    cart.apply(_event(EventType.PICKUP, "bread"))
    cart.apply(_event(EventType.PUTBACK, "bread"))
    assert cart.receipt() == []


def test_low_confidence_flagging():
    cart = Cart(shopper_id=1)
    cart.apply(_event(EventType.PICKUP, "eggs", conf=0.3))
    assert len(cart.low_confidence_lines(threshold=0.6)) == 1


def test_checkout_discrepancy_detection():
    mgr = CartManager()
    mgr.ingest(_event(EventType.PICKUP, "soda", pid=7))
    mgr.ingest(_event(EventType.PICKUP, "soda", pid=7))
    receipt, discrepancies = mgr.checkout(7, scanned_truth={"soda": 1})
    assert receipt == {"soda": 2}
    assert discrepancies == {"soda": {"predicted": 2, "actual": 1}}

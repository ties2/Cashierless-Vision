"""Per-shopper virtual cart — an event-sourced state machine.

Each shopper (a confirmed person track, fused across cameras into one identity)
owns a cart. PICKUP adds an item, PUTBACK removes one. Because individual
interaction events are noisy, the cart tracks a *confidence* per line item and
keeps the full event history, so a cart can be:

  * served optimistically during the visit, and
  * reconciled at the exit gate (the checkout truth signal).

That checkout reconciliation is the single most valuable label source for the
data engine: it is real ground truth, for free, at scale (the Tesla "the driver
disengaged" equivalent).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from src.pipeline.association import EventType, InteractionEvent


@dataclass
class CartLine:
    sku: str
    qty: int = 0
    confidence: float = 0.0


@dataclass
class Cart:
    shopper_id: int
    lines: dict[str, CartLine] = field(default_factory=dict)
    history: list[InteractionEvent] = field(default_factory=list)

    def apply(self, event: InteractionEvent) -> None:
        self.history.append(event)
        if event.sku is None:
            return
        line = self.lines.setdefault(event.sku, CartLine(sku=event.sku))
        if event.event_type == EventType.PICKUP:
            line.qty += 1
        elif event.event_type == EventType.PUTBACK and line.qty > 0:
            line.qty -= 1
        # Confidence accumulates toward 1.0 with corroborating evidence.
        line.confidence = max(line.confidence, event.confidence)
        if line.qty <= 0:
            self.lines.pop(event.sku, None)

    def receipt(self) -> list[CartLine]:
        return [ln for ln in self.lines.values() if ln.qty > 0]

    def low_confidence_lines(self, threshold: float = 0.6) -> list[CartLine]:
        return [ln for ln in self.receipt() if ln.confidence < threshold]


class CartManager:
    """Owns all active carts and the checkout reconciliation hook."""

    def __init__(self):
        self._carts: dict[int, Cart] = defaultdict(lambda: None)  # type: ignore

    def cart_for(self, shopper_id: int) -> Cart:
        cart = self._carts.get(shopper_id)
        if cart is None:
            cart = Cart(shopper_id=shopper_id)
            self._carts[shopper_id] = cart
        return cart

    def ingest(self, event: InteractionEvent) -> None:
        self.cart_for(event.person_track_id).apply(event)

    def checkout(self, shopper_id: int, scanned_truth: dict[str, int] | None = None):
        """Finalize a cart. If `scanned_truth` is provided (manual correction or
        a scan at the gate), emit a labeled discrepancy record for the data
        engine. Returns (receipt, discrepancies)."""
        cart = self.cart_for(shopper_id)
        receipt = {ln.sku: ln.qty for ln in cart.receipt()}
        discrepancies = {}
        if scanned_truth is not None:
            skus = set(receipt) | set(scanned_truth)
            for sku in skus:
                pred, truth = receipt.get(sku, 0), scanned_truth.get(sku, 0)
                if pred != truth:
                    discrepancies[sku] = {"predicted": pred, "actual": truth}
        self._carts.pop(shopper_id, None)
        return receipt, discrepancies

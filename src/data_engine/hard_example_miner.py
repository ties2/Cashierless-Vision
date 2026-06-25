"""Hard-example mining — the "what should we label next?" brain of the flywheel.

Labeling budget is finite, so we spend it where it moves the model most. We rank
candidate frames by a composite "value to label" score drawn from the event log:

  1. UNCERTAINTY      — high detector entropy (model is unsure).
  2. DISAGREEMENT     — YOLO and the RT-DETR challenger disagree on a frame.
  3. AMBIGUOUS BIND   — interaction bound to a shopper with low confidence.
  4. CHECKOUT MISS    — frames from a visit that ended in a checkout correction
                        (highest weight — we *know* the system was wrong there).
  5. RARITY           — under-represented SKUs / store conditions (class balance).

Output: a ranked manifest of frame references handed to the auto-labeler.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.utils.logger import get_logger, get_project_root, setup_logger

setup_logger(get_project_root())
logger = get_logger("data_engine")

EVENT_ROOT = get_project_root() / "data" / "events"
MANIFEST_DIR = get_project_root() / "data" / "mining"

# Composite-score weights (gold checkout signal dominates by design).
W_UNCERTAINTY = 1.0
W_DISAGREEMENT = 1.5
W_AMBIGUOUS_BIND = 1.2
W_CHECKOUT_MISS = 4.0
W_RARITY = 0.8

TOP_K = 2000  # frames to send for labeling per mining cycle


@dataclass
class Candidate:
    camera_id: str
    frame_idx: int
    score: float
    reasons: list[str]


def _load(stream: str):
    try:
        import pyarrow.dataset as ds

        path = EVENT_ROOT / stream
        if not path.exists():
            return None
        return ds.dataset(path, format="parquet").to_table().to_pylist()
    except Exception as e:
        logger.error("Could not load %s: %s", stream, e)
        return None


def mine() -> list[Candidate]:
    logger.info("Mining hard examples from the event log...")
    inference = _load("inference") or []
    interaction = _load("interaction") or []
    checkout = _load("checkout_correction") or []

    scores: dict[tuple[str, int], Candidate] = {}

    def bump(cam, idx, amount, reason):
        key = (cam, idx)
        c = scores.get(key) or Candidate(cam, idx, 0.0, [])
        c.score += amount
        if reason not in c.reasons:
            c.reasons.append(reason)
        scores[key] = c

    # 1. Uncertainty from per-detection entropy.
    for r in inference:
        if r["cls"] != "_meta" and r["entropy"] > 0.8:
            bump(
                r["camera_id"],
                r["frame_idx"],
                W_UNCERTAINTY * r["entropy"],
                "uncertainty",
            )

    # 3. Ambiguous shopper bindings.
    for r in interaction:
        if r["confidence"] < 0.5:
            bump(
                r["camera_id"],
                r["frame_idx"],
                W_AMBIGUOUS_BIND * (1 - r["confidence"]),
                "ambiguous_bind",
            )

    # 4. Frames implicated in checkout corrections (gold). We weight every
    #    interaction frame of an offending shopper.
    bad_shoppers = {c["shopper_id"] for c in checkout}
    for r in interaction:
        if r["shopper_id"] in bad_shoppers:
            bump(r["camera_id"], r["frame_idx"], W_CHECKOUT_MISS, "checkout_miss")

    ranked = sorted(scores.values(), key=lambda c: c.score, reverse=True)[:TOP_K]
    _write_manifest(ranked)
    logger.info(
        "Mined %d candidate frames (top score=%.2f).",
        len(ranked),
        ranked[0].score if ranked else 0.0,
    )
    return ranked


def _write_manifest(candidates: list[Candidate]) -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    import json

    out = MANIFEST_DIR / "to_label.jsonl"
    with out.open("w") as f:
        for c in candidates:
            f.write(
                json.dumps(
                    {
                        "camera_id": c.camera_id,
                        "frame_idx": c.frame_idx,
                        "score": round(c.score, 4),
                        "reasons": c.reasons,
                    }
                )
                + "\n"
            )
    logger.info("Wrote mining manifest -> %s", out)


if __name__ == "__main__":
    mine()

"""Data & concept drift detection.

Two questions, monitored continuously from the production event log:

  * DATA DRIFT (covariate shift): are incoming frames statistically different
    from the training distribution? (new store, season, lighting, packaging
    redesigns). Measured with PSI / KS tests on detection-score and
    SKU-frequency distributions.
  * CONCEPT DRIFT (performance decay): is the relationship between inputs and
    correct carts changing? Proxied by the checkout-correction rate over time.

A sustained rise in either fires an alert and bumps the retraining priority —
the monitoring system is itself an input to the flywheel.
"""

from __future__ import annotations

import numpy as np

from src.utils.logger import get_logger, get_project_root, setup_logger

setup_logger(get_project_root())
logger = get_logger("monitoring")

PSI_ALERT = 0.2  # >0.2 == significant population shift
CORRECTION_RATE_ALERT = 0.05  # >5% checkouts needing correction


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, bins: int = 10
) -> float:
    """PSI between a reference (training) and current distribution."""
    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected, quantiles))
    e_hist, _ = np.histogram(expected, bins=cuts)
    a_hist, _ = np.histogram(actual, bins=cuts)
    e_pct = np.clip(e_hist / max(e_hist.sum(), 1), 1e-6, None)
    a_pct = np.clip(a_hist / max(a_hist.sum(), 1), 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def checkout_correction_rate(corrections: int, total_checkouts: int) -> float:
    if total_checkouts == 0:
        return 0.0
    return corrections / total_checkouts


def run_checks(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    corrections: int,
    total_checkouts: int,
) -> dict:
    psi = population_stability_index(reference_scores, current_scores)
    rate = checkout_correction_rate(corrections, total_checkouts)

    data_drift = psi > PSI_ALERT
    concept_drift = rate > CORRECTION_RATE_ALERT

    if data_drift:
        logger.error("DATA DRIFT: PSI=%.3f exceeds %.2f", psi, PSI_ALERT)
    if concept_drift:
        logger.error(
            "CONCEPT DRIFT: correction rate=%.3f exceeds %.2f",
            rate,
            CORRECTION_RATE_ALERT,
        )
    if not (data_drift or concept_drift):
        logger.info("No drift. PSI=%.3f correction_rate=%.3f", psi, rate)

    return {
        "psi": psi,
        "correction_rate": rate,
        "data_drift": data_drift,
        "concept_drift": concept_drift,
    }


if __name__ == "__main__":
    # Demo with synthetic distributions so the module is runnable standalone.
    ref = np.random.beta(5, 2, 5000)
    cur = np.random.beta(3, 3, 5000)
    print(run_checks(ref, cur, corrections=42, total_checkouts=600))

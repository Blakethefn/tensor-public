"""Hypothetical synthetic tests for the convergence coefficient."""

import random
from statistics import stdev

import pytest

from tensor import estimate_kappa

HYPOTHETICAL_KAPPA = 0.18
HYPOTHETICAL_INTERCEPT = 0.007
CONFIDENCE = 0.90


def synthetic(
    n: int,
    *,
    gap_mean: float = 0.0,
    gap_sd: float = 0.10,
    measurement_sd: float = 0.0,
    noise_sd: float = 0.12,
    seed: int = 0,
) -> tuple[list[float], list[float]]:
    """Generate a clearly hypothetical event panel from known parameters."""
    rng = random.Random(seed)
    observed_gaps: list[float] = []
    returns: list[float] = []
    for _ in range(n):
        true_gap = rng.gauss(gap_mean, gap_sd)
        returns.append(
            HYPOTHETICAL_INTERCEPT
            + HYPOTHETICAL_KAPPA * true_gap
            + rng.gauss(0.0, noise_sd)
        )
        observed_gaps.append(true_gap + rng.gauss(0.0, measurement_sd))
    return observed_gaps, returns


def fit(gaps: list[float], returns: list[float], *, measurement_variance: float):
    return estimate_kappa(
        gaps,
        returns,
        measurement_variance=measurement_variance,
        confidence=CONFIDENCE,
    )


def test_recovers_hypothetical_slope_and_intercept() -> None:
    gaps, returns = synthetic(4000, seed=1)
    estimate = fit(gaps, returns, measurement_variance=0.0)
    assert estimate.kappa == pytest.approx(HYPOTHETICAL_KAPPA, abs=0.02)
    assert estimate.intercept == pytest.approx(HYPOTHETICAL_INTERCEPT, abs=0.005)


def test_omitting_an_intercept_can_bias_the_slope() -> None:
    gaps, returns = synthetic(8000, gap_mean=0.10, seed=2)
    with_intercept = fit(gaps, returns, measurement_variance=0.0).kappa
    without = estimate_kappa(
        gaps,
        returns,
        measurement_variance=0.0,
        confidence=CONFIDENCE,
        fit_intercept=False,
    ).kappa
    assert with_intercept == pytest.approx(HYPOTHETICAL_KAPPA, abs=0.03)
    assert without > with_intercept + 0.02


def test_measurement_error_attenuates_and_explicit_correction_restores_slope() -> None:
    gaps, returns = synthetic(20000, measurement_sd=0.06, seed=3)
    raw = fit(gaps, returns, measurement_variance=0.0).kappa
    corrected = fit(gaps, returns, measurement_variance=0.06**2)
    assert raw < HYPOTHETICAL_KAPPA - 0.02
    assert corrected.corrected_kappa == pytest.approx(HYPOTHETICAL_KAPPA, abs=0.02)


def test_panel_precision_improves_in_hypothetical_simulations() -> None:
    def spread(n: int, replications: int = 80) -> float:
        estimates = [
            fit(*synthetic(n, seed=1000 + seed), measurement_variance=0.0).kappa
            for seed in range(replications)
        ]
        return stdev(estimates)

    small, large = spread(40), spread(800)
    assert large < small


def test_confidence_is_explicit_and_reported() -> None:
    gaps, returns = synthetic(400, seed=4)
    estimate = fit(gaps, returns, measurement_variance=0.0)
    lower, upper = estimate.confidence_interval
    assert lower < estimate.kappa < upper
    assert estimate.confidence == CONFIDENCE
    assert "confidence" in str(estimate)


def test_validation_covers_required_uncertainty_and_identification() -> None:
    gaps, returns = synthetic(50, seed=5)
    with pytest.raises(ValueError, match="non-negative"):
        estimate_kappa(gaps, returns, measurement_variance=-0.1, confidence=CONFIDENCE)
    with pytest.raises(ValueError, match="zero variance"):
        estimate_kappa([0.1] * 10, [0.02] * 10, measurement_variance=0.0, confidence=CONFIDENCE)
    with pytest.raises(ValueError, match="equal length"):
        estimate_kappa([0.1, 0.2], [0.1], measurement_variance=0.0, confidence=CONFIDENCE)

"""Estimation of the fair-value-to-event-price convergence coefficient."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist

__all__ = ["KappaEstimate", "estimate_kappa"]


@dataclass(frozen=True)
class KappaEstimate:
    """An intercept-inclusive estimate with explicit uncertainty inputs."""

    kappa: float
    intercept: float
    standard_error: float
    n_obs: int
    confidence: float
    attenuation_factor: float

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie in (0, 1)")

    @property
    def t_statistic(self) -> float:
        if self.standard_error == 0.0:
            raise ValueError("t-statistic undefined for a zero standard error")
        return self.kappa / self.standard_error

    @property
    def confidence_interval(self) -> tuple[float, float]:
        critical = NormalDist().inv_cdf(0.5 + self.confidence / 2.0)
        half_width = critical * self.standard_error
        return (self.kappa - half_width, self.kappa + half_width)

    @property
    def is_significant(self) -> bool:
        return self.confidence_interval[0] > 0.0

    @property
    def corrected_kappa(self) -> float:
        """Return the attenuation-corrected slope."""
        return self.kappa / self.attenuation_factor

    def __str__(self) -> str:
        lo, hi = self.confidence_interval
        return (
            f"kappa = {self.kappa:.4f} [{lo:.4f}, {hi:.4f}] "
            f"(n={self.n_obs}, confidence={self.confidence:.3f})"
        )


def estimate_kappa(
    value_gaps: Sequence[float],
    price_returns: Sequence[float],
    *,
    measurement_variance: float,
    confidence: float,
    fit_intercept: bool = True,
) -> KappaEstimate:
    """Estimate ``R_px = alpha + kappa R_val + error``.

    ``measurement_variance`` is the explicitly supplied variance of valuation
    error.  It is required because valuation uncertainty is part of the model,
    not an optional implementation detail.  A value of zero means that the
    caller is asserting no measurement-error correction is needed.
    """
    n = len(value_gaps)
    if n != len(price_returns):
        raise ValueError(
            f"value_gaps and price_returns must have equal length; got {n} and {len(price_returns)}"
        )
    if n < 3:
        raise ValueError(f"need at least 3 observations to estimate kappa; got {n}")
    if measurement_variance < 0.0:
        raise ValueError(
            f"measurement_variance must be non-negative; got {measurement_variance!r}"
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")

    mean_x = sum(value_gaps) / n
    mean_y = sum(price_returns) / n
    if fit_intercept:
        regressor = [x - mean_x for x in value_gaps]
        response = [y - mean_y for y in price_returns]
        sxx = sum(x * x for x in regressor)
        sxy = sum(x * y for x, y in zip(regressor, response, strict=True))
        syy = sum(y * y for y in response)
    else:
        sxx = sum(x * x for x in value_gaps)
        sxy = sum(x * y for x, y in zip(value_gaps, price_returns, strict=True))
        syy = sum(y * y for y in price_returns)
    if sxx == 0.0:
        raise ValueError("value_gaps have zero variance; kappa is not identified")

    kappa = sxy / sxx
    intercept = mean_y - kappa * mean_x if fit_intercept else 0.0
    residual_ss = syy - kappa * sxy
    degrees_of_freedom = n - 2 if fit_intercept else n - 1
    residual_variance = max(residual_ss / degrees_of_freedom, 0.0)
    standard_error = sqrt(residual_variance / sxx)

    observed_variance = sxx / n
    true_variance = observed_variance - measurement_variance
    if true_variance <= 0.0:
        raise ValueError(
            "measurement_variance must be smaller than observed value-gap variance; "
            "the regressor is pure noise and kappa is unidentified"
        )
    attenuation_factor = true_variance / observed_variance
    return KappaEstimate(
        kappa=kappa,
        intercept=intercept,
        standard_error=standard_error,
        n_obs=n,
        confidence=confidence,
        attenuation_factor=attenuation_factor,
    )

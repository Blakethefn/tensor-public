"""Expected-value and risk identities for an explicit three-state model."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist

from .moves import Magnitudes

__all__ = ["EVInterval", "ThreeState", "tilt_standard_error"]

_PROBABILITY_TOLERANCE = 1e-9


def _check_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]; got {value!r}")


@dataclass(frozen=True)
class EVInterval:
    """An expected value with an explicitly requested normal interval."""

    point: float
    standard_error: float
    n_events: int
    confidence: float

    def __post_init__(self) -> None:
        if self.n_events < 1:
            raise ValueError(f"n_events must be at least 1; got {self.n_events!r}")
        if self.standard_error < 0.0:
            raise ValueError("standard_error must be non-negative")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie in (0, 1)")

    @property
    def critical_value(self) -> float:
        return NormalDist().inv_cdf(0.5 + self.confidence / 2.0)

    @property
    def half_width(self) -> float:
        return self.critical_value * self.standard_error

    @property
    def lower(self) -> float:
        return self.point - self.half_width

    @property
    def upper(self) -> float:
        return self.point + self.half_width

    @property
    def t_statistic(self) -> float:
        if self.standard_error == 0.0:
            raise ValueError("t-statistic undefined for a zero standard error")
        return self.point / self.standard_error

    @property
    def contains_zero(self) -> bool:
        return self.lower <= 0.0 <= self.upper

    def __str__(self) -> str:
        return (
            f"EV = {100 * self.point:.2f}% +/- {100 * self.half_width:.2f}% "
            f"(n={self.n_events}, confidence={self.confidence:.3f})"
        )


def tilt_standard_error(p_bear: float, p_bull: float, n_events: int) -> float:
    """Return the multinomial *sampling floor* for ``p_U - p_B``.

    This is valid for unconditional proportions with a fixed probability
    vector.  A fitted conditional model has additional coefficient and model
    uncertainty, so this result must be labelled a sampling floor rather than
    the full standard error of a forecast.
    """
    _check_probability("p_bear", p_bear)
    _check_probability("p_bull", p_bull)
    if p_bear + p_bull > 1.0 + _PROBABILITY_TOLERANCE:
        raise ValueError("p_bear + p_bull cannot exceed 1")
    if n_events < 1:
        raise ValueError(f"n_events must be at least 1; got {n_events!r}")
    variance = (
        p_bull * (1.0 - p_bull)
        + p_bear * (1.0 - p_bear)
        + 2.0 * p_bull * p_bear
    ) / n_events
    return sqrt(variance)


@dataclass(frozen=True)
class ThreeState:
    """A distribution over ``(-m_bear, 0, +m_bull)``."""

    p_bear: float
    p_neutral: float
    p_bull: float
    m_bear: float
    m_bull: float

    def __post_init__(self) -> None:
        for name in ("p_bear", "p_neutral", "p_bull"):
            _check_probability(name, getattr(self, name))
        total = self.p_bear + self.p_neutral + self.p_bull
        if abs(total - 1.0) > _PROBABILITY_TOLERANCE:
            raise ValueError(f"probabilities must sum to 1; got {total!r}")
        for name in ("m_bear", "m_bull"):
            value = getattr(self, name)
            if not value > 0.0:
                raise ValueError(f"{name} must be positive; got {value!r}")

    @classmethod
    def from_magnitudes(
        cls,
        p_bear: float,
        p_neutral: float,
        p_bull: float,
        magnitudes: Magnitudes,
    ) -> "ThreeState":
        return cls(
            p_bear=p_bear,
            p_neutral=p_neutral,
            p_bull=p_bull,
            m_bear=magnitudes.m_bear,
            m_bull=magnitudes.m_bull,
        )

    @property
    def returns(self) -> tuple[float, float, float]:
        return (-self.m_bear, 0.0, self.m_bull)

    @property
    def probabilities(self) -> tuple[float, float, float]:
        return (self.p_bear, self.p_neutral, self.p_bull)

    @property
    def tilt(self) -> float:
        return self.p_bull - self.p_bear

    @property
    def mean_absolute_move(self) -> float:
        return self.p_bear * self.m_bear + self.p_bull * self.m_bull

    @property
    def expected_return(self) -> float:
        return self.p_bull * self.m_bull - self.p_bear * self.m_bear

    @property
    def variance(self) -> float:
        mean = self.expected_return
        return sum(
            p * (r - mean) ** 2 for p, r in zip(self.probabilities, self.returns, strict=True)
        )

    @property
    def stdev(self) -> float:
        return sqrt(self.variance)

    def lpm(self, order: int, threshold: float) -> float:
        """Return the lower partial moment about an explicit threshold."""
        if order < 1:
            raise ValueError(f"order must be at least 1; got {order!r}")
        return sum(
            p * max(threshold - r, 0.0) ** order
            for p, r in zip(self.probabilities, self.returns, strict=True)
        )

    @property
    def downside_deviation(self) -> float:
        return sqrt(self.lpm(order=2, threshold=0.0))

    @property
    def expected_loss_down(self) -> float:
        return self.lpm(order=1, threshold=0.0)

    @property
    def sharpe(self) -> float:
        return self.expected_return / self.stdev

    @property
    def sortino(self) -> float:
        deviation = self.downside_deviation
        if deviation == 0.0:
            raise ValueError("Sortino undefined when there is no downside state")
        return self.expected_return / deviation

    def utility(self, gamma: float) -> float:
        """Return ``E[R] - gamma Var(R) / 2`` for explicit risk aversion."""
        return self.expected_return - 0.5 * gamma * self.variance

    @property
    def breakeven_gamma(self) -> float:
        if self.expected_return <= 0.0:
            raise ValueError("break-even gamma is undefined for a non-positive expected return")
        if self.variance == 0.0:
            raise ValueError("break-even gamma is undefined for zero variance")
        return 2.0 * self.expected_return / self.variance

    def ev_interval(self, n_events: int, confidence: float) -> EVInterval:
        """Return an interval using the multinomial sampling floor only."""
        if n_events < 1:
            raise ValueError(f"n_events must be at least 1; got {n_events!r}")
        var_bull = self.p_bull * (1.0 - self.p_bull) / n_events
        var_bear = self.p_bear * (1.0 - self.p_bear) / n_events
        covariance = -self.p_bull * self.p_bear / n_events
        variance = (
            self.m_bull**2 * var_bull
            + self.m_bear**2 * var_bear
            - 2.0 * self.m_bull * self.m_bear * covariance
        )
        return EVInterval(
            point=self.expected_return,
            standard_error=sqrt(variance),
            n_events=n_events,
            confidence=confidence,
        )

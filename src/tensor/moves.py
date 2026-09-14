"""Moment matching between an option straddle and a three-state model.

The straddle supplies a risk-neutral mean absolute move.  The functions in
this module apply an explicitly supplied physical adjustment ``h`` and a
separately supplied *normal-reference* leakage probability.  They do not
estimate physical scenario probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from statistics import NormalDist

__all__ = [
    "Magnitudes",
    "asymmetric_magnitudes",
    "neutral_leakage",
    "straddle_to_mean_absolute_move",
    "symmetric_magnitude",
]

_NORMAL = NormalDist()
_PROBABILITY_TOLERANCE = 1e-9


def _check_unit(name: str, value: float, *, strict: bool = False) -> float:
    lower_ok = value > 0.0 if strict else value >= 0.0
    upper_ok = value < 1.0 if strict else value <= 1.0
    if not (lower_ok and upper_ok):
        bound = "(0, 1)" if strict else "[0, 1]"
        raise ValueError(f"{name} must lie in {bound}; got {value!r}")
    return value


def _check_positive(name: str, value: float) -> float:
    if not value > 0.0:
        raise ValueError(f"{name} must be positive; got {value!r}")
    return value


def _check_probabilities(p_bear: float, p_neutral: float, p_bull: float) -> None:
    _check_unit("p_bear", p_bear)
    _check_unit("p_neutral", p_neutral)
    _check_unit("p_bull", p_bull)
    total = p_bear + p_neutral + p_bull
    if abs(total - 1.0) > _PROBABILITY_TOLERANCE:
        raise ValueError(f"scenario probabilities must sum to 1; got {total!r}")


def straddle_to_mean_absolute_move(call_atm: float, put_atm: float, spot: float) -> float:
    """Return the straddle-implied risk-neutral mean absolute move.

    The ATM approximation is ``m_IV = (C_ATM + P_ATM) / P_0``.  It is a
    risk-neutral option moment and is not a physical probability estimate.
    """
    _check_positive("spot", spot)
    if call_atm < 0.0 or put_atm < 0.0:
        raise ValueError("option prices must be non-negative")
    return (call_atm + put_atm) / spot


def neutral_leakage(p_neutral_reference: float) -> float:
    """Return normal-reference absolute-move leakage for a neutral band.

    ``p_neutral_reference`` is the neutral probability under an explicitly
    chosen centered-normal reference distribution.  It is not a fitted or
    physical scenario probability.  If ``z = Phi^-1((1+p)/2)``, the fraction
    of ``E|R|`` inside the band is ``1 - exp(-z^2/2)``.
    """
    _check_unit("p_neutral_reference", p_neutral_reference)
    if p_neutral_reference == 0.0:
        return 0.0
    if p_neutral_reference == 1.0:
        return 1.0
    z = _NORMAL.inv_cdf((1.0 + p_neutral_reference) / 2.0)
    return 1.0 - exp(-0.5 * z * z)


def _tail_mean_absolute(m_iv: float, h: float, p_neutral_reference: float) -> float:
    _check_positive("m_iv", m_iv)
    _check_unit("h", h, strict=True)
    leakage = neutral_leakage(p_neutral_reference)
    return h * m_iv * (1.0 - leakage)


@dataclass(frozen=True)
class Magnitudes:
    """Calibrated positive magnitudes for bearish and bullish states."""

    m_bear: float
    m_bull: float
    m_iv: float
    h: float
    p_neutral_reference: float
    leakage: float
    tail_mean_absolute: float

    @property
    def is_symmetric(self) -> bool:
        return self.m_bear == self.m_bull

    @property
    def skew_ratio(self) -> float:
        """Return ``rho = m_bear / m_bull``."""
        return self.m_bear / self.m_bull


def symmetric_magnitude(
    m_iv: float,
    p_bear: float,
    p_neutral: float,
    p_bull: float,
    *,
    h: float,
    p_neutral_reference: float,
) -> Magnitudes:
    """Moment-match equal tail magnitudes to explicit scenario inputs.

    ``p_neutral`` is the physical scenario probability.  The separate
    ``p_neutral_reference`` controls only the centered-normal leakage
    correction.  Both are required so the two concepts cannot be conflated.
    """
    _check_probabilities(p_bear, p_neutral, p_bull)
    tail_mass = p_bear + p_bull
    if tail_mass <= 0.0:
        raise ValueError("p_bear + p_bull must be positive to identify a magnitude")

    tail_abs = _tail_mean_absolute(m_iv, h, p_neutral_reference)
    magnitude = tail_abs / tail_mass
    return Magnitudes(
        m_bear=magnitude,
        m_bull=magnitude,
        m_iv=m_iv,
        h=h,
        p_neutral_reference=p_neutral_reference,
        leakage=neutral_leakage(p_neutral_reference),
        tail_mean_absolute=tail_abs,
    )


def asymmetric_magnitudes(
    m_iv: float,
    p_bear: float,
    p_neutral: float,
    p_bull: float,
    rho: float,
    *,
    h: float,
    p_neutral_reference: float,
) -> Magnitudes:
    """Moment-match asymmetric tails using an explicit ``rho``.

    ``rho = m_bear / m_bull`` is an unidentified asymmetry parameter.  The
    straddle moment and the explicit scenario probabilities determine the
    scale only after ``rho`` is supplied.
    """
    _check_positive("rho", rho)
    _check_probabilities(p_bear, p_neutral, p_bull)
    denominator = rho * p_bear + p_bull
    if denominator <= 0.0:
        raise ValueError("rho * p_bear + p_bull must be positive")

    tail_abs = _tail_mean_absolute(m_iv, h, p_neutral_reference)
    m_bull = tail_abs / denominator
    return Magnitudes(
        m_bear=rho * m_bull,
        m_bull=m_bull,
        m_iv=m_iv,
        h=h,
        p_neutral_reference=p_neutral_reference,
        leakage=neutral_leakage(p_neutral_reference),
        tail_mean_absolute=tail_abs,
    )

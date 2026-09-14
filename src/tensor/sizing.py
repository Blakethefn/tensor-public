"""Closed-form sizing identities for an explicit three-state distribution.

These are mathematical quantities only.  They do not choose a position,
recommend an investment, or account for costs and model uncertainty beyond the
explicit shrinkage inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log

from .ev import ThreeState, tilt_standard_error

__all__ = [
    "SizingResult",
    "growth_rate",
    "kelly_fraction",
    "kelly_fraction_approx",
    "size_position",
    "uncertainty_shrinkage",
]


def kelly_fraction(state: ThreeState) -> float:
    """Return the exact growth-optimal fraction for the known distribution."""
    expected = state.expected_return
    if expected <= 0.0:
        return 0.0
    denominator = state.m_bear * state.m_bull * (state.p_bear + state.p_bull)
    return expected / denominator


def kelly_fraction_approx(state: ThreeState) -> float:
    """Return the small-return approximation to the exact fraction."""
    expected = state.expected_return
    if expected <= 0.0:
        return 0.0
    tilt = state.tilt
    variance_scale = (state.p_bear + state.p_bull) - tilt * tilt
    return tilt / (state.m_bull * variance_scale)


def growth_rate(state: ThreeState, fraction: float) -> float:
    """Return expected log growth, rejecting any state with certain ruin."""
    if any(1.0 + fraction * return_value <= 0.0 for return_value in state.returns):
        raise ValueError("fraction causes ruin in at least one state")
    return sum(
        p * log(1.0 + fraction * return_value)
        for p, return_value in zip(state.probabilities, state.returns, strict=True)
    )


def uncertainty_shrinkage(tilt: float, tilt_se: float) -> float:
    """Return the signal-to-total-uncertainty shrinkage factor."""
    if tilt_se < 0.0:
        raise ValueError(f"tilt_se must be non-negative; got {tilt_se!r}")
    denominator = tilt * tilt + tilt_se * tilt_se
    if denominator == 0.0:
        return 0.0
    return tilt * tilt / denominator


@dataclass(frozen=True)
class SizingResult:
    """Mathematical sizing outputs for one explicit event distribution."""

    full_kelly: float
    shrinkage: float
    recommended: float
    n_events: int
    kelly_fraction_used: float
    tilt_se: float
    bear_state_loss: float

    def __str__(self) -> str:
        return (
            f"recommended f = {100 * self.recommended:.1f}% "
            f"(full Kelly {100 * self.full_kelly:.1f}%, shrinkage {self.shrinkage:.3f})"
        )


def size_position(
    state: ThreeState,
    n_events: int,
    *,
    kelly_fraction_used: float,
) -> SizingResult:
    """Combine exact Kelly with explicit sampling uncertainty and scaling."""
    if n_events < 1:
        raise ValueError(f"n_events must be at least 1; got {n_events!r}")
    if not 0.0 <= kelly_fraction_used <= 1.0:
        raise ValueError(
            f"kelly_fraction_used must lie in [0, 1]; got {kelly_fraction_used!r}"
        )
    full = kelly_fraction(state)
    tilt_se = tilt_standard_error(state.p_bear, state.p_bull, n_events)
    shrinkage = uncertainty_shrinkage(state.tilt, tilt_se)
    recommended = full * shrinkage * kelly_fraction_used
    return SizingResult(
        full_kelly=full,
        shrinkage=shrinkage,
        recommended=recommended,
        n_events=n_events,
        kelly_fraction_used=kelly_fraction_used,
        tilt_se=tilt_se,
        bear_state_loss=recommended * state.m_bear,
    )

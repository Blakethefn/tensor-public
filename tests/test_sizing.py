"""Hypothetical mathematical tests for sizing identities."""

import pytest

from tensor import (
    ThreeState,
    growth_rate,
    kelly_fraction,
    kelly_fraction_approx,
    size_position,
    uncertainty_shrinkage,
)

P_BEAR, P_NEUTRAL, P_BULL = 0.22, 0.42, 0.36
M_BEAR, M_BULL = 0.16, 0.12


def state() -> ThreeState:
    return ThreeState(P_BEAR, P_NEUTRAL, P_BULL, M_BEAR, M_BULL)


def test_exact_kelly_is_zero_for_non_positive_edge() -> None:
    symmetric = ThreeState(0.29, 0.42, 0.29, 0.14, 0.14)
    assert kelly_fraction(symmetric) == pytest.approx(0.0)
    assert kelly_fraction_approx(symmetric) == pytest.approx(0.0)


def test_exact_kelly_is_not_the_small_return_approximation() -> None:
    exact = kelly_fraction(state())
    approximation = kelly_fraction_approx(state())
    assert exact > 0.0
    assert approximation > exact


def test_exact_fraction_maximizes_hypothetical_log_growth_locally() -> None:
    current = state()
    optimum = kelly_fraction(current)
    best = growth_rate(current, optimum)
    for offset in (-0.05, -0.01, 0.01, 0.05):
        assert growth_rate(current, optimum + offset) < best


def test_growth_rate_rejects_certain_ruin() -> None:
    with pytest.raises(ValueError, match="ruin"):
        growth_rate(state(), 1.0 / M_BEAR)


def test_uncertainty_shrinkage_is_monotone_and_bounded() -> None:
    values = [uncertainty_shrinkage(0.10, se) for se in (0.03, 0.07, 0.12, 0.18)]
    assert values == sorted(values, reverse=True)
    assert all(0.0 <= value <= 1.0 for value in values)
    assert uncertainty_shrinkage(0.10, 0.0) == pytest.approx(1.0)
    assert uncertainty_shrinkage(0.0, 0.10) == pytest.approx(0.0)


def test_size_position_requires_explicit_fraction_and_reports_bear_loss() -> None:
    result = size_position(state(), 40, kelly_fraction_used=0.20)
    assert result.recommended == pytest.approx(
        result.full_kelly * result.shrinkage * result.kelly_fraction_used
    )
    assert result.bear_state_loss == pytest.approx(result.recommended * M_BEAR)
    assert "recommended f" in str(result)


def test_sizing_validates_explicit_fraction_and_uncertainty() -> None:
    with pytest.raises(TypeError):
        size_position(state(), 40)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="kelly_fraction_used"):
        size_position(state(), 40, kelly_fraction_used=1.2)
    with pytest.raises(ValueError, match="tilt_se"):
        uncertainty_shrinkage(0.10, -0.01)

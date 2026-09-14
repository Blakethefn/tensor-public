"""Hypothetical mathematical tests for expected value and uncertainty."""

from math import sqrt

import pytest

from tensor import ThreeState, asymmetric_magnitudes, symmetric_magnitude, tilt_standard_error

M_IV = 0.12
HYPOTHETICAL_H = 0.73
P_BEAR, P_NEUTRAL, P_BULL = 0.29, 0.42, 0.29
P_NEUTRAL_REFERENCE = 0.31


def state_at(magnitude: float) -> ThreeState:
    return ThreeState(P_BEAR, P_NEUTRAL, P_BULL, magnitude, magnitude)


def calibrated_state() -> ThreeState:
    magnitudes = symmetric_magnitude(
        M_IV,
        P_BEAR,
        P_NEUTRAL,
        P_BULL,
        h=HYPOTHETICAL_H,
        p_neutral_reference=P_NEUTRAL_REFERENCE,
    )
    return ThreeState.from_magnitudes(P_BEAR, P_NEUTRAL, P_BULL, magnitudes)


def test_three_state_validates_probabilities_and_magnitudes() -> None:
    with pytest.raises(ValueError, match="sum"):
        ThreeState(0.2, 0.2, 0.2, 0.1, 0.1)
    with pytest.raises(ValueError, match="m_bull"):
        ThreeState(P_BEAR, P_NEUTRAL, P_BULL, 0.1, 0.0)


def test_expected_value_and_absolute_move_are_explicit_moments() -> None:
    state = state_at(0.18)
    assert state.expected_return == pytest.approx(0.0)
    assert state.mean_absolute_move == pytest.approx((P_BEAR + P_BULL) * 0.18)
    assert state.tilt == pytest.approx(0.0)


def test_symmetric_expected_value_is_invariant_to_neutral_probability() -> None:
    values = []
    for p_neutral in (0.20, 0.42, 0.55):
        tail = 1.0 - p_neutral
        p_bear, p_bull = tail * 0.45, tail * 0.55
        magnitude = symmetric_magnitude(
            M_IV,
            p_bear,
            p_neutral,
            p_bull,
            h=HYPOTHETICAL_H,
            p_neutral_reference=0.0,
        )
        values.append(ThreeState.from_magnitudes(p_bear, p_neutral, p_bull, magnitude).expected_return)
    assert values[0] == pytest.approx(values[1])
    assert values[1] == pytest.approx(values[2])


def test_asymmetry_changes_the_sign_condition() -> None:
    rho = P_BULL / P_BEAR
    magnitude = asymmetric_magnitudes(
        M_IV,
        P_BEAR,
        P_NEUTRAL,
        P_BULL,
        rho,
        h=HYPOTHETICAL_H,
        p_neutral_reference=0.0,
    )
    state = ThreeState.from_magnitudes(P_BEAR, P_NEUTRAL, P_BULL, magnitude)
    assert state.expected_return == pytest.approx(0.0, abs=1e-12)


def test_risk_metrics_use_explicit_threshold_and_risk_aversion() -> None:
    state = calibrated_state()
    assert state.lpm(order=1, threshold=0.0) == pytest.approx(state.expected_loss_down)
    assert state.downside_deviation == pytest.approx(sqrt(state.lpm(order=2, threshold=0.0)))
    assert state.utility(1.7) == pytest.approx(state.expected_return - (0.5 * 1.7) * state.variance)


def test_sampling_floor_and_interval_require_event_count_and_confidence() -> None:
    state = calibrated_state()
    floor = tilt_standard_error(P_BEAR, P_BULL, 40)
    assert floor == pytest.approx(
        sqrt((P_BULL * (1.0 - P_BULL) + P_BEAR * (1.0 - P_BEAR) + 2.0 * P_BULL * P_BEAR) / 40)
    )
    interval = state.ev_interval(40, confidence=0.90)
    assert interval.standard_error >= 0.0
    assert interval.contains_zero
    assert "confidence" in str(interval)


def test_interval_rejects_invalid_confidence_and_event_count() -> None:
    with pytest.raises(ValueError, match="confidence"):
        calibrated_state().ev_interval(20, confidence=1.0)
    with pytest.raises(ValueError, match="n_events"):
        tilt_standard_error(P_BEAR, P_BULL, 0)


def test_downside_and_significance_properties_are_defined() -> None:
    state = state_at(0.18)
    assert state.sortino == pytest.approx(0.0)
    assert state.sharpe == pytest.approx(0.0)
    with pytest.raises(ValueError, match="non-positive"):
        _ = state.breakeven_gamma

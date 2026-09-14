"""Hypothetical mathematical fixtures for straddle-to-magnitude identities."""

from math import exp
from statistics import NormalDist

import pytest

from tensor import asymmetric_magnitudes, neutral_leakage, straddle_to_mean_absolute_move
from tensor.moves import symmetric_magnitude

M_IV = 0.12
HYPOTHETICAL_H = 0.73
P_BEAR, P_NEUTRAL, P_BULL = 0.29, 0.42, 0.29
P_NEUTRAL_REFERENCE = 0.31


def test_straddle_is_a_risk_neutral_mean_absolute_move() -> None:
    assert straddle_to_mean_absolute_move(7.0, 5.0, 100.0) == pytest.approx(0.12)


def test_straddle_rejects_invalid_prices_and_spot() -> None:
    with pytest.raises(ValueError, match="spot"):
        straddle_to_mean_absolute_move(1.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        straddle_to_mean_absolute_move(-1.0, 1.0, 100.0)


def test_normal_reference_leakage_matches_its_closed_form() -> None:
    z = 0.4
    p_reference = 2.0 * NormalDist().cdf(z) - 1.0
    expected = 1.0 - exp(-0.5 * z * z)
    assert neutral_leakage(p_reference) == pytest.approx(expected)


def test_normal_reference_leakage_is_monotone() -> None:
    values = [neutral_leakage(p) for p in (0.0, 0.2, 0.31, 0.6, 1.0)]
    assert values == sorted(values)


def test_symmetric_magnitude_requires_explicit_physical_and_reference_inputs() -> None:
    magnitude = symmetric_magnitude(
        M_IV,
        P_BEAR,
        P_NEUTRAL,
        P_BULL,
        h=HYPOTHETICAL_H,
        p_neutral_reference=P_NEUTRAL_REFERENCE,
    )
    expected_tail = HYPOTHETICAL_H * M_IV * (1.0 - neutral_leakage(P_NEUTRAL_REFERENCE))
    assert magnitude.m_bear == pytest.approx(magnitude.m_bull)
    assert magnitude.tail_mean_absolute == pytest.approx(expected_tail)
    assert P_BEAR * magnitude.m_bear + P_BULL * magnitude.m_bull == pytest.approx(expected_tail)


def test_magnitude_does_not_infer_scenario_probabilities() -> None:
    with pytest.raises(TypeError):
        symmetric_magnitude(M_IV, P_BEAR, P_NEUTRAL, P_BULL)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="sum"):
        symmetric_magnitude(
            M_IV,
            P_BEAR,
            0.40,
            P_BULL,
            h=HYPOTHETICAL_H,
            p_neutral_reference=P_NEUTRAL_REFERENCE,
        )


def test_asymmetric_magnitude_honours_explicit_rho() -> None:
    rho = 1.25
    magnitude = asymmetric_magnitudes(
        M_IV,
        P_BEAR,
        P_NEUTRAL,
        P_BULL,
        rho,
        h=HYPOTHETICAL_H,
        p_neutral_reference=P_NEUTRAL_REFERENCE,
    )
    assert magnitude.skew_ratio == pytest.approx(rho)
    assert magnitude.m_bear > magnitude.m_bull


def test_moves_reject_invalid_physical_adjustment_and_rho() -> None:
    with pytest.raises(ValueError, match="h"):
        symmetric_magnitude(
            M_IV,
            P_BEAR,
            P_NEUTRAL,
            P_BULL,
            h=0.0,
            p_neutral_reference=P_NEUTRAL_REFERENCE,
        )
    with pytest.raises(ValueError, match="rho"):
        asymmetric_magnitudes(
            M_IV,
            P_BEAR,
            P_NEUTRAL,
            P_BULL,
            0.0,
            h=HYPOTHETICAL_H,
            p_neutral_reference=P_NEUTRAL_REFERENCE,
        )

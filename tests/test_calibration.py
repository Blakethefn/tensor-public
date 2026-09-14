"""Hypothetical tests for public calibration metrics and reference mapping."""

import pytest

from tensor.calibration import (
    brier_score,
    ece,
    log_loss,
    normal_reference_probabilities,
    score_all,
)


def test_perfect_and_worst_brier_scores() -> None:
    y = [0, 1, 2]
    assert brier_score(y, [1, 0, 0], [0, 1, 0], [0, 0, 1]) == pytest.approx(0.0)
    assert brier_score([0, 0], [0, 0], [1, 1], [0, 0]) == pytest.approx(2.0)


def test_log_loss_prefers_honest_predictions() -> None:
    y = [0, 1, 2, 0, 1, 2]
    good = log_loss(
        y,
        [0.8, 0.1, 0.1, 0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1, 0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8, 0.1, 0.1, 0.8],
    )
    weak = log_loss(y, [1 / 3] * 6, [1 / 3] * 6, [1 / 3] * 6)
    assert good < weak


def test_classwise_ece_accepts_explicit_bins() -> None:
    y = [0, 1, 2] * 20
    perfect = ([1.0, 0.0, 0.0] * 20, [0.0, 1.0, 0.0] * 20, [0.0, 0.0, 1.0] * 20)
    assert ece(y, *perfect, n_bins=5) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="n_bins"):
        ece(y, *perfect, n_bins=0)


def test_normal_reference_is_not_a_physical_probability_estimate() -> None:
    reference = normal_reference_probabilities(0.12, h=0.73, neutral_band_multiplier=0.4)
    assert sum(reference) == pytest.approx(1.0)
    assert reference[0] == pytest.approx(reference[2])
    assert reference[1] > 0.0
    assert normal_reference_probabilities(0.12, h=0.41, neutral_band_multiplier=0.4) == pytest.approx(reference)


def test_scores_can_be_compared_to_an_explicit_reference() -> None:
    y = [0, 1, 2, 0, 1, 2]
    result = score_all(
        y,
        [0.8, 0.1, 0.1, 0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1, 0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8, 0.1, 0.1, 0.8],
        n_bins=5,
        reference_probabilities=(0.3, 0.4, 0.3),
    )
    assert result.reference_brier is not None
    assert result.brier < result.reference_brier
    assert result.beats_reference_logloss


def test_scoring_validates_probability_rows() -> None:
    with pytest.raises(ValueError, match="sum"):
        brier_score([0], [0.5], [0.5], [0.5])
    assert brier_score([], [], [], []) == 0.0

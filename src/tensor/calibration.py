"""Public scoring rules and a labelled normal-reference probability map."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import erf, pi, sqrt

import numpy as np

__all__ = [
    "CalibrationResult",
    "brier_score",
    "ece",
    "log_loss",
    "normal_reference_probabilities",
    "score_all",
]

_EPS = 1e-15
_PROBABILITY_TOLERANCE = 1e-9


def _probability_arrays(
    y_true: Sequence[int],
    p_bear: Sequence[float],
    p_neutral: Sequence[float],
    p_bull: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.int64)
    probabilities = np.column_stack(
        [
            np.asarray(p_bear, dtype=np.float64),
            np.asarray(p_neutral, dtype=np.float64),
            np.asarray(p_bull, dtype=np.float64),
        ]
    )
    if probabilities.shape[0] != len(y):
        raise ValueError("outcomes and probability arrays must have equal length")
    if np.any((y < 0) | (y > 2)):
        raise ValueError("outcomes must be labelled 0, 1, or 2")
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    totals = probabilities.sum(axis=1)
    if np.any(np.abs(totals - 1.0) > _PROBABILITY_TOLERANCE):
        raise ValueError("each probability row must sum to 1")
    return y, probabilities[:, 0], probabilities[:, 1], probabilities[:, 2]


def brier_score(
    y_true: Sequence[int],
    p_bear: Sequence[float],
    p_neutral: Sequence[float],
    p_bull: Sequence[float],
) -> float:
    """Return the multiclass Brier score; lower is better."""
    y, pb, pn, pu = _probability_arrays(y_true, p_bear, p_neutral, p_bull)
    if len(y) == 0:
        return 0.0
    score = 0.0
    for i, p_vec in enumerate((pb, pn, pu)):
        score += np.mean((p_vec - (y == i)) ** 2)
    return float(score)


def log_loss(
    y_true: Sequence[int],
    p_bear: Sequence[float],
    p_neutral: Sequence[float],
    p_bull: Sequence[float],
) -> float:
    """Return multiclass negative log likelihood; lower is better."""
    y, pb, pn, pu = _probability_arrays(y_true, p_bear, p_neutral, p_bull)
    if len(y) == 0:
        return 0.0
    probability_matrix = np.clip(np.column_stack((pb, pn, pu)), _EPS, 1.0)
    chosen = probability_matrix[np.arange(len(y)), y]
    return float(-np.mean(np.log(chosen)))


def ece(
    y_true: Sequence[int],
    p_bear: Sequence[float],
    p_neutral: Sequence[float],
    p_bull: Sequence[float],
    *,
    n_bins: int,
) -> float:
    """Return classwise expected calibration error for explicit bins."""
    y, pb, pn, pu = _probability_arrays(y_true, p_bear, p_neutral, p_bull)
    if n_bins < 1:
        raise ValueError(f"n_bins must be positive; got {n_bins!r}")
    if len(y) == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i, p_vec in enumerate((pb, pn, pu)):
        indicator = (y == i).astype(np.float64)
        class_error = 0.0
        for j in range(n_bins):
            upper_inclusive = j == n_bins - 1
            mask = (p_vec >= bin_edges[j]) & (
                p_vec <= bin_edges[j + 1] if upper_inclusive else p_vec < bin_edges[j + 1]
            )
            count = int(mask.sum())
            if count == 0:
                continue
            class_error += (count / len(y)) * abs(
                float(p_vec[mask].mean()) - float(indicator[mask].mean())
            )
        total += class_error
    return total / 3.0


def normal_reference_probabilities(
    m_iv: float,
    h: float,
    neutral_band_multiplier: float,
) -> tuple[float, float, float]:
    """Map an explicit normal reference to labelled reference probabilities.

    The result is ``(p_B_ref, p_N_ref, p_U_ref)`` under a centered normal
    reference.  It is a distributional benchmark, not a physical forecast and
    must not be substituted for probabilities from a fitted scenario model.
    ``h`` converts the risk-neutral straddle moment to an explicitly supplied
    physical scale; ``neutral_band_multiplier`` is the explicitly supplied
    threshold in units of that scale.
    """
    if not m_iv > 0.0:
        raise ValueError(f"m_iv must be positive; got {m_iv!r}")
    if not h > 0.0:
        raise ValueError(f"h must be positive; got {h!r}")
    if neutral_band_multiplier < 0.0:
        raise ValueError(
            "neutral_band_multiplier must be non-negative; "
            f"got {neutral_band_multiplier!r}"
        )

    physical_move = h * m_iv
    sigma = physical_move * sqrt(pi / 2.0)
    threshold = neutral_band_multiplier * physical_move
    p_neutral = erf(threshold / (sigma * sqrt(2.0)))
    p_tail = (1.0 - p_neutral) / 2.0
    return (p_tail, p_neutral, p_tail)


@dataclass(frozen=True)
class CalibrationResult:
    """Scoring metrics, optionally paired with an explicit reference."""

    brier: float
    logloss: float
    ece: float
    n_observations: int
    reference_brier: float | None = None
    reference_logloss: float | None = None
    reference_ece: float | None = None

    @property
    def beats_reference_brier(self) -> bool:
        return self.reference_brier is not None and self.brier < self.reference_brier

    @property
    def beats_reference_logloss(self) -> bool:
        return self.reference_logloss is not None and self.logloss < self.reference_logloss

    @property
    def beats_reference_ece(self) -> bool:
        return self.reference_ece is not None and self.ece < self.reference_ece

    def __str__(self) -> str:
        parts = [
            f"Brier={self.brier:.4f}",
            f"LogLoss={self.logloss:.4f}",
            f"ECE={self.ece:.4f}",
            f"n={self.n_observations}",
        ]
        if self.reference_brier is not None:
            parts.append(
                f"reference Brier={self.reference_brier:.4f} "
                f"({'yes' if self.beats_reference_brier else 'no'})"
            )
        return " ".join(parts)


def score_all(
    y_true: Sequence[int],
    p_bear: Sequence[float],
    p_neutral: Sequence[float],
    p_bull: Sequence[float],
    *,
    n_bins: int,
    reference_probabilities: tuple[float, float, float] | None = None,
) -> CalibrationResult:
    """Compute scoring rules against an optional explicit reference vector."""
    brier = brier_score(y_true, p_bear, p_neutral, p_bull)
    loss = log_loss(y_true, p_bear, p_neutral, p_bull)
    calibration = ece(y_true, p_bear, p_neutral, p_bull, n_bins=n_bins)

    reference_scores: tuple[float | None, float | None, float | None]
    if reference_probabilities is None:
        reference_scores = (None, None, None)
    else:
        y, _, _, _ = _probability_arrays(y_true, p_bear, p_neutral, p_bull)
        reference = tuple(reference_probabilities)
        if len(reference) != 3:
            raise ValueError("reference_probabilities must contain three values")
        if any(value < 0.0 for value in reference) or abs(sum(reference) - 1.0) > _PROBABILITY_TOLERANCE:
            raise ValueError("reference_probabilities must be non-negative and sum to 1")
        n = len(y)
        reference_scores = (
            brier_score(y, [reference[0]] * n, [reference[1]] * n, [reference[2]] * n),
            log_loss(y, [reference[0]] * n, [reference[1]] * n, [reference[2]] * n),
            ece(
                y,
                [reference[0]] * n,
                [reference[1]] * n,
                [reference[2]] * n,
                n_bins=n_bins,
            ),
        )

    return CalibrationResult(
        brier=brier,
        logloss=loss,
        ece=calibration,
        n_observations=len(y_true),
        reference_brier=reference_scores[0],
        reference_logloss=reference_scores[1],
        reference_ece=reference_scores[2],
    )

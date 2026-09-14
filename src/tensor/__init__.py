"""Math-only public companion for the Tensor theoretical specification.

All model-specific quantities are caller-supplied.  In particular, the public
package does not contain default physical adjustments, scenario probabilities,
thresholds, valuation uncertainty, ``rho``, or ``kappa`` values.
"""

from .calibration import (
    CalibrationResult,
    brier_score,
    ece,
    log_loss,
    normal_reference_probabilities,
    score_all,
)
from .ev import EVInterval, ThreeState, tilt_standard_error
from .kappa import KappaEstimate, estimate_kappa
from .moves import (
    Magnitudes,
    asymmetric_magnitudes,
    neutral_leakage,
    straddle_to_mean_absolute_move,
    symmetric_magnitude,
)
from .scenario import (
    OrderedLogitParams,
    ScenarioProbabilities,
    estimate_ordered_logit,
    predict_probabilities,
)
from .sizing import (
    SizingResult,
    growth_rate,
    kelly_fraction,
    kelly_fraction_approx,
    size_position,
    uncertainty_shrinkage,
)

__version__ = "0.1.0"

__all__ = [
    "CalibrationResult",
    "EVInterval",
    "KappaEstimate",
    "Magnitudes",
    "OrderedLogitParams",
    "ScenarioProbabilities",
    "SizingResult",
    "ThreeState",
    "asymmetric_magnitudes",
    "brier_score",
    "ece",
    "estimate_kappa",
    "estimate_ordered_logit",
    "growth_rate",
    "kelly_fraction",
    "kelly_fraction_approx",
    "log_loss",
    "neutral_leakage",
    "normal_reference_probabilities",
    "predict_probabilities",
    "score_all",
    "size_position",
    "straddle_to_mean_absolute_move",
    "symmetric_magnitude",
    "tilt_standard_error",
    "uncertainty_shrinkage",
]

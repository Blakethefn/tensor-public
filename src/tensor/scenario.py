"""Heteroskedastic ordered logit for earnings-event probabilities.

Implements the ordered-logit specification in ``docs/main.tex``:

.. math::

    y^{\\ast} = \\mathbf{z}^{\\top}\\boldsymbol\\beta + s(\\mathbf{w})\\,\\varepsilon,
    \\qquad \\varepsilon \\sim \\text{Logistic}(0,1)

    s(\\mathbf{w}) = \\exp(\\delta_1 Z_{IV} + \\delta_2 Z_{MOM}^2)

    p_B = \\Lambda\\!\\left(\\frac{c_1 - \\mathbf{z}^{\\top}\\boldsymbol\\beta}{s(\\mathbf{w})}\\right) \\\\
    p_N = \\Lambda\\!\\left(\\frac{c_2 - \\mathbf{z}^{\\top}\\boldsymbol\\beta}{s(\\mathbf{w})}\\right)
          - \\Lambda\\!\\left(\\frac{c_1 - \\mathbf{z}^{\\top}\\boldsymbol\\beta}{s(\\mathbf{w})}\\right) \\\\
    p_U = 1 - \\Lambda\\!\\left(\\frac{c_2 - \\mathbf{z}^{\\top}\\boldsymbol\\beta}{s(\\mathbf{w})}\\right)

The model is identified without further constraints: the cut points absorb the
location, and the scale is pinned by the logistic variance :math:`\\pi^2/3`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit as logistic

__all__ = [
    "OrderedLogitParams",
    "ScenarioProbabilities",
    "estimate_ordered_logit",
    "predict_probabilities",
]

_EPS = np.finfo(np.float64).eps


# -- public types ------------------------------------------------------------


@dataclass(frozen=True)
class OrderedLogitParams:
    """Fitted parameters of the heteroskedastic ordered logit.

    Attributes prefixed ``se_`` are delta-method standard errors.  ``n_obs`` is
    the number of events the model was fitted on; a small single-name panel
    carries substantially more uncertainty than the standard errors alone
    convey, and the caller must apply explicitly chosen shrinkage.
    """

    beta: np.ndarray
    """Location coefficients.  ``z @ beta`` is the latent bullishness index."""

    c1: float
    """Lower cut point.  ``c1 < c2`` by construction."""

    c2: float
    """Upper cut point."""

    delta: np.ndarray
    """Scale coefficients for the heteroskedasticity term.  ``exp(w @ delta)``
    multiplies the logistic error, so larger values flatten the distribution
    by moving mass from the neutral state into both tails."""

    feature_names: tuple[str, ...]
    """Names of the location features (``z``), in the order ``beta`` expects."""

    scale_feature_names: tuple[str, ...]
    """Names of the scale features (``w``), in the order ``delta`` expects."""

    se_beta: np.ndarray | None = None
    se_c1: float | None = None
    se_c2: float | None = None
    se_delta: np.ndarray | None = None

    cov: np.ndarray | None = None
    """Full covariance of ``[beta, c1, c2, delta]``.

    The ``se_`` attributes are its square-rooted diagonal.  The delta method
    of the specification is a quadratic form in the *whole* matrix -- cut
    points and slopes are strongly correlated in an ordered logit, so the
    off-diagonal blocks are not a refinement -- which is why the matrix is
    carried rather than only its diagonal.
    """

    n_obs: int = 0
    log_likelihood: float = 0.0
    regularization: float = 0.0

    converged: bool = True


@dataclass(frozen=True)
class ScenarioProbabilities:
    """Estimated probabilities for one event, with standard errors."""

    p_bear: float
    p_neutral: float
    p_bull: float
    tilt: float

    tilt_se: float | None
    """Delta-method standard error of the tilt, or ``None`` when unavailable.

    ``None`` -- not ``0.0`` -- when the covariance could not be estimated or
    was not requested.  A zero here would read as a perfectly measured tilt and
    would drive :func:`~tensor.sizing.uncertainty_shrinkage` to 1, sizing the
    position as if the estimate carried no error at all.  The specification
    makes this standard error a required output; absent is honest, zero is not.
    """

    feature_vector: np.ndarray = field(repr=False)
    scale_vector: np.ndarray = field(repr=False)


# -- likelihood and gradient -------------------------------------------------


def _logistic(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic, clamped away from 0 and 1."""
    return np.clip(logistic(x), _EPS, 1.0 - _EPS)


def _neg_log_likelihood(
    theta: np.ndarray,
    Z: np.ndarray,
    W: np.ndarray,
    y: np.ndarray,
    k_beta: int,
    k_delta: int,
    penalty: float,
) -> float:
    """Negative log-likelihood of the heteroskedastic ordered logit.

    ``theta`` packs ``[beta (k_beta), c1, c2, delta (k_delta)]``.
    """
    beta = theta[:k_beta]
    c1 = theta[k_beta]
    c2 = theta[k_beta + 1]
    delta = theta[k_beta + 2 : k_beta + 2 + k_delta]

    if c1 >= c2:
        return 1e12

    idx = Z @ beta
    log_scale = W @ delta
    log_scale = np.clip(log_scale, -20.0, 20.0)
    scale = np.exp(log_scale)

    eta1 = (c1 - idx) / scale
    eta2 = (c2 - idx) / scale

    p1 = _logistic(eta1)  # P(Y <= B)
    p2 = _logistic(eta2)  # P(Y <= N)

    ll = 0.0
    for i in range(3):
        mask = y == i
        if not mask.any():
            continue
        if i == 0:
            ll += np.sum(np.log(p1[mask]))
        elif i == 1:
            ll += np.sum(np.log(np.maximum(p2[mask] - p1[mask], _EPS)))
        else:
            ll += np.sum(np.log(np.maximum(1.0 - p2[mask], _EPS)))

    n = len(y)
    reg = 0.0
    if penalty > 0.0:
        reg = penalty * (np.sum(beta**2) + np.sum(delta**2))
    return -(ll / n) + reg


def _fit_ordered_logit(
    Z: np.ndarray,
    W: np.ndarray,
    y: np.ndarray,
    *,
    penalty: float,
    max_iter: int = 2000,
) -> tuple[np.ndarray, float, bool]:
    """Fit the heteroskedastic ordered logit via L-BFGS-B.

    Returns ``(theta, neg_log_likelihood, converged)``.
    """
    n, k_beta = Z.shape
    k_delta = W.shape[1]

    # Initialise: beta = 0, cut points at logistic quantiles, delta = 0
    counts = np.bincount(y, minlength=3)
    cum_frac = np.cumsum(counts) / n
    c1_init = np.log(
        np.clip(cum_frac[0], _EPS, 1.0 - _EPS)
        / np.clip(1.0 - cum_frac[0], _EPS, 1.0 - _EPS)
    )
    c2_init = np.log(
        np.clip(cum_frac[1], _EPS, 1.0 - _EPS)
        / np.clip(1.0 - cum_frac[1], _EPS, 1.0 - _EPS)
    )

    theta0 = np.zeros(k_beta + 2 + k_delta)
    theta0[k_beta] = c1_init
    theta0[k_beta + 1] = c2_init

    result = minimize(
        _neg_log_likelihood,
        theta0,
        args=(Z, W, y, k_beta, k_delta, penalty),
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": 1e-10, "gtol": 1e-8},
    )
    return result.x, result.fun, result.success


# -- delta-method standard errors --------------------------------------------


def _hessian_numerical(
    theta: np.ndarray,
    Z: np.ndarray,
    W: np.ndarray,
    y: np.ndarray,
    k_beta: int,
    k_delta: int,
    penalty: float,
    eps: float = 1e-5,
) -> np.ndarray:
    """Numerical Hessian of the *negative* log-likelihood at ``theta``."""
    m = len(theta)
    H = np.zeros((m, m))
    f0 = _neg_log_likelihood(theta, Z, W, y, k_beta, k_delta, penalty)

    for i in range(m):
        ei = np.zeros(m)
        ei[i] = eps
        fp = _neg_log_likelihood(theta + ei, Z, W, y, k_beta, k_delta, penalty)
        fm = _neg_log_likelihood(theta - ei, Z, W, y, k_beta, k_delta, penalty)
        H[i, i] = (fp - 2.0 * f0 + fm) / (eps * eps)
        for j in range(i + 1, m):
            ej = np.zeros(m)
            ej[j] = eps
            fpp = _neg_log_likelihood(theta + ei + ej, Z, W, y, k_beta, k_delta, penalty)
            fpm = _neg_log_likelihood(theta + ei - ej, Z, W, y, k_beta, k_delta, penalty)
            fmp = _neg_log_likelihood(theta - ei + ej, Z, W, y, k_beta, k_delta, penalty)
            fmm = _neg_log_likelihood(theta - ei - ej, Z, W, y, k_beta, k_delta, penalty)
            H[i, j] = (fpp - fpm - fmp + fmm) / (4.0 * eps * eps)
            H[j, i] = H[i, j]

    return H


def _extract_params(
    theta: np.ndarray,
    Z: np.ndarray,
    W: np.ndarray,
    y: np.ndarray,
    k_beta: int,
    k_delta: int,
    penalty: float,
    converged: bool,
    neg_ll: float,
    feature_names: tuple[str, ...],
    scale_feature_names: tuple[str, ...],
) -> OrderedLogitParams:
    """Build an ``OrderedLogitParams`` from the raw solution vector."""
    n = len(y)
    beta = theta[:k_beta]
    c1 = theta[k_beta]
    c2 = theta[k_beta + 1]
    delta = theta[k_beta + 2 : k_beta + 2 + k_delta]

    se_beta = None
    se_c1 = None
    se_c2 = None
    se_delta = None
    cov = None

    try:
        # ``_neg_log_likelihood`` returns the *mean* penalised negative
        # log-likelihood, so its Hessian is 1/n of the total.  Inverting the
        # total gives the covariance, hence the division by n.
        H = _hessian_numerical(theta, Z, W, y, k_beta, k_delta, penalty)
        H = 0.5 * (H + H.T)
        eigvals = np.linalg.eigvalsh(H)
        if np.all(eigvals > 1e-10):
            candidate = np.linalg.inv(H) / n
            candidate = 0.5 * (candidate + candidate.T)
            diag = np.diag(candidate)
            if np.all(diag >= 0.0):
                cov = candidate
                se_beta = np.sqrt(np.maximum(diag[:k_beta], 0.0))
                se_c1 = np.sqrt(max(diag[k_beta], 0.0))
                se_c2 = np.sqrt(max(diag[k_beta + 1], 0.0))
                se_delta = np.sqrt(np.maximum(diag[k_beta + 2 : k_beta + 2 + k_delta], 0.0))
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        # A singular or indefinite Hessian means that coefficient covariance
        # is unavailable; predictions remain usable but expose ``tilt_se`` as
        # None rather than fabricating zero uncertainty.
        cov = None

    return OrderedLogitParams(
        beta=beta.copy(),
        c1=c1,
        c2=c2,
        delta=delta.copy(),
        feature_names=feature_names,
        scale_feature_names=scale_feature_names,
        se_beta=se_beta,
        se_c1=se_c1,
        se_c2=se_c2,
        se_delta=se_delta,
        cov=cov,
        n_obs=n,
        log_likelihood=-neg_ll * n,
        regularization=penalty,
        converged=converged,
    )


# -- public API --------------------------------------------------------------


def estimate_ordered_logit(
    Z: np.ndarray,
    W: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
    scale_feature_names: Sequence[str] | None = None,
    penalty: float,
) -> OrderedLogitParams:
    """Fit the heteroskedastic ordered logit.

    Args:
        Z: (n, k_beta) feature matrix for location.
        W: (n, k_delta) feature matrix for scale.
        y: (n,) integer labels in {0, 1, 2} for {Bear, Neutral, Bull}.
        feature_names: Human-readable names for the columns of ``Z``.
        scale_feature_names: Human-readable names for the columns of ``W``.
        penalty: Explicit L2 regularisation strength on ``beta`` and ``delta``
            only (cut points are never penalised).  The theory does not choose
            this value.

    Returns:
        Fitted parameters with delta-method standard errors.
    """
    Z = np.asarray(Z, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    if Z.ndim != 2:
        raise ValueError(f"Z must be 2-D; got shape {Z.shape}")
    if W.ndim != 2:
        raise ValueError(f"W must be 2-D; got shape {W.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D; got shape {y.shape}")
    if len(Z) != len(y):
        raise ValueError(f"Z and y must have the same length; got {len(Z)} and {len(y)}")
    if len(W) != len(y):
        raise ValueError(f"W and y must have the same length; got {len(W)} and {len(y)}")
    if not np.all(np.isin(y, [0, 1, 2])):
        raise ValueError("y must contain only values in {0, 1, 2}")
    if len(y) < 10:
        raise ValueError(f"need at least 10 observations; got {len(y)}")
    if penalty < 0.0:
        raise ValueError(f"penalty must be non-negative; got {penalty!r}")

    k_beta = Z.shape[1]
    k_delta = W.shape[1]

    fn = feature_names or tuple(f"z{i}" for i in range(k_beta))
    if len(fn) != k_beta:
        raise ValueError(f"feature_names has {len(fn)} entries; expected {k_beta}")
    sfn = scale_feature_names or tuple(f"w{i}" for i in range(k_delta))
    if len(sfn) != k_delta:
        raise ValueError(f"scale_feature_names has {len(sfn)} entries; expected {k_delta}")

    theta, neg_ll, converged = _fit_ordered_logit(Z, W, y, penalty=penalty)
    return _extract_params(
        theta, Z, W, y, k_beta, k_delta, penalty, converged, neg_ll, tuple(fn), tuple(sfn),
    )


def predict_probabilities(
    params: OrderedLogitParams,
    Z: np.ndarray,
    W: np.ndarray,
    *,
    compute_se: bool = True,
) -> list[ScenarioProbabilities]:
    """Return ``(p_B, p_N, p_U)`` for each row of ``Z`` and ``W``.

    When ``compute_se`` is True, the delta-method standard error of the tilt is
    included.  This requires the parameter covariance to have been estimated
    during fitting; when it was not, ``tilt_se`` is ``None`` rather than zero.
    """
    Z = np.asarray(Z, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    if Z.ndim == 1:
        Z = Z.reshape(1, -1)
    if W.ndim == 1:
        W = W.reshape(1, -1)

    if Z.shape[1] != len(params.beta):
        raise ValueError(
            f"Z has {Z.shape[1]} columns; expected {len(params.beta)}"
        )
    if W.shape[1] != len(params.delta):
        raise ValueError(
            f"W has {W.shape[1]} columns; expected {len(params.delta)}"
        )

    idx = Z @ params.beta
    log_scale = W @ params.delta
    log_scale = np.clip(log_scale, -20.0, 20.0)
    scale = np.exp(log_scale)

    eta1 = (params.c1 - idx) / scale
    eta2 = (params.c2 - idx) / scale

    p1 = logistic(eta1)
    p2 = logistic(eta2)

    p_bear = p1
    p_neutral = p2 - p1
    p_bull = 1.0 - p2
    tilt = p_bull - p_bear

    tilt_ses: list[float] = []
    if compute_se and params.cov is not None:
        tilt_ses = _tilt_se_delta_method(
            params, Z, W, scale, eta1, eta2, p1, p2,
        )

    results: list[ScenarioProbabilities] = []
    for i in range(len(Z)):
        se = tilt_ses[i] if tilt_ses else None
        results.append(
            ScenarioProbabilities(
                p_bear=float(p_bear[i]),
                p_neutral=float(p_neutral[i]),
                p_bull=float(p_bull[i]),
                tilt=float(tilt[i]),
                tilt_se=se,
                feature_vector=Z[i].copy(),
                scale_vector=W[i].copy(),
            )
        )
    return results


def _tilt_se_delta_method(
    params: OrderedLogitParams,
    Z: np.ndarray,
    W: np.ndarray,
    scale: np.ndarray,
    eta1: np.ndarray,
    eta2: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> list[float]:
    """Delta-method standard error of ``p_U - p_B`` for each observation.

    The specification requires the quadratic form

    .. math::

        \\operatorname{se}(\\hat p_U-\\hat p_B)
        = \\sqrt{\\mathbf{g}^{\\top}\\widehat{\\operatorname{Cov}}
                 (\\hat{\\boldsymbol\\theta})\\,\\mathbf{g}},
        \\qquad
        \\mathbf{g}=\\nabla_{\\boldsymbol\\theta}\\,(p_U-p_B)

    over the whole covariance matrix.  Using only its diagonal drops the
    covariance between the cut points and the slopes, which in an ordered logit
    is large and of either sign, so the diagonal-only version is not a
    conservative approximation -- it is simply a different number.

    Writing :math:`\\eta_1=(c_1-\\mathbf{z}^{\\top}\\boldsymbol\\beta)/s`,
    :math:`\\eta_2=(c_2-\\mathbf{z}^{\\top}\\boldsymbol\\beta)/s`,
    :math:`\\lambda_k=\\Lambda(\\eta_k)\\big(1-\\Lambda(\\eta_k)\\big)` and
    :math:`s=\\exp(\\mathbf{w}^{\\top}\\boldsymbol\\delta)`, the tilt
    :math:`p_U-p_B=1-\\Lambda(\\eta_2)-\\Lambda(\\eta_1)` has gradient

    .. math::

        \\frac{\\partial}{\\partial\\boldsymbol\\beta}
            = \\frac{(\\lambda_1+\\lambda_2)\\,\\mathbf{z}}{s},
        \\qquad
        \\frac{\\partial}{\\partial c_1} = -\\frac{\\lambda_1}{s},
        \\qquad
        \\frac{\\partial}{\\partial c_2} = -\\frac{\\lambda_2}{s},
        \\qquad
        \\frac{\\partial}{\\partial\\boldsymbol\\delta}
            = (\\lambda_1\\eta_1+\\lambda_2\\eta_2)\\,\\mathbf{w}

    The two location terms *add*: raising the index moves mass out of ``B`` and
    into ``U`` simultaneously, so the two effects reinforce rather than cancel.
    The scale block carries no :math:`1/s` because
    :math:`\\partial\\eta_k/\\partial\\boldsymbol\\delta=-\\eta_k\\mathbf{w}`
    -- the ``s`` from differentiating the exponential cancels the ``s`` in the
    denominator of :math:`\\eta_k`.

    Returns an empty list when the covariance could not be estimated.
    """
    if params.cov is None:
        return []

    k_beta = len(params.beta)
    k_delta = len(params.delta)
    total = k_beta + 2 + k_delta

    cov = np.asarray(params.cov, dtype=np.float64)
    if cov.shape != (total, total):
        return []

    results: list[float] = []
    for i in range(len(Z)):
        grad = _tilt_gradient(
            Z[i], W[i], scale[i], eta1[i], eta2[i], p1[i], p2[i], k_beta, k_delta,
        )
        var_tilt = float(grad @ cov @ grad)
        results.append(float(np.sqrt(max(var_tilt, 0.0))))

    return results


def _tilt_gradient(
    z: np.ndarray,
    w: np.ndarray,
    scale: float,
    eta1: float,
    eta2: float,
    p1: float,
    p2: float,
    k_beta: int,
    k_delta: int,
) -> np.ndarray:
    """Gradient of ``p_U - p_B`` with respect to ``[beta, c1, c2, delta]``.

    See :func:`_tilt_se_delta_method` for the derivation.  Split out so the
    gradient can be checked against a numerical derivative independently of the
    covariance estimate.
    """
    lam1 = p1 * (1.0 - p1)
    lam2 = p2 * (1.0 - p2)

    grad = np.zeros(k_beta + 2 + k_delta)
    grad[:k_beta] = (lam1 + lam2) * z / scale
    grad[k_beta] = -lam1 / scale
    grad[k_beta + 1] = -lam2 / scale
    grad[k_beta + 2 : k_beta + 2 + k_delta] = (lam1 * eta1 + lam2 * eta2) * w
    return grad

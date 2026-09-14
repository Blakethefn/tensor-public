"""Hypothetical mathematical tests for the heteroskedastic ordered logit.

Verifies:
- Parameter estimation recovers known coefficients from synthetic data.
- Probability predictions sum to 1 and respect ordering.
- The delta-method standard errors are non-negative.
- Edge cases: single-class data, boundary probabilities, scale effects.
"""

import numpy as np
import pytest

from tensor.scenario import (
    OrderedLogitParams,
    estimate_ordered_logit,
    predict_probabilities,
)


def _synthetic_data(
    n: int = 200,
    beta: np.ndarray | None = None,
    *,
    c1: float,
    c2: float,
    delta: np.ndarray | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, OrderedLogitParams]:
    """Generate clearly hypothetical data from a known ordered logit."""
    rng = np.random.default_rng(seed)
    k_beta = 3
    k_delta = 2

    if beta is None:
        beta = np.array([0.5, -0.3, 0.8])
    if delta is None:
        delta = np.array([0.2, -0.1])

    Z = rng.normal(0, 1, (n, k_beta))
    W = rng.normal(0, 1, (n, k_delta))

    idx = Z @ beta
    log_scale = W @ delta
    scale = np.exp(np.clip(log_scale, -5, 5))

    eps = rng.logistic(0, 1, n)
    y_star = idx + scale * eps

    y = np.zeros(n, dtype=np.int64)
    y[y_star > c1] = 1
    y[y_star > c2] = 2

    params = OrderedLogitParams(
        beta=beta,
        c1=c1,
        c2=c2,
        delta=delta,
        feature_names=("z1", "z2", "z3"),
        scale_feature_names=("w1", "w2"),
        n_obs=n,
    )
    return Z, W, y, params


class TestEstimation:
    """The fitted model recovers known parameters from synthetic data."""

    def test_recovers_direction(self) -> None:
        """Coefficients have the correct sign and reasonable magnitude."""
        Z, W, y, true = _synthetic_data(n=500, c1=-0.5, c2=0.5)
        fitted = estimate_ordered_logit(Z, W, y, penalty=0.001)

        assert fitted.converged
        # Signs should match
        for i in range(len(true.beta)):
            assert np.sign(fitted.beta[i]) == np.sign(true.beta[i]), (
                f"beta[{i}] sign mismatch: fitted={fitted.beta[i]:.3f}, true={true.beta[i]:.3f}"
            )

        # Cut points should be ordered
        assert fitted.c1 < fitted.c2

    def test_scale_effect_matters(self) -> None:
        """When scale features are active, the homoskedastic model fits worse."""
        Z, W, y, _ = _synthetic_data(n=300, c1=-0.5, c2=0.5, delta=np.array([0.5, 0.3]))

        # Heteroskedastic fit
        het = estimate_ordered_logit(Z, W, y, penalty=0.001)

        # Homoskedastic fit (W = zeros)
        W_zero = np.zeros_like(W)
        homo = estimate_ordered_logit(Z, W_zero, y, penalty=0.001)

        # Heteroskedastic should have better log-likelihood
        assert het.log_likelihood > homo.log_likelihood, (
            f"heteroskedastic LL={het.log_likelihood:.3f} <= "
            f"homoskedastic LL={homo.log_likelihood:.3f}"
        )

    def test_converges_on_small_sample(self) -> None:
        """The model should converge even with limited data."""
        Z, W, y, _ = _synthetic_data(n=40, c1=-0.5, c2=0.5)
        fitted = estimate_ordered_logit(Z, W, y, penalty=0.05)
        assert fitted.converged
        assert fitted.c1 < fitted.c2

    def test_standard_errors_are_non_negative(self) -> None:
        """Delta-method SEs should be non-negative when the Hessian is invertible."""
        Z, W, y, _ = _synthetic_data(n=500, c1=-0.5, c2=0.5)
        fitted = estimate_ordered_logit(Z, W, y, penalty=0.001)

        if fitted.se_beta is not None:
            assert np.all(fitted.se_beta >= 0.0)
            assert fitted.se_c1 >= 0.0
            assert fitted.se_c2 >= 0.0
            assert np.all(fitted.se_delta >= 0.0)


class TestPredictions:
    """Probability predictions are valid probability distributions."""

    def test_probabilities_sum_to_one(self) -> None:
        Z, W, y, _ = _synthetic_data(n=200, c1=-0.5, c2=0.5)
        fitted = estimate_ordered_logit(Z, W, y, penalty=0.01)
        probs = predict_probabilities(fitted, Z, W)

        for p in probs:
            total = p.p_bear + p.p_neutral + p.p_bull
            assert total == pytest.approx(1.0, abs=1e-10), f"sum = {total}"

    def test_probabilities_are_in_unit_interval(self) -> None:
        Z, W, y, _ = _synthetic_data(n=200, c1=-0.5, c2=0.5)
        fitted = estimate_ordered_logit(Z, W, y, penalty=0.01)
        probs = predict_probabilities(fitted, Z, W)

        for p in probs:
            assert 0.0 <= p.p_bear <= 1.0
            assert 0.0 <= p.p_neutral <= 1.0
            assert 0.0 <= p.p_bull <= 1.0

    def test_tilt_is_difference(self) -> None:
        Z, W, y, _ = _synthetic_data(n=200, c1=-0.5, c2=0.5)
        fitted = estimate_ordered_logit(Z, W, y, penalty=0.01)
        probs = predict_probabilities(fitted, Z, W)

        for p in probs:
            assert p.tilt == pytest.approx(p.p_bull - p.p_bear)

    def test_scale_flattens_distribution(self) -> None:
        """Larger scale (= larger IV) should push mass from neutral to tails."""
        Z, W, y, _ = _synthetic_data(n=300, c1=-0.5, c2=0.5, delta=np.array([0.5, 0.0]))
        fitted = estimate_ordered_logit(Z, W, y, penalty=0.001)

        # Predict at mean Z, vary W's first component
        Z_mean = Z.mean(axis=0, keepdims=True)
        W_low = np.array([[0.0, 0.0]])
        W_high = np.array([[2.0, 0.0]])

        probs_low = predict_probabilities(fitted, Z_mean, W_low)[0]
        probs_high = predict_probabilities(fitted, Z_mean, W_high)[0]

        # Higher scale → more mass in tails, less in neutral
        assert probs_high.p_neutral < probs_low.p_neutral, (
            f"scale should flatten: neutral={probs_high.p_neutral:.3f} vs {probs_low.p_neutral:.3f}"
        )

    def test_tilt_gradient_matches_numerical_derivative(self) -> None:
        """The delta-method gradient must be the true gradient of the tilt.

        The specification requires se(p_U - p_B) = sqrt(g' Cov g) with
        g = grad(p_U - p_B).  If g is wrong the interval is wrong in a way no
        smoke test catches, so it is checked against a central difference.
        """
        from tensor.scenario import _tilt_gradient

        rng = np.random.default_rng(7)
        k_beta, k_delta = 3, 2
        theta = np.concatenate(
            [rng.normal(size=k_beta) * 0.5, [-0.8, 0.6], rng.normal(size=k_delta) * 0.3]
        )
        z = rng.normal(size=k_beta)
        w = rng.normal(size=k_delta)

        def tilt_at(t: np.ndarray) -> float:
            beta = t[:k_beta]
            c1, c2 = t[k_beta], t[k_beta + 1]
            delta = t[k_beta + 2 :]
            scale = np.exp(w @ delta)
            e1 = (c1 - z @ beta) / scale
            e2 = (c2 - z @ beta) / scale
            lam = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731
            return float((1.0 - lam(e2)) - lam(e1))

        step = 1e-6
        numerical = np.zeros(len(theta))
        for i in range(len(theta)):
            bump = np.zeros(len(theta))
            bump[i] = step
            numerical[i] = (tilt_at(theta + bump) - tilt_at(theta - bump)) / (2 * step)

        beta = theta[:k_beta]
        c1, c2 = theta[k_beta], theta[k_beta + 1]
        delta = theta[k_beta + 2 :]
        scale = float(np.exp(w @ delta))
        eta1 = (c1 - z @ beta) / scale
        eta2 = (c2 - z @ beta) / scale
        p1 = 1.0 / (1.0 + np.exp(-eta1))
        p2 = 1.0 / (1.0 + np.exp(-eta2))

        analytic = _tilt_gradient(z, w, scale, eta1, eta2, p1, p2, k_beta, k_delta)
        assert analytic == pytest.approx(numerical, abs=1e-7)

    def test_tilt_se_uses_the_off_diagonal_covariance(self) -> None:
        """se(tilt) is a quadratic form, not a sum of squared marginal terms.

        The cut points and the slopes are strongly correlated in an ordered
        logit, so dropping the off-diagonal blocks gives a different number.
        """
        Z, W, y, _ = _synthetic_data(n=400, c1=-0.5, c2=0.5)
        fitted = estimate_ordered_logit(Z, W, y, penalty=0.001)
        assert fitted.cov is not None

        probs = predict_probabilities(fitted, Z[:20], W[:20], compute_se=True)

        from tensor.scenario import _tilt_gradient

        k_beta, k_delta = len(fitted.beta), len(fitted.delta)
        diag_only = np.diag(np.diag(fitted.cov))
        differs = False
        for i, p in enumerate(probs):
            scale = float(np.exp(W[i] @ fitted.delta))
            eta1 = (fitted.c1 - Z[i] @ fitted.beta) / scale
            eta2 = (fitted.c2 - Z[i] @ fitted.beta) / scale
            p1 = 1.0 / (1.0 + np.exp(-eta1))
            p2 = 1.0 / (1.0 + np.exp(-eta2))
            g = _tilt_gradient(Z[i], W[i], scale, eta1, eta2, p1, p2, k_beta, k_delta)

            assert p.tilt_se == pytest.approx(float(np.sqrt(g @ fitted.cov @ g)))
            if abs(p.tilt_se - float(np.sqrt(g @ diag_only @ g))) > 1e-6:
                differs = True
        assert differs, "off-diagonal covariance had no effect — check the fit"

    def test_tilt_se_is_non_negative(self) -> None:
        Z, W, y, _ = _synthetic_data(n=500, c1=-0.5, c2=0.5)
        fitted = estimate_ordered_logit(Z, W, y, penalty=0.001)
        probs = predict_probabilities(fitted, Z, W, compute_se=True)

        for p in probs:
            assert p.tilt_se is not None
            assert p.tilt_se >= 0.0

    def test_tilt_se_is_none_when_not_requested(self) -> None:
        """An absent standard error is None, never zero.

        Zero would read as a perfectly measured tilt and would drive the
        Kelly shrinkage factor to 1, sizing the position as if the estimate
        carried no error at all.
        """
        Z, W, y, _ = _synthetic_data(n=200, c1=-0.5, c2=0.5)
        fitted = estimate_ordered_logit(Z, W, y, penalty=0.01)
        probs = predict_probabilities(fitted, Z, W, compute_se=False)

        for p in probs:
            assert p.tilt_se is None


class TestInputValidation:
    def test_rejects_1d_Z(self) -> None:
        Z = np.array([1.0, 2.0, 3.0])
        W = np.array([[0.0], [0.0], [0.0]])
        y = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="2-D"):
            estimate_ordered_logit(Z, W, y, penalty=0.01)

    def test_rejects_mismatched_lengths(self) -> None:
        Z = np.array([[1.0], [2.0]])
        W = np.array([[0.0], [0.0], [0.0]])
        y = np.array([0, 1])
        with pytest.raises(ValueError, match="same length"):
            estimate_ordered_logit(Z, W, y, penalty=0.01)

    def test_rejects_invalid_labels(self) -> None:
        Z = np.random.normal(0, 1, (20, 2))
        W = np.random.normal(0, 1, (20, 1))
        y = np.array([0, 1, 3, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1])
        with pytest.raises(ValueError, match="0, 1, 2"):
            estimate_ordered_logit(Z, W, y, penalty=0.01)

    def test_rejects_too_few_observations(self) -> None:
        Z = np.random.normal(0, 1, (5, 2))
        W = np.random.normal(0, 1, (5, 1))
        y = np.array([0, 1, 2, 0, 1])
        with pytest.raises(ValueError, match="at least 10"):
            estimate_ordered_logit(Z, W, y, penalty=0.01)

    def test_rejects_wrong_feature_name_count(self) -> None:
        Z, W, y, _ = _synthetic_data(n=50, c1=-0.5, c2=0.5)
        with pytest.raises(ValueError, match="feature_names"):
            estimate_ordered_logit(Z, W, y, penalty=0.01, feature_names=["a", "b"])


class TestPredictInputValidation:
    def test_predict_rejects_wrong_Z_columns(self) -> None:
        Z, W, y, _ = _synthetic_data(n=50, c1=-0.5, c2=0.5)
        fitted = estimate_ordered_logit(Z, W, y, penalty=0.01)
        Z_bad = np.random.normal(0, 1, (10, 5))  # wrong number of columns
        with pytest.raises(ValueError, match="columns"):
            predict_probabilities(fitted, Z_bad, W[:10])

    def test_predict_accepts_1d_input(self) -> None:
        Z, W, y, _ = _synthetic_data(n=50, c1=-0.5, c2=0.5)
        fitted = estimate_ordered_logit(Z, W, y, penalty=0.01)
        probs = predict_probabilities(fitted, Z[0], W[0])
        assert len(probs) == 1


class TestCutPointOrdering:
    def test_cut_points_are_ordered(self) -> None:
        Z, W, y, _ = _synthetic_data(n=500, c1=-0.5, c2=0.5)
        fitted = estimate_ordered_logit(Z, W, y, penalty=0.01)
        assert fitted.c1 < fitted.c2

    def test_penalty_does_not_affect_cut_points(self) -> None:
        """L2 penalty applies to slopes only, not cut points."""
        Z, W, y, _ = _synthetic_data(n=200, c1=-0.5, c2=0.5)
        unreg = estimate_ordered_logit(Z, W, y, penalty=0.0)
        reg = estimate_ordered_logit(Z, W, y, penalty=1.0)

        # Cut points should be similar regardless of penalty
        assert abs(unreg.c1 - reg.c1) < 1.0
        assert abs(unreg.c2 - reg.c2) < 1.0


class TestFeatureNames:
    def test_default_feature_names(self) -> None:
        Z, W, y, _ = _synthetic_data(n=50, c1=-0.5, c2=0.5)
        fitted = estimate_ordered_logit(Z, W, y, penalty=0.01)
        assert fitted.feature_names == ("z0", "z1", "z2")
        assert fitted.scale_feature_names == ("w0", "w1")

    def test_custom_feature_names(self) -> None:
        Z, W, y, _ = _synthetic_data(n=50, c1=-0.5, c2=0.5)
        fitted = estimate_ordered_logit(
            Z, W, y,
            penalty=0.01,
            feature_names=["eps", "rev", "guide"],
            scale_feature_names=["iv", "mom2"],
        )
        assert fitted.feature_names == ("eps", "rev", "guide")
        assert fitted.scale_feature_names == ("iv", "mom2")

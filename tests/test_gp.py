import numpy as np
import pytest

from src.gp import GP
from src.kernels import RBFKernel
from src.linalg import ReorthogonalizationRule
from src.means import ZeroMean


def make_data():
    X_train = np.linspace(-1.0, 1.0, 8).reshape(-1, 1)
    y_train = np.sin(X_train).reshape(-1)
    X_test = np.linspace(-0.5, 0.5, 5).reshape(-1, 1)

    return X_train, y_train, X_test


def test_exact_posterior_mean_and_covariance_shapes():
    X_train, y_train, X_test = make_data()

    gp = GP(
        X_train,
        y_train,
        kernel=RBFKernel(lengthscale=0.4),
        mean=ZeroMean(),
        noise=1e-2,
    )

    gp.compute_posterior(method="exact")

    mean = gp.predict_mean(X_test)
    covariance = gp.predict_covariance(X_test)

    assert mean.shape == (X_test.shape[0],)
    assert covariance.shape == (X_test.shape[0], X_test.shape[0])


def test_exact_covariance_matches_manual_formula():
    X_train, y_train, X_test = make_data()

    kernel = RBFKernel(lengthscale=0.4)
    mean = ZeroMean()

    gp = GP(X_train, y_train, kernel=kernel, mean=mean, noise=1e-2)
    gp.compute_posterior(method="exact")

    K_test = kernel(X_test, X_test)
    k = kernel(X_train, X_test)
    expected = K_test - k.T @ np.linalg.solve(gp.K_noise, k)

    assert np.allclose(gp.predict_covariance(X_test), expected)


def test_cg_posterior_runs_and_returns_variance():
    X_train, y_train, X_test = make_data()

    gp = GP(
        X_train,
        y_train,
        kernel=RBFKernel(lengthscale=0.4),
        mean=ZeroMean(),
        noise=1e-2,
    )

    gp.compute_posterior(
        method="cg",
        cg_J=8,
        cg_reorthogonalization_rule=ReorthogonalizationRule(mode="always"),
    )

    variance = gp.predict_variance(X_test, cg_correction_method="cholesky")

    assert variance.shape == (X_test.shape[0],)


def test_love_posterior_runs_and_returns_variance():
    X_train, y_train, X_test = make_data()

    gp = GP(
        X_train,
        y_train,
        kernel=RBFKernel(lengthscale=0.4),
        mean=ZeroMean(),
        noise=1e-2,
    )

    gp.compute_posterior(
        method="love",
        cg_J=8,
        lanczos_J=8,
        lanczos_reorthogonalize=True,
    )

    variance = gp.predict_variance(X_test)

    assert variance.shape == (X_test.shape[0],)


def test_predict_raises_before_compute_posterior():
    X_train, y_train, X_test = make_data()

    gp = GP(
        X_train,
        y_train,
        kernel=RBFKernel(lengthscale=0.4),
        mean=ZeroMean(),
        noise=1e-2,
    )

    with pytest.raises(RuntimeError):
        gp.predict(X_test)
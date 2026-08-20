import numpy as np
import pytest

from src.kernels import LinearKernel, MaternKernel, RBFKernel


@pytest.fixture
def X():
    return np.linspace(-1.0, 1.0, 100).reshape(-1, 1)


@pytest.mark.parametrize(
    "kernel",
    [
        RBFKernel(lengthscale=0.4, outputscale=1.2),
        MaternKernel(lengthscale=0.4, outputscale=1.2, nu=0.5),
        MaternKernel(lengthscale=0.4, outputscale=1.2, nu=1.5),
        MaternKernel(lengthscale=0.4, outputscale=1.2, nu=2.5),
        LinearKernel(outputscale=1.2, offset=0.0),
    ],
)
def test_kernel_matrix_is_square_symmetric_and_psd(kernel, X):
    K = kernel(X, X)
    eigvals = np.linalg.eigvalsh(K)

    assert K.shape == (X.shape[0], X.shape[0])
    assert np.allclose(K, K.T)
    assert eigvals[0] > -1e-10


@pytest.mark.parametrize(
    "kernel",
    [
        RBFKernel(lengthscale=0.4, outputscale=1.2),
        MaternKernel(lengthscale=0.4, outputscale=1.2, nu=0.5),
        MaternKernel(lengthscale=0.4, outputscale=1.2, nu=1.5),
        MaternKernel(lengthscale=0.4, outputscale=1.2, nu=2.5),
    ],
)
def test_stationary_kernel_diagonal_equals_outputscale_squared(kernel, X):
    K = kernel(X, X)

    expected_diag = kernel.outputscale**2 * np.ones(X.shape[0])

    assert np.allclose(np.diag(K), expected_diag)


def test_linear_kernel_matches_manual_formula(X):
    kernel = LinearKernel(outputscale=1.2, offset=0.5)

    K = kernel(X, X)

    X_centered = X - 0.5
    expected = 1.2**2 * (X_centered @ X_centered.T)

    assert np.allclose(K, expected)


@pytest.mark.parametrize("bad_lengthscale", [0.0, -1.0])
def test_rbf_rejects_nonpositive_lengthscale(bad_lengthscale):
    with pytest.raises(ValueError):
        RBFKernel(lengthscale=bad_lengthscale)


@pytest.mark.parametrize("bad_nu", [0.0, 1.0, 3.5])
def test_matern_rejects_unsupported_nu(bad_nu):
    with pytest.raises(ValueError):
        MaternKernel(nu=bad_nu)


@pytest.mark.parametrize(
    "kernel",
    [
        RBFKernel(lengthscale=0.4),
        MaternKernel(lengthscale=0.4),
        LinearKernel(),
    ],
)
def test_one_dimensional_input_is_accepted(kernel):
    X = np.linspace(-1.0, 1.0, 6)

    K = kernel(X, X)

    assert K.shape == (6, 6)
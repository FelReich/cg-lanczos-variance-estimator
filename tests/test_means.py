import numpy as np
import pytest

from src.means import LinearMean, TrigExpMean, ZeroMean


@pytest.fixture
def X():
    return np.linspace(-1.0, 1.0, 6).reshape(-1, 1)


def test_zero_mean_returns_zero_vector(X):
    mean = ZeroMean()

    values = mean(X)

    assert values.shape == (X.shape[0],)
    assert np.allclose(values, 0.0)


def test_linear_mean_one_dimensional(X):
    mean = LinearMean(weights=2.0, bias=1.0)

    values = mean(X)

    expected = 2.0 * X.squeeze(-1) + 1.0
    assert values.shape == (X.shape[0],)
    assert np.allclose(values, expected)


def test_linear_mean_multi_dimensional():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    mean = LinearMean(weights=np.array([2.0, -1.0]), bias=0.5)

    values = mean(X)

    expected = X @ np.array([2.0, -1.0]) + 0.5
    assert values.shape == (2,)
    assert np.allclose(values, expected)


def test_linear_mean_rejects_wrong_weight_shape():
    X = np.ones((3, 2))
    mean = LinearMean(weights=np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError):
        mean(X)


def test_trig_exp_mean_matches_manual_formula(X):
    mean = TrigExpMean(
        offset=1.0,
        sine_scale=2.0,
        sine_frequency=3.0,
        cosine_scale=4.0,
        cosine_frequency=5.0,
        exp_scale=6.0,
        exp_rate=7.0,
    )

    x = X.squeeze(-1)

    expected = (
        1.0
        + 2.0 * np.sin(3.0 * x)
        + 4.0 * np.cos(5.0 * x)
        + 6.0 * np.exp(-7.0 * x**2)
    )

    values = mean(X)

    assert values.shape == (X.shape[0],)
    assert np.allclose(values, expected)


def test_trig_exp_mean_rejects_multi_dimensional_input():
    X = np.ones((3, 2))
    mean = TrigExpMean()

    with pytest.raises(ValueError):
        mean(X)
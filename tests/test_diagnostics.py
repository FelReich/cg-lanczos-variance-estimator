import numpy as np
import pytest

from src.diagnostics import conjugacy_error, projection_residual, relative_error


def test_relative_error_is_zero_for_equal_arrays():
    x = np.array([1.0, 2.0, 3.0])

    assert relative_error(x, x) == 0.0


def test_relative_error_matches_manual_formula():
    approx = np.array([1.0, 2.0, 4.0])
    exact = np.array([1.0, 2.0, 3.0])

    expected = np.linalg.norm(approx - exact) / np.linalg.norm(exact)

    assert np.isclose(relative_error(approx, exact), expected)


def test_relative_error_handles_zero_exact_array():
    approx = np.array([1.0, 2.0])
    exact = np.zeros(2)

    assert np.isclose(relative_error(approx, exact), np.linalg.norm(approx))


def test_conjugacy_error_zero_for_diagonal_projected_gram():
    D = np.eye(3)
    KD = np.diag([1.0, 2.0, 3.0])

    assert np.isclose(conjugacy_error(D, KD), 0.0)


def test_conjugacy_error_positive_for_nonconjugate_directions():
    D = np.eye(2)
    KD = np.array(
        [
            [1.0, 0.5],
            [0.5, 2.0],
        ]
    )

    assert conjugacy_error(D, KD) > 0.0


def test_conjugacy_error_rejects_shape_mismatch():
    D = np.ones((3, 2))
    KD = np.ones((4, 2))

    with pytest.raises(ValueError):
        conjugacy_error(D, KD)


def test_projection_residual_zero_if_k_lies_in_span_of_D():
    D = np.eye(3)[:, :2]
    k = np.array([1.0, 2.0, 0.0])

    assert np.isclose(projection_residual(D, k), 0.0)


def test_projection_residual_positive_if_k_outside_span_of_D():
    D = np.eye(3)[:, :2]
    k = np.array([1.0, 2.0, 3.0])

    assert projection_residual(D, k) > 0.0


def test_projection_residual_accepts_matrix_k():
    D = np.eye(3)[:, :2]
    k = np.array(
        [
            [1.0, 0.0],
            [2.0, 1.0],
            [0.0, 0.0],
        ]
    )

    assert np.isclose(projection_residual(D, k), 0.0)


def test_projection_residual_rejects_dimension_mismatch():
    D = np.ones((3, 2))
    k = np.ones(4)

    with pytest.raises(ValueError):
        projection_residual(D, k)
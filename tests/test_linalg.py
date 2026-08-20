import numpy as np
import pytest

from src.linalg import ReorthogonalizationRule, cg, lanczos_tridiagonalization


def make_spd_matrix(n: int, seed: int = 0, jitter: float = 1e-6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n))
    return M.T @ M + jitter * np.eye(n)


def test_cg_matches_numpy_solve():
    A = make_spd_matrix(20, seed=1)
    b = np.random.default_rng(2).normal(size=20)

    x = cg(lambda v: A @ v, b, J=100, tol=1e-10)

    expected = np.linalg.solve(A, b)

    assert np.allclose(x, expected, atol=1e-8, rtol=1e-8)


def test_cg_residual_is_small():
    A = make_spd_matrix(20, seed=3)
    b = np.random.default_rng(4).normal(size=20)

    x = cg(lambda v: A @ v, b, J=100, tol=1e-10)

    residual = b - A @ x

    assert np.linalg.norm(residual) / np.linalg.norm(b) < 1e-8


def test_cg_saved_directions_have_correct_shapes():
    A = make_spd_matrix(20, seed=5)
    b = np.random.default_rng(6).normal(size=20)

    x, D, KD = cg(
        lambda v: A @ v,
        b,
        J=10,
        tol=0.0,
        save_directions=True,
    )

    assert x.shape == (20,)
    assert D.shape == (20, 10)
    assert KD.shape == (20, 10)
    assert np.allclose(KD, A @ D)


def test_cg_saved_directions_are_a_conjugate():
    A = make_spd_matrix(20, seed=7)
    b = np.random.default_rng(8).normal(size=20)

    _, D, KD = cg(
        lambda v: A @ v,
        b,
        J=10,
        tol=0.0,
        save_directions=True,
    )

    G = D.T @ KD
    off_diag = G - np.diag(np.diag(G))

    assert np.linalg.norm(off_diag) / np.linalg.norm(np.diag(np.diag(G))) < 1e-8


def test_cg_reorthogonalization_rule_forces_saved_directions():
    A = make_spd_matrix(20, seed=13)
    b = np.random.default_rng(14).normal(size=20)

    x, D, KD = cg(
        lambda v: A @ v,
        b,
        J=10,
        tol=0.0,
        save_directions=False,
        reorthogonalization_rule=ReorthogonalizationRule(mode="always"),
    )

    assert x.shape == (20,)
    assert D.shape == (20, 10)
    assert KD.shape == (20, 10)
    assert np.allclose(KD, A @ D)


def test_cg_reorthogonalization_rule_every():
    A = make_spd_matrix(20, seed=15)
    b = np.random.default_rng(16).normal(size=20)

    x, D, KD = cg(
        lambda v: A @ v,
        b,
        J=10,
        tol=0.0,
        save_directions=True,
        reorthogonalization_rule=ReorthogonalizationRule(mode="every", every=3),
    )

    assert x.shape == (20,)
    assert D.shape == (20, 10)
    assert KD.shape == (20, 10)
    assert np.allclose(KD, A @ D)


def test_reorthogonalization_rule_rejects_invalid_mode():
    with pytest.raises(ValueError):
        ReorthogonalizationRule(mode="sometimes")


def test_lanczos_returns_orthonormal_basis_and_projected_matrix():
    A = make_spd_matrix(30, seed=9)
    b = np.random.default_rng(10).normal(size=30)

    Q, T = lanczos_tridiagonalization(
        lambda v: A @ v,
        b,
        num_iter=10,
        tol=1e-12,
        reorthogonalize=True,
    )

    assert Q.shape == (30, 10)
    assert T.shape == (10, 10)
    assert np.allclose(Q.T @ Q, np.eye(10), atol=1e-10)

    projected = Q.T @ A @ Q

    assert np.allclose(T, projected, atol=1e-8)


def test_lanczos_rejects_zero_initial_vector():
    A = make_spd_matrix(10, seed=11)
    b = np.zeros(10)

    with pytest.raises(ValueError):
        lanczos_tridiagonalization(lambda v: A @ v, b, num_iter=5)


def test_cg_rejects_nonpositive_iteration_count():
    A = make_spd_matrix(10, seed=12)
    b = np.ones(10)

    with pytest.raises(ValueError):
        cg(lambda v: A @ v, b, J=0)
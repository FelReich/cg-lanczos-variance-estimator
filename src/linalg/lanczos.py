from __future__ import annotations

from collections.abc import Callable

import numpy as np


def transform_T(T: np.ndarray) -> np.ndarray:
    """Converts compact tridiagonal storage into a dense matrix.

    The first column contains the diagonal entries and the second column contains
    the upper/lower off-diagonal entries.
    """
    T = np.asarray(T, dtype=float)

    if T.ndim != 2 or T.shape[1] != 2:
        raise ValueError("T must have shape (n, 2).")
    
    return np.diag(T[:, 0]) + np.diag(T[:-1, 1], k=1) + np.diag(T[:-1, 1], k=-1)


def lanczos_tridiagonalization(
    matmul: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    num_iter: int,
    tol: float = 1e-6,
    reorthogonalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes a Lanczos basis and the associated tridiagonal matrix.

    :param matmul: Function implementing multiplication by the symmetric matrix.
    :param numpy.ndarray b: Initial vector for the Krylov subspace.
    :param int num_iter: Maximum number of Lanczos iterations.
    :param float tol: Breakdown tolerance for stopping early. (Default: `1e-6`.)
    :param bool reorthogonalize: If True, use two reorthogonalization passes. (Default: True.)
    :return: Lanczos basis `Q` and dense tridiagonal matrix `T`.
    """
    b = np.asarray(b, dtype=float).reshape(-1)

    if num_iter <= 0:
        raise ValueError("num_iter must be positive.")

    n = b.size
    Q = np.zeros((n, num_iter))
    T = np.zeros((num_iter, 2))

    b_norm = np.linalg.norm(b)
    if b_norm == 0:
        raise ValueError("Initial vector b must be nonzero.")

    q = b / b_norm
    q_prev = np.zeros_like(q)
    beta_prev = 0.0

    for j in range(num_iter):
        Q[:, j] = q

        v = np.asarray(matmul(q), dtype=float).reshape(-1)

        if v.shape != q.shape:
            raise ValueError("matmul must return a vector with the same shape as b.")

        if j > 0:
            v -= beta_prev * q_prev

        alpha = np.dot(q, v)
        T[j, 0] = alpha

        v -= alpha * q

        if reorthogonalize:
            for i in range(j + 1):
                coeff = np.dot(Q[:, i], v)
                v -= coeff * Q[:, i]

            for i in range(j + 1):
                coeff = np.dot(Q[:, i], v)
                v -= coeff * Q[:, i]

        beta = np.linalg.norm(v)

        if beta < tol:
            return Q[:, : j + 1], transform_T(T[: j + 1, :])

        if j == num_iter - 1:
            break

        T[j, 1] = beta

        q_prev = q
        q = v / beta
        beta_prev = beta

    return Q, transform_T(T)


def extend_lanczos_basis_qkq(
    matmul: Callable[[np.ndarray], np.ndarray],
    Q: np.ndarray,
    KQ: np.ndarray,
    target_J: int,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Extends an existing Lanczos basis and returns its projected matrix.

    The input basis is extended by continuing the Lanczos orthogonalization
    process from the last stored basis vector. Matrix-vector products are stored
    in `KQ`, so the projected matrix is formed as `Q.T @ KQ`.

    :param matmul: Function implementing multiplication by the symmetric matrix.
    :param numpy.ndarray Q: Existing basis of shape `n x J`.
    :param numpy.ndarray KQ: Matrix-vector products corresponding to `Q`.
    :param int target_J: Desired maximum final basis size.
    :param float tol: Breakdown tolerance for stopping early. (Default: `1e-12`.)
    :return: Extended basis `Q_ext` and projected matrix `T_ext`.
    """
    Q = np.asarray(Q, dtype=float)

    if Q.ndim != 2:
        raise ValueError("Q must be a two-dimensional array.")

    n, current_J = Q.shape

    if KQ.shape !=  Q.shape:
        raise ValueError("KQ must have the same shape as Q.")

    if target_J <= 0:
        raise ValueError("target_J must be positive.")

    if current_J == 0:
        raise ValueError("Q must contain at least one basis vector.")

    if target_J <= current_J:
        Q_final = Q[:, :target_J]
        KQ_final = KQ[:, :target_J]
        T_final = Q_final.T @ KQ_final
        return Q_final, 0.5 * (T_final + T_final.T)

    Q_ext = np.zeros((n, target_J))
    Q_ext[:, :current_J] = Q

    KQ_ext = np.zeros((n, target_J))
    KQ_ext[:, :current_J] = KQ

    q = Q_ext[:, current_J - 1].copy()
    v = np.asarray(matmul(q), dtype=float).reshape(-1)

    for _ in range(2):
        v = v - Q_ext[:, :current_J] @ (Q_ext[:, :current_J].T @ v)

    for j in range(current_J, target_J):
    

        if v.shape != q.shape:
            raise ValueError("matmul must return a vector with the same shape as the basis vectors.")

        beta = np.linalg.norm(v)

        if beta < tol:
            Q_final = Q_ext[:, :j]
            KQ_final = KQ_ext[:, :j]
            T_final = Q_final.T @ KQ_final
            return Q_final, 0.5 * (T_final + T_final.T)

        q = v / beta

        Q_ext[:, j] = q

        v = np.asarray(matmul(q), dtype=float).reshape(-1)

        KQ_ext[:, j] = v

        if v.shape != q.shape:
            raise ValueError("matmul must return a vector with the same shape as the basis vectors.")

        for _ in range(2):
            v = v - Q_ext[:, :j+1] @ (Q_ext[:, :j+1].T @ v)

    T_final = Q_ext.T @ KQ_ext
    return Q_ext, 0.5 * (T_final + T_final.T)


def extend_lanczos_basis(
    matmul: Callable[[np.ndarray], np.ndarray],
    Q: np.ndarray,
    T: np.ndarray,
    target_J: int,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Extends an existing Lanczos basis using only the basis and tridiagonal matrix.

    The extension is initialized from the last stored Lanczos vector and the
    recurrence coefficients contained in `T`. During the extension, only the
    basis vectors and the tridiagonal coefficients are stored; matrix-vector
    products `KQ` are not stored.

    :param matmul: Function implementing multiplication by the symmetric matrix.
    :param numpy.ndarray Q: Existing Lanczos basis of shape `n x J`.
    :param numpy.ndarray T: Existing projected tridiagonal matrix of shape `J x J`.
    :param int target_J: Desired maximum final basis size.
    :param float tol: Breakdown tolerance for stopping early. (Default: `1e-12`.)
    :return: Extended basis `Q_ext` and extended tridiagonal matrix `T_ext`.
    """
    Q = np.asarray(Q, dtype=float)
    T = np.asarray(T, dtype=float)

    if Q.ndim != 2:
        raise ValueError("Q must be a two-dimensional array.")

    n, current_J = Q.shape

    if current_J == 0:
        raise ValueError("Q must contain at least one basis vector.")

    if T.shape != (current_J, current_J):
        raise ValueError("T must have shape (current_J, current_J).")

    if target_J <= 0:
        raise ValueError("target_J must be positive.")

    if target_J <= current_J:
        Q_final = Q[:, :target_J]
        T_final = T[:target_J, :target_J]
        return Q_final, 0.5 * (T_final + T_final.T)

    Q_ext = np.zeros((n, target_J))
    Q_ext[:, :current_J] = Q

    T_ext = np.zeros((target_J, target_J))
    T_ext[:current_J, :current_J] = 0.5 * (T + T.T)

    q = Q_ext[:, current_J - 1].copy()

    v = np.asarray(matmul(q), dtype=float).reshape(-1)

    if v.shape != q.shape:
        raise ValueError("matmul must return a vector with the same shape as the basis vectors.")

    v -= T_ext[current_J - 1, current_J - 1] * q

    if current_J > 1:
        v -= T_ext[current_J - 2, current_J - 1] * Q_ext[:, current_J - 2]

    for _ in range(2):
        coeffs = Q_ext[:, :current_J].T @ v
        v -= Q_ext[:, :current_J] @ coeffs

    for j in range(current_J, target_J):
        beta = np.linalg.norm(v)

        if beta < tol:
            Q_final = Q_ext[:, :j]
            T_final = T_ext[:j, :j]
            return Q_final, 0.5 * (T_final + T_final.T)

        T_ext[j - 1, j] = beta
        T_ext[j, j - 1] = beta

        q_prev = q
        q = v / beta

        Q_ext[:, j] = q

        v = np.asarray(matmul(q), dtype=float).reshape(-1)

        if v.shape != q.shape:
            raise ValueError("matmul must return a vector with the same shape as the basis vectors.")

        alpha = np.dot(q, v)
        T_ext[j, j] = alpha

        v -= alpha * q
        v -= beta * q_prev

        for _ in range(2):
            coeffs = Q_ext[:, : j + 1].T @ v
            v -= Q_ext[:, : j + 1] @ coeffs

    return Q_ext, 0.5 * (T_ext + T_ext.T)
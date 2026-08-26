from __future__ import annotations

from collections.abc import Callable

import numpy as np


def transform_T(T: np.ndarray) -> np.ndarray:
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


def extend_lanczos_basis(
    matmul: Callable[[np.ndarray], np.ndarray],
    Q: np.ndarray,
    KQ: np.ndarray,
    target_J: int,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
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
            KQ_final = Q_ext[:, :j]
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
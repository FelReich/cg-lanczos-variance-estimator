from __future__ import annotations

from collections.abc import Callable

import numpy as np


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
    T = np.zeros((num_iter, num_iter))

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
        T[j, j] = alpha

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
            return Q[:, : j + 1], T[: j + 1, : j + 1]

        if j == num_iter - 1:
            break

        T[j, j + 1] = beta
        T[j + 1, j] = beta

        q_prev = q
        q = v / beta
        beta_prev = beta

    return Q, T
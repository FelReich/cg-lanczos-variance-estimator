from __future__ import annotations

from collections.abc import Callable

import numpy as np


def cg(
    matmul: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    J: int,
    tol: float = 1e-6,
    save_directions: bool = False,
    reorthogonalize: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = np.asarray(b, dtype=float).reshape(-1)

    if J <= 0:
        raise ValueError("J must be positive.")

    c = np.zeros_like(b)
    r = b.copy()
    d = r.copy()

    rdot1 = np.dot(r, r)

    b_norm = np.linalg.norm(b)
    if b_norm == 0:
        b_norm = 1.0

    if save_directions:
        D = np.zeros((b.size, J))
        KD = np.zeros((b.size, J))

    for i in range(J):
        v = np.asarray(matmul(d), dtype=float).reshape(-1)

        if v.shape != d.shape:
            raise ValueError("matmul must return a vector with the same shape as b.")

        if save_directions and reorthogonalize and i > 0:
            dKd = np.sum(D[:, :i] * KD[:, :i], axis=0)
            coeffs = (D[:, :i].T @ v) / dKd

            d = d - D[:, :i] @ coeffs
            v = v - KD[:, :i] @ coeffs

            dKd = np.sum(D[:, :i] * KD[:, :i], axis=0)
            coeffs = (D[:, :i].T @ v) / dKd

            d = d - D[:, :i] @ coeffs
            v = v - KD[:, :i] @ coeffs

        dot_dv = np.dot(d, v)

        if dot_dv <= 0:
            raise RuntimeError("Non-positive d.T @ A @ d encountered.")

        alpha = np.dot(r, d) / dot_dv

        if save_directions:
            D[:, i] = d
            KD[:, i] = v

        c += alpha * d
        r -= alpha * v

        if np.linalg.norm(r) / b_norm < tol:
            if save_directions:
                return c, D[:, : i + 1], KD[:, : i + 1]

            return c

        rdot2 = np.dot(r, r)
        beta = rdot2 / rdot1

        d = r + beta * d
        rdot1 = rdot2

    if save_directions:
        return c, D, KD

    return c
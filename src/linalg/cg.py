from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .reorthogonalization import ReorthogonalizationRule


def cg(
    matmul: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    J: int,
    tol: float = 1e-6,
    save_directions: bool = False,
    reorthogonalization_rule: ReorthogonalizationRule | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = np.asarray(b, dtype=float).reshape(-1)

    if J <= 0:
        raise ValueError("J must be positive.")

    c = np.zeros_like(b)
    r = b.copy()
    d = r.copy()

    initial_direction_norm = np.linalg.norm(d)
    rdot1 = np.dot(r, r)

    b_norm = np.linalg.norm(b)
    if b_norm == 0:
        b_norm = 1.0
    
    if reorthogonalization_rule is None:
        reorthogonalization_rule = ReorthogonalizationRule(mode="never")
    
    if reorthogonalization_rule.mode != "never":
        save_directions = True

    if save_directions:
        D = np.zeros((b.size, J))
        KD = np.zeros((b.size, J))

    for i in range(J):
        v = np.asarray(matmul(d), dtype=float).reshape(-1)

        if v.shape != d.shape:
            raise ValueError("matmul must return a vector with the same shape as b.")
        
        dot_dv = np.dot(d, v)

        if reorthogonalization_rule(i=i, d=d, v=v, dot_dv=dot_dv, initial_direction_norm=initial_direction_norm):
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


def cg_store_lanczos_basis(
    matmul: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    J: int,
    tol: float = 1e-6,
    reorthogonalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    Q = np.zeros((b.size, J))
    KQ = np.zeros((b.size, J))

    Kd_prev = np.zeros_like(b)
    beta = 0.0

    for i in range(J):
        Kd = np.asarray(matmul(d), dtype=float).reshape(-1)

        if Kd.shape != d.shape:
            raise ValueError("matmul must return a vector with the same shape as b.")

        dot_dv = np.dot(d, Kd)

        if dot_dv <= 0:
            raise RuntimeError("Non-positive d.T @ A @ d encountered.")

        r_norm = np.linalg.norm(r)

        if r_norm == 0:
            Q_final = Q[:, :i]
            KQ_final = KQ[:, :i]
            return c, Q_final, KQ_final

        q = ((-1) ** i) * r / r_norm

        if i == 0:
            Kq = ((-1) ** i) * Kd / r_norm
        else:
            Kq = ((-1) ** i) * (Kd - beta * Kd_prev) / r_norm

        if reorthogonalize and i > 0:
            for _ in range(2):
                coeffs = Q[:, :i].T @ q
                q = q - Q[:, :i] @ coeffs
                Kq = Kq - KQ[:, :i] @ coeffs

            q_norm = np.linalg.norm(q)

            if q_norm < 1e-10:
                Q_final = Q[:, :i]
                KQ_final = KQ[:, :i]

                return c, Q_final, KQ_final

            q = q / q_norm
            Kq = Kq / q_norm

        Q[:, i] = q
        Kq = np.asarray(matmul(q), dtype=float).reshape(-1)
        KQ[:, i] = Kq

        alpha = rdot1 / dot_dv

        c += alpha * d
        r -= alpha * Kd

        r_norm_new = np.linalg.norm(r)

        if r_norm_new / b_norm < tol:
            Q_final = Q[:, : i + 1]
            KQ_final = KQ[:, : i + 1]
            return c, Q_final, KQ_final

        rdot2 = r_norm_new**2
        beta = rdot2 / rdot1

        Kd_prev = Kd

        d = r + beta * d
        rdot1 = rdot2

    return c, Q, KQ
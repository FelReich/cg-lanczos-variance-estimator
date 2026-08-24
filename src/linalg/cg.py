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


def cg_with_basis_extension(
    matmul: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    J: int,
    tol: float = 1e-6,
    save_directions: bool = False,
    reorthogonalization_rule: ReorthogonalizationRule | None = None,
    min_iter: int = 1,
    repair_tol: float = 1e-12,
    denominator_tol: float = 1e-14,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = np.asarray(b, dtype=float).reshape(-1)

    if J <= 0:
        raise ValueError("J must be positive.")

    if min_iter <= 0:
        raise ValueError("min_iter must be positive.")

    min_iter = min(min_iter, J)

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

    if reorthogonalization_rule.mode != "never" or min_iter > 1:
        save_directions = True

    if save_directions:
        D = np.zeros((b.size, J))
        KD = np.zeros((b.size, J))

    def k_reorthogonalize(d_candidate, v_candidate, i):
        if i == 0:
            return d_candidate, v_candidate

        D_i = D[:, :i]
        KD_i = KD[:, :i]

        for _ in range(2):
            dKd = np.sum(D_i * KD_i, axis=0)
            scale = max(np.max(np.abs(dKd)), 1.0)
            mask = dKd > denominator_tol * scale

            if not np.any(mask):
                break

            coeffs = (D_i[:, mask].T @ v_candidate) / dKd[mask]

            d_candidate = d_candidate - D_i[:, mask] @ coeffs
            v_candidate = v_candidate - KD_i[:, mask] @ coeffs

        return d_candidate, v_candidate

    solve_converged = False
    last_basis_direction = None

    for i in range(J):
        if solve_converged:
            d = matmul(last_basis_direction)
            v = matmul(d)

            d, v = k_reorthogonalize(d, v, i)
            dot_dv = np.dot(d, v)

            if dot_dv <= repair_tol:
                if save_directions:
                    return c, D[:, :i], KD[:, :i]
                return c

            D[:, i] = d
            KD[:, i] = v
            last_basis_direction = d.copy()

            if i + 1 >= min_iter:
                return c, D[:, : i + 1], KD[:, : i + 1]

            continue

        v = np.asarray(matmul(d), dtype=float).reshape(-1)

        if v.shape != d.shape:
            raise ValueError("matmul must return a vector with the same shape as b.")

        dot_dv = np.dot(d, v)

        if reorthogonalization_rule(
            i=i,
            d=d,
            v=v,
            dot_dv=dot_dv,
            initial_direction_norm=initial_direction_norm,
        ):
            d, v = k_reorthogonalize(d, v, i)
            dot_dv = np.dot(d, v)

        if dot_dv <= repair_tol:
            solve_converged = True

            if i >= min_iter:
                if save_directions:
                    return c, D[:, :i], KD[:, :i]
                return c

            if i == 0:
                raise RuntimeError("Cannot repair before any basis direction has been stored.")

            last_basis_direction = D[:, i - 1].copy()
            continue

        alpha = np.dot(r, d) / dot_dv

        if save_directions:
            D[:, i] = d
            KD[:, i] = v
            last_basis_direction = d.copy()

        c += alpha * d
        r -= alpha * v

        if np.linalg.norm(r) / b_norm < tol:
            solve_converged = True

            if i + 1 >= min_iter:
                if save_directions:
                    return c, D[:, : i + 1], KD[:, : i + 1]
                return c

        rdot2 = np.dot(r, r)

        if rdot1 == 0:
            solve_converged = True
            continue

        beta = rdot2 / rdot1

        d = r + beta * d
        rdot1 = rdot2

    if save_directions:
        return c, D, KD

    return c


def cg_store_residuals(
    matmul: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    J: int,
    tol: float = 1e-6,
    reorthogonalize: bool = True,
    save_residuals=True,
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

    for i in range(J):
        r_norm = np.linalg.norm(r)

        if r_norm == 0:
            break

        q = (-1) ** i * r / r_norm

        if reorthogonalize and i > 0:
            for _ in range(2):
                coeffs = Q[:, :i].T @ q
                q = q - Q[:, :i] @ coeffs

            q_norm = np.linalg.norm(q)

            if q_norm == 0:
                break

            q = q / q_norm

        Kq = np.asarray(matmul(q), dtype=float).reshape(-1)

        Q[:, i] = q
        KQ[:, i] = Kq

        v = np.asarray(matmul(d), dtype=float).reshape(-1)

        if v.shape != d.shape:
            raise ValueError("matmul must return a vector with the same shape as b.")

        dot_dv = np.dot(d, v)

        if dot_dv <= 0:
            break

        alpha = rdot1 / dot_dv

        c += alpha * d
        r -= alpha * v

        if np.linalg.norm(r) / b_norm < tol:
            i += 1
            break

        rdot2 = np.dot(r, r)
        beta = rdot2 / rdot1

        d = r + beta * d
        rdot1 = rdot2

    Q = Q[:, :i]
    KQ = KQ[:, :i]

    T = Q.T @ KQ
    T = 0.5 * (T + T.T)

    return c, Q, T
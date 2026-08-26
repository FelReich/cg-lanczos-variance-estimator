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
    """Solves a symmetric positive definite linear system with conjugate gradients.

    Optionally stores the CG search directions and their matrix-vector products.
    These quantities are used later for CG-based covariance corrections.

    :param matmul: Function implementing multiplication by the system matrix.
    :param numpy.ndarray b: Right-hand side vector.
    :param int J: Maximum number of CG iterations.
    :param float tol: Relative residual tolerance for early stopping. (Default: `1e-6`.)
    :param bool save_directions: If True, return stored search directions and matrix-vector products. (Default: False.)
    :param ReorthogonalizationRule reorthogonalization_rule: Optional rule deciding when to reorthogonalize search directions.
    :return: Approximate solution, and optionally stored directions `D` and products `KD`.
    """
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
        #Reorthogonalization requires access to previous directions
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
    """Runs CG while storing normalized residuals as a Lanczos basis.

    In exact arithmetic, normalized CG residuals form the Lanczos basis associated
    with the same Krylov subspace, up to alternating signs. This function stores
    that basis and the corresponding matrix-vector products so that the projected
    matrix can later be formed as `Q.T @ KQ`.

    :param matmul: Function implementing multiplication by the system matrix.
    :param numpy.ndarray b: Right-hand side vector.
    :param int J: Maximum number of CG iterations.
    :param float tol: Relative residual tolerance for early stopping. (Default: `1e-6`.)
    :param bool reorthogonalize: If True, reorthogonalize stored residual basis vectors. (Default: True.)
    :return: Approximate solution, residual-based Lanczos basis `Q`, and products `KQ`.
    """
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

        #Normalized CG residuals form the Lanczos basis up to alternating signs
        q = ((-1) ** i) * r / r_norm

        if reorthogonalize and i > 0:
            for _ in range(2):
                coeffs = Q[:, :i].T @ q
                q = q - Q[:, :i] @ coeffs

            q_norm = np.linalg.norm(q)

            if q_norm < 1e-10:
                Q_final = Q[:, :i]
                KQ_final = KQ[:, :i]

                return c, Q_final, KQ_final

            q = q / q_norm

        Q[:, i] = q
        KQ[:, i] = np.asarray(matmul(q), dtype=float).reshape(-1)

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

        d = r + beta * d
        rdot1 = rdot2

    return c, Q, KQ
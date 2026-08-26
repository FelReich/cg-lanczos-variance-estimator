from __future__ import annotations

import numpy as np


def relative_error(approx: np.ndarray, exact: np.ndarray) -> float:
    #Computes the relative Frobenius error between an approximation and a reference
    approx = np.asarray(approx, dtype=float)
    exact = np.asarray(exact, dtype=float)

    denom = np.linalg.norm(exact)
    if denom == 0:
        return float(np.linalg.norm(approx - exact))

    return float(np.linalg.norm(approx - exact) / denom)


def conjugacy_error(D: np.ndarray, KD: np.ndarray) -> float:
    #Measures how far the stored directions are from being K-conjugate
    D = np.asarray(D, dtype=float)
    KD = np.asarray(KD, dtype=float)

    if D.shape != KD.shape:
        raise ValueError("D and KD must have the same shape.")

    G = D.T @ KD
    diag_G = np.diag(np.diag(G))

    denom = np.linalg.norm(diag_G)
    if denom == 0:
        return float(np.linalg.norm(G - diag_G))

    return float(np.linalg.norm(G - diag_G) / denom)


def projection_residual(D: np.ndarray, k: np.ndarray) -> float:
    #Measures how much of k lies outside the span of the stored directions
    D = np.asarray(D, dtype=float)
    k = np.asarray(k, dtype=float)

    if D.ndim != 2:
        raise ValueError("D must be a two-dimensional array.")

    if k.ndim == 1:
        k = k.reshape(-1, 1)

    if k.ndim != 2:
        raise ValueError("k must be one- or two-dimensional.")

    if D.shape[0] != k.shape[0]:
        raise ValueError("D and k must have the same first dimension.")

    Q, _ = np.linalg.qr(D, mode="reduced")
    residual = k - Q @ (Q.T @ k)

    denom = np.linalg.norm(k)
    if denom == 0:
        return float(np.linalg.norm(residual))

    return float(np.linalg.norm(residual) / denom)
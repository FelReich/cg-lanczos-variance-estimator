from __future__ import annotations

import numpy as np


def exact_correction(K_noise: np.ndarray, k: np.ndarray) -> np.ndarray:
    #Computes the exact covariance correction `k.T @ K_noise^{-1} @ k`
    return k.T @ np.linalg.solve(K_noise, k)


def cg_cholesky_correction(
    D: np.ndarray,
    KD: np.ndarray,
    k: np.ndarray,
    jitter: float = 1e-6,
    coordinate_jitter: bool = True,
) -> np.ndarray:
    #Computes the CG covariance correction in the raw search-direction basis
    G = D.T @ KD

    if coordinate_jitter:
        G = G + jitter * np.eye(G.shape[0])
    else:
        G = G + jitter * (D.T @ D)

    G = 0.5 * (G + G.T)

    L = np.linalg.cholesky(G)
    Z = np.linalg.solve(L, D.T @ k)

    return Z.T @ Z


def cg_qr_correction(
    D: np.ndarray,
    KD: np.ndarray,
    k: np.ndarray,
    jitter: float = 1e-6,
) -> np.ndarray:
    #Computes the CG covariance correction after QR-orthogonalizing the directions
    Q, R = np.linalg.qr(D, mode="reduced")

    T = np.linalg.solve(R.T, (Q.T @ KD).T).T
    T = 0.5 * (T + T.T)

    L = np.linalg.cholesky(T + jitter * np.eye(T.shape[0]))
    Z = np.linalg.solve(L, Q.T @ k)

    return Z.T @ Z


def love_correction(
    Q: np.ndarray,
    T: np.ndarray,
    k: np.ndarray,
    jitter: float = 1e-6,
) -> np.ndarray:
    #Computes a LOVE-style covariance correction from a Lanczos basis and projected matrix
    T_jittered = T + jitter * np.eye(T.shape[0])

    L = np.linalg.cholesky(T_jittered)
    Z = np.linalg.solve(L, Q.T @ k)

    return Z.T @ Z
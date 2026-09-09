from __future__ import annotations

import torch


def exact_correction(K_noise: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # Computes the exact covariance reduction term k.T @ K_noise^{-1} @ k.
    return k.T @ torch.linalg.solve(K_noise, k)


def love_correction(
    Q: torch.Tensor,
    T: torch.Tensor,
    k: torch.Tensor,
    jitter: float = 1e-6,
) -> torch.Tensor:
    # Computes a LOVE-style covariance reduction term from a basis and projected matrix.
    T_jittered = T + jitter * torch.eye(T.shape[0], dtype=T.dtype, device=T.device)

    L = torch.linalg.cholesky(T_jittered)
    Z = torch.linalg.solve_triangular(L, Q.T @ k, upper=False)

    return Z.T @ Z
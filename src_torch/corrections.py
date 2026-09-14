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
    max_tries: int = 5,
) -> torch.Tensor:
    # Computes a LOVE-style covariance reduction term from a basis and projected matrix.
    eye = torch.eye(T.shape[0], dtype=T.dtype, device=T.device)

    for i in range(max_tries):
        jitter_i = jitter * (10**i)
        T_jittered = T + jitter_i * eye

        L, info = torch.linalg.cholesky_ex(T_jittered)
        
        if not torch.any(info):
            Z = torch.linalg.solve_triangular(L, Q.T @ k, upper=False)

            return Z.T @ Z

    raise RuntimeError(
        f"Cholesky failed after {max_tries} tries. "
        f"Last jitter was {jitter * (10.0 ** (max_tries - 1)):.1e}."
    )